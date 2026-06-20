from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("feature_pipeline")

@dataclass(frozen=True)
class ESIBBOConfig:
    pop_size: int = 10
    epochs: int = 25
    elite_size: int = 2
    patience: int = 6

    alpha_max: float = 0.9
    alpha_min: float = 0.2
    beta_min: float = 0.1
    beta_max: float = 0.8

    levy_probability: float = 0.25
    mutation_probability: float = 0.35
    mu_max: float = 0.12
    chaotic_r: float = 3.99

    dice_weight: float = 0.40
    iou_weight: float = 0.35
    boundary_weight: float = 0.20
    complexity_weight: float = 0.05

@dataclass
class ESIBBOResult:
    best_solution: np.ndarray
    best_fitness: float
    best_params: Dict[str, object]
    history: List[float]

def default_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized post-processing vector:
        threshold, morph_kernel, close_iter, open_iter, min_area_ratio
    """
    lb = np.array([0.20, 3, 0, 0, 0.001], dtype=np.float64)
    ub = np.array([0.80, 9, 3, 2, 0.080], dtype=np.float64)
    return lb, ub

def decode_solution(x: np.ndarray) -> Dict[str, object]:
    kernel = int(round(float(x[1])))
    if kernel % 2 == 0:
        kernel += 1

    return {
        "threshold": float(np.clip(x[0], 0.20, 0.80)),
        "morph_kernel": int(np.clip(kernel, 3, 9)),
        "close_iter": int(np.clip(round(float(x[2])), 0, 3)),
        "open_iter": int(np.clip(round(float(x[3])), 0, 2)),
        "min_area_ratio": float(np.clip(x[4], 0.001, 0.080)),
    }

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    inter = np.sum(y_true * y_pred)
    return float((2.0 * inter + eps) / (np.sum(y_true) + np.sum(y_pred) + eps))

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return float((inter + eps) / (union + eps))

def boundary_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_edge = cv2.Canny((y_true * 255).astype(np.uint8), 50, 150) > 0
    pred_edge = cv2.Canny((y_pred * 255).astype(np.uint8), 50, 150) > 0
    return 1.0 - dice_score(true_edge.astype(np.uint8), pred_edge.astype(np.uint8))

def postprocess_mask(prob_mask: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    if prob_mask.ndim == 3:
        prob_mask = prob_mask[..., 0]

    mask = (prob_mask >= float(params["threshold"])).astype(np.uint8)

    k = int(params["morph_kernel"])
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)

    close_iter = int(params.get("close_iter", 0))
    open_iter = int(params.get("open_iter", 0))

    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    min_area_ratio = float(params.get("min_area_ratio", 0.005))
    min_area = int(mask.size * min_area_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label_idx] = 1

    return clean.astype(np.uint8)

def complexity_penalty(params: Dict[str, object]) -> float:
    return float(
        0.4 * ((int(params["morph_kernel"]) - 3) / 6)
        + 0.3 * (int(params["close_iter"]) / 3)
        + 0.2 * (int(params["open_iter"]) / 2)
        + 0.1 * (float(params["min_area_ratio"]) / 0.080)
    )

def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    sigma = 0.6966
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-8)

def evaluate_candidate(
    x: np.ndarray,
    prob_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    config: ESIBBOConfig,
) -> float:
    params = decode_solution(x)

    dice_vals = []
    iou_vals = []
    boundary_vals = []

    for prob, gt in zip(prob_masks, gt_masks):
        pred = postprocess_mask(prob, params)
        gt = (gt > 0).astype(np.uint8)

        dice_vals.append(dice_score(gt, pred))
        iou_vals.append(iou_score(gt, pred))
        boundary_vals.append(boundary_loss(gt, pred))

    dice = float(np.mean(dice_vals))
    iou = float(np.mean(iou_vals))
    boundary = float(np.mean(boundary_vals))
    complexity = complexity_penalty(params)

    return float(
        config.dice_weight * (1.0 - dice)
        + config.iou_weight * (1.0 - iou)
        + config.boundary_weight * boundary
        + config.complexity_weight * complexity
    )

def opposition_init(lb: np.ndarray, ub: np.ndarray, pop_size: int, objective) -> np.ndarray:
    pop = np.random.uniform(lb, ub, size=(pop_size, len(lb)))
    opp = lb + ub - pop

    selected = []
    for p, o in zip(pop, opp):
        selected.append(p if objective(p) <= objective(o) else o)

    return np.asarray(selected, dtype=np.float64)

def e_si_bbo_postprocess(
    prob_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    config: Optional[ESIBBOConfig] = None,
) -> ESIBBOResult:
    if len(prob_masks) != len(gt_masks):
        raise ValueError("prob_masks and gt_masks must have equal length")

    if not prob_masks:
        raise ValueError("prob_masks cannot be empty")

    config = config or ESIBBOConfig()
    lb, ub = default_bounds()
    dim = len(lb)

    def objective(x: np.ndarray) -> float:
        return evaluate_candidate(np.clip(x, lb, ub), prob_masks, gt_masks, config)

    population = opposition_init(lb, ub, config.pop_size, objective)
    fitness = np.asarray([objective(p) for p in population], dtype=np.float64)

    best_idx = int(np.argmin(fitness))
    best = population[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]

    chaotic_x = np.random.uniform(0.2, 0.8)
    no_improve = 0

    for epoch in range(config.epochs):
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]

        elites = population[: config.elite_size].copy()
        current_best = population[0].copy()
        worst = population[-1].copy()

        alpha_t = config.alpha_max - (epoch / config.epochs) * (
            config.alpha_max - config.alpha_min
        )
        beta_t = config.beta_min + (epoch / config.epochs) * (
            config.beta_max - config.beta_min
        )

        chaotic_x = config.chaotic_r * chaotic_x * (1.0 - chaotic_x)

        new_pop = population.copy()

        for i in range(config.elite_size, config.pop_size):
            candidate = population[i].copy()

            candidate += alpha_t * chaotic_x * (current_best - candidate)
            candidate += beta_t * np.random.rand(dim) * (candidate - worst)

            if np.random.rand() < config.levy_probability:
                candidate += levy_flight(dim) * (current_best - candidate)

            if np.random.rand() < config.mutation_probability:
                rank_ratio = (i + 1) / config.pop_size
                mu_i = config.mu_max * rank_ratio
                candidate += mu_i * np.random.randn(dim) * (ub - lb)

            new_pop[i] = np.clip(candidate, lb, ub)

        new_pop[: config.elite_size] = elites
        new_fit = np.asarray([objective(p) for p in new_pop], dtype=np.float64)

        improved = new_fit < fitness
        population[improved] = new_pop[improved]
        fitness[improved] = new_fit[improved]

        epoch_best_idx = int(np.argmin(fitness))
        epoch_best = float(fitness[epoch_best_idx])

        if epoch_best < best_fit:
            best_fit = epoch_best
            best = population[epoch_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_fit)

        logger.info(
            "E-SI-BBO epoch=%s best_fitness=%.6f params=%s",
            epoch + 1,
            best_fit,
            decode_solution(best),
        )

        if no_improve >= config.patience:
            logger.info("E-SI-BBO early stopping at epoch=%s", epoch + 1)
            break

    return ESIBBOResult(
        best_solution=best,
        best_fitness=best_fit,
        best_params=decode_solution(best),
        history=history,
    )