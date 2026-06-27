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

    # Phase controls
    theta_max: float = 0.30          # pedal scent marking
    theta_min: float = 0.05

    delta_max: float = 1.00          # careful stepping
    delta_min: float = 0.10

    beta_min: float = 0.10
    beta_max: float = 0.80

    omega_scale: float = 2.0         # twisting feet
    sniff_gamma: float = 0.35        # peer interaction

    levy_probability: float = 0.25
    levy_beta: float = 1.5

    # Fitness weights
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
    # threshold, morph_kernel, close_iter, open_iter, min_area_ratio
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


def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    sigma = 0.6966
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-8)


def boundary_handle(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(x, lb, ub)


def opposition_initialization(lb: np.ndarray, ub: np.ndarray, pop_size: int, objective) -> np.ndarray:
    population = np.random.uniform(lb, ub, size=(pop_size, len(lb)))
    opposite = lb + ub - population

    selected = []
    for p, o in zip(population, opposite):
        selected.append(p if objective(p) <= objective(o) else o)

    return np.asarray(selected, dtype=np.float64)


def pedal_scent_marking(
    population: np.ndarray,
    epoch: int,
    config: ESIBBOConfig,
) -> np.ndarray:
    theta = config.theta_max - (epoch / config.epochs) * (
        config.theta_max - config.theta_min
    )

    rand = np.random.rand(*population.shape)
    return population - theta * rand * population


def careful_stepping(
    population: np.ndarray,
    best: np.ndarray,
    worst: np.ndarray,
    epoch: int,
    config: ESIBBOConfig,
) -> np.ndarray:
    delta_t = config.delta_max - (config.delta_max - config.delta_min) * (
        epoch / config.epochs
    )

    beta_t = config.beta_min + (config.beta_max - config.beta_min) * (
        epoch / config.epochs
    )

    updated = population.copy()

    for i in range(population.shape[0]):
        L_k = round(1.0 + beta_t)
        F_k = np.random.uniform(0.5, 1.0)

        updated[i] = population[i] + F_k * delta_t * (
            best - L_k * worst
        )

    return updated


def twisting_feet(
    population: np.ndarray,
    best: np.ndarray,
    worst: np.ndarray,
    epoch: int,
    config: ESIBBOConfig,
) -> np.ndarray:
    updated = population.copy()

    for i in range(population.shape[0]):
        theta_k = np.random.uniform(0.0, 1.0)
        y_k = np.random.uniform(0.0, 1.0)

        omega_k = config.omega_scale * np.pi * theta_k * y_k

        updated[i] = (
            population[i]
            + omega_k * (best - population[i])
            - omega_k * (worst - np.abs(population[i]))
        )

    return updated


def sniffing_peer_interaction(
    population: np.ndarray,
    fitness: np.ndarray,
    config: ESIBBOConfig,
) -> np.ndarray:
    updated = population.copy()
    n = population.shape[0]

    for m in range(n):
        candidates = [idx for idx in range(n) if idx != m]
        peer = np.random.choice(candidates)

        if fitness[m] < fitness[peer]:
            updated[m] = population[m] + config.sniff_gamma * np.random.rand() * (
                population[m] - population[peer]
            )
        else:
            updated[m] = population[m] + config.sniff_gamma * np.random.rand() * (
                population[peer] - population[m]
            )

    return updated


def levy_perturbation(
    population: np.ndarray,
    best: np.ndarray,
    config: ESIBBOConfig,
) -> np.ndarray:
    updated = population.copy()
    dim = population.shape[1]

    for i in range(population.shape[0]):
        if np.random.rand() < config.levy_probability:
            updated[i] = population[i] + levy_flight(dim, config.levy_beta) * (
                best - population[i]
            )

    return updated


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

    def objective(x: np.ndarray) -> float:
        return evaluate_candidate(boundary_handle(x, lb, ub), prob_masks, gt_masks, config)

    # Phase 0: opposition-based initialization
    population = opposition_initialization(lb, ub, config.pop_size, objective)
    fitness = np.asarray([objective(p) for p in population], dtype=np.float64)

    best_idx = int(np.argmin(fitness))
    best = population[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]
    no_improve = 0

    for epoch in range(config.epochs):
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]

        elites = population[: config.elite_size].copy()

        current_best = population[0].copy()
        worst = population[-1].copy()

        # Phase 1 / 2 / 3 based on iteration stage
        if epoch < config.epochs / 3:
            candidate_population = pedal_scent_marking(
                population,
                epoch,
                config,
            )
        elif epoch < 2 * config.epochs / 3:
            candidate_population = careful_stepping(
                population,
                current_best,
                worst,
                epoch,
                config,
            )
        else:
            candidate_population = twisting_feet(
                population,
                current_best,
                worst,
                epoch,
                config,
            )

        # Phase 4: sniffing/random peer interaction
        candidate_population = sniffing_peer_interaction(
            candidate_population,
            fitness,
            config,
        )

        # Phase 5: Levy perturbation
        candidate_population = levy_perturbation(
            candidate_population,
            current_best,
            config,
        )

        # Phase 6: boundary handling
        candidate_population = np.asarray(
            [boundary_handle(x, lb, ub) for x in candidate_population],
            dtype=np.float64,
        )

        # Phase 7: elitism
        candidate_population[: config.elite_size] = elites

        candidate_fitness = np.asarray(
            [objective(p) for p in candidate_population],
            dtype=np.float64,
        )

        improved = candidate_fitness < fitness
        population[improved] = candidate_population[improved]
        fitness[improved] = candidate_fitness[improved]

        epoch_best_idx = int(np.argmin(fitness))
        epoch_best_fit = float(fitness[epoch_best_idx])

        if epoch_best_fit < best_fit:
            best_fit = epoch_best_fit
            best = population[epoch_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_fit)

        logger.info(
            "E-SI-BBO epoch=%s phase=%s best_fitness=%.6f params=%s",
            epoch + 1,
            (
                "pedal_scent_marking"
                if epoch < config.epochs / 3
                else "careful_stepping"
                if epoch < 2 * config.epochs / 3
                else "twisting_feet"
            ),
            best_fit,
            decode_solution(best),
        )

        # Phase 7: early stopping
        if no_improve >= config.patience:
            logger.info("E-SI-BBO early stopping at epoch=%s", epoch + 1)
            break

    return ESIBBOResult(
        best_solution=best,
        best_fitness=best_fit,
        best_params=decode_solution(best),
        history=history,
    )