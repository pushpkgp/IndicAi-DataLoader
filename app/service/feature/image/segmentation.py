from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import tensorflow as tf

from app.model.fcn import apply_mask
from app.optimization.sa_bbo import e_si_bbo_postprocess, postprocess_mask

logger = logging.getLogger("feature_pipeline")

def predict_probability_mask(
    image: np.ndarray,
    model: tf.keras.Model,
    input_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    resized = cv2.resize(image, input_size)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False).numpy()[0]

    if pred.ndim == 3:
        pred = pred[..., 0]

    return pred.astype(np.float32)

def load_mask(path: str, input_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = cv2.resize(mask, input_size)
    return (mask > 127).astype(np.uint8)

def calibrate_postprocessing_params(
    model: tf.keras.Model,
    reference_images: Sequence[str],
    reference_masks: Sequence[str],
    max_samples: int = 16,
) -> Dict[str, object]:
    prob_masks = []
    gt_masks = []

    for img_path, mask_path in zip(reference_images[:max_samples], reference_masks[:max_samples]):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue

        prob_masks.append(predict_probability_mask(image, model))
        gt_masks.append(load_mask(str(mask_path)))

    if not prob_masks:
        raise ValueError("No valid image-mask pairs found for E-SI-BBO calibration")

    result = e_si_bbo_postprocess(prob_masks=prob_masks, gt_masks=gt_masks)

    logger.info(
        "Best E-SI-BBO postprocess params=%s fitness=%s",
        result.best_params,
        result.best_fitness,
    )

    return result.best_params

def segment_image(
    image: np.ndarray,
    model: tf.keras.Model,
    postprocess_params: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    params = postprocess_params or {
        "threshold": 0.5,
        "morph_kernel": 5,
        "close_iter": 1,
        "open_iter": 1,
        "min_area_ratio": 0.005,
    }

    prob_mask = predict_probability_mask(image, model)
    binary_mask = postprocess_mask(prob_mask, params)

    return apply_mask(image, binary_mask, threshold=0.5)