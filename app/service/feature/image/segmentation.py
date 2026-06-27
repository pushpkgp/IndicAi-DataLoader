from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import tensorflow as tf

from app.model.fcn import apply_mask
from app.optimization.sa_bbo import e_si_bbo_postprocess, postprocess_mask

logger = logging.getLogger("feature_pipeline")


@dataclass
class SegmentationResult:
    segmented_image: np.ndarray
    binary_mask: np.ndarray
    probability_mask: np.ndarray


def _model_input_hw_c(model: tf.keras.Model) -> tuple[int, int, int]:
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = int(shape[1] or 256)
    width = int(shape[2] or 256)
    channels = int(shape[3] or 1)

    return height, width, channels


def _prepare_for_model(
    image: np.ndarray,
    model: tf.keras.Model,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, c = _model_input_hw_c(model)

    resized = cv2.resize(image, (w, h))

    if c == 1:
        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        x = resized.astype(np.float32)

        if x.max() > 1.0:
            x = x / 255.0

        x = np.expand_dims(x, axis=-1)

    elif c == 3:
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        x = resized.astype(np.float32)

        if x.max() > 1.0:
            x = x / 255.0

    else:
        raise ValueError(f"Unsupported model input channels: {c}")

    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32), (h, w)


def predict_probability_mask(
    image: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    x, _ = _prepare_for_model(image, model)

    pred = model(
        tf.convert_to_tensor(x, dtype=tf.float32),
        training=False,
    ).numpy()[0]

    if pred.ndim == 3:
        pred = pred[..., 0]

    return pred.astype(np.float32)


def load_mask(path: str, input_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = cv2.resize(mask, input_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def calibrate_postprocessing_params(
    model: tf.keras.Model,
    reference_images: Sequence[str],
    reference_masks: Sequence[str],
    max_samples: int = 16,
) -> Dict[str, object]:
    prob_masks = []
    gt_masks = []

    _, model_hw = _prepare_for_model(
        cv2.imread(str(reference_images[0])),
        model,
    )

    for img_path, mask_path in zip(
        reference_images[:max_samples],
        reference_masks[:max_samples],
    ):
        image = cv2.imread(str(img_path))

        if image is None:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue

        prob_masks.append(predict_probability_mask(image, model))
        gt_masks.append(load_mask(str(mask_path), input_size=model_hw))

    if not prob_masks:
        raise ValueError("No valid image-mask pairs found for E-SI-BBO calibration")

    result = e_si_bbo_postprocess(
        prob_masks=prob_masks,
        gt_masks=gt_masks,
    )

    logger.info(
        "Best E-SI-BBO postprocess params=%s fitness=%s",
        result.best_params,
        result.best_fitness,
    )

    return result.best_params


def segment_image_result(
    image: np.ndarray,
    model: tf.keras.Model,
    postprocess_params: Optional[Dict[str, object]] = None,
) -> SegmentationResult:
    params = postprocess_params or {
        "threshold": 0.5,
        "morph_kernel": 5,
        "close_iter": 1,
        "open_iter": 1,
        "min_area_ratio": 0.005,
    }

    original_h, original_w = image.shape[:2]

    prob_mask = predict_probability_mask(image, model)
    binary_mask = postprocess_mask(prob_mask, params)

    binary_mask_resized = cv2.resize(
        binary_mask.astype(np.uint8),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    segmented_image = apply_mask(
        image,
        binary_mask_resized,
        threshold=0.5,
    )

    return SegmentationResult(
        segmented_image=segmented_image,
        binary_mask=binary_mask_resized.astype(np.uint8),
        probability_mask=prob_mask,
    )


def segment_image(
    image: np.ndarray,
    model: tf.keras.Model,
    postprocess_params: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    return segment_image_result(
        image=image,
        model=model,
        postprocess_params=postprocess_params,
    ).segmented_image