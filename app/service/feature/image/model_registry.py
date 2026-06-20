from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from app.model.fcn import fcn_model

logger = logging.getLogger("feature_pipeline")

_SEG_MODEL: Optional[tf.keras.Model] = None
_POSTPROCESS_PARAMS: Optional[Dict[str, Any]] = None

POSTPROCESS_PATH = os.getenv(
    "OFCN_POSTPROCESS_PATH",
    "/models/best_postprocess.json",
)


def configure_tensorflow_runtime() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        logger.warning("TensorFlow runtime configuration skipped: %s", exc)


def _load_postprocess_params(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        logger.warning("Postprocess params not found at %s. Using defaults.", p)
        return {
            "threshold": 0.5,
            "morph_kernel": 5,
            "close_iter": 1,
            "open_iter": 1,
            "min_area_ratio": 0.005,
        }

    with p.open("r", encoding="utf-8") as f:
        params = json.load(f)

    return {
        "threshold": float(params.get("threshold", 0.5)),
        "morph_kernel": int(params.get("morph_kernel", 5)),
        "close_iter": int(params.get("close_iter", 1)),
        "open_iter": int(params.get("open_iter", 1)),
        "min_area_ratio": float(params.get("min_area_ratio", 0.005)),
    }


def load_segmentation_assets() -> Tuple[tf.keras.Model, Dict[str, Any]]:
    global _SEG_MODEL, _POSTPROCESS_PARAMS

    configure_tensorflow_runtime()

    if _SEG_MODEL is None:
        logger.info("Creating O-FCN segmentation model from local fcn_model()")
        _SEG_MODEL = fcn_model(
            input_shape=(256, 256, 3),
            dilation_rate=(1, 1),
            learning_rate=1e-4,
            num_filters=64,
            kernel_size=3,
            dropout_rate=0.2,
        )

    if _POSTPROCESS_PARAMS is None:
        _POSTPROCESS_PARAMS = _load_postprocess_params(POSTPROCESS_PATH)
        logger.info("Loaded postprocess params: %s", _POSTPROCESS_PARAMS)

    return _SEG_MODEL, _POSTPROCESS_PARAMS

def get_segmentation_assets() -> Tuple[tf.keras.Model, Dict[str, Any]]:
    return load_segmentation_assets()