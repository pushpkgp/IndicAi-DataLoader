from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

logger = logging.getLogger("feature_pipeline")


DEFAULT_BASE_DIR = "/data/features/segmentation/image"


def _default_postprocess_params() -> Dict[str, Any]:
    return {
        "threshold": 0.5,
        "morph_kernel": 5,
        "close_iter": 1,
        "open_iter": 1,
        "min_area_ratio": 0.005,
    }


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    logger.warning("Postprocess config not found: %s. Using defaults.", path)
    return _default_postprocess_params()


def _resolve_path(env_name: str, default_path: str) -> Path:
    return Path(os.getenv(env_name, default_path)).expanduser().resolve()


def _load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Segmentation model not found: {model_path}. "
            f"Set the correct environment variable or check model path."
        )

    logger.info("Loading segmentation model: %s", model_path)
    return tf.keras.models.load_model(str(model_path), compile=False)


def load_segmentation_assets() -> Dict[str, Dict[str, Any]]:
    base_dir = Path(
        os.getenv("SEGMENTATION_MODEL_DIR", DEFAULT_BASE_DIR)
    ).expanduser().resolve()

    ct_model_path = _resolve_path(
        "CT_SEGMENTATION_MODEL_PATH",
        str(base_dir / "best_ct.keras"),
    )

    cxr_model_path = _resolve_path(
        "CXR_SEGMENTATION_MODEL_PATH",
        str(base_dir / "best_cxr.keras"),
    )

    ct_postprocess_path = _resolve_path(
        "CT_POSTPROCESS_PARAMS_PATH",
        str(base_dir / "postprocess_ct.json"),
    )

    cxr_postprocess_path = _resolve_path(
        "CXR_POSTPROCESS_PARAMS_PATH",
        str(base_dir / "postprocess_cxr.json"),
    )

    assets = {
        "ct": {
            "model": _load_model(ct_model_path),
            "postprocess_params": _load_json_if_exists(ct_postprocess_path),
            "model_path": str(ct_model_path),
            "postprocess_path": str(ct_postprocess_path),
        },
        "cxr": {
            "model": _load_model(cxr_model_path),
            "postprocess_params": _load_json_if_exists(cxr_postprocess_path),
            "model_path": str(cxr_model_path),
            "postprocess_path": str(cxr_postprocess_path),
        },
    }

    logger.info(
        "Loaded segmentation assets. CT=%s, CXR=%s",
        ct_model_path,
        cxr_model_path,
    )

    return assets


def get_segmentation_assets() -> Dict[str, Dict[str, Any]]:
    return load_segmentation_assets()