from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder

from app.metadata_cache import get_metadata_files
from app.model.densnet import get_densenet_extractor_model
from app.service.feature.extractor_factory import Factory
from app.service.feature.image.model_registry import get_segmentation_assets

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
pd.options.mode.copy_on_write = True

logger = logging.getLogger("feature_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FILENAME_PATTERN = re.compile(
    r"^(?P<modality>[^_]+)_"
    r"(?P<bodypart>[^_]+)_"
    r"(?P<modalitytype>[^_]+)_"
    r"(?P<split>[^_]+)_"
    r"(?P<category>.+)$"
)


def parse_meta_filename(filename: str) -> Tuple[str, str, str, str, str]:
    stem = Path(filename).stem
    match = FILENAME_PATTERN.match(stem)

    if not match:
        raise ValueError(f"Invalid metadata filename format: {filename}")

    g = match.groupdict()
    return (
        g["modality"],
        g["bodypart"],
        g["modalitytype"],
        g["split"],
        g["category"],
    )


def resolve_output_dir(default_dir: str = "/data/features/image-segmentation") -> Path:
    """
    Works both in Kubernetes and local Mac.

    Priority:
    1. FEATURE_OUTPUT_DIR env var
    2. /data/features/image-segmentation if writable
    3. local fallback under ~/IndicAI/data/features/image
    """
    env_dir = os.getenv("FEATURE_OUTPUT_DIR")
    if env_dir:
        out = Path(env_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    try:
        out = Path(default_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
    except OSError:
        local_out = Path.home() / "IndicAI" / "data" / "features" / "image"
        local_out.mkdir(parents=True, exist_ok=True)
        logger.warning("Using local output path because /data is not writable: %s", local_out)
        return local_out


def infer_modality_type(row: Any, metadata_file: Optional[Path] = None) -> str:
    """
    Detects whether this sample should use CT or CXR segmentation model.
    """
    candidates = []

    for attr in ["modality_type", "modalitytype", "category", "body_part", "bodypart", "modality"]:
        if hasattr(row, attr):
            val = getattr(row, attr)
            if val is not None:
                candidates.append(str(val).lower())

    if hasattr(row, "filepath"):
        candidates.append(str(row.filepath).lower())

    if metadata_file is not None:
        candidates.append(metadata_file.name.lower())

    joined = " ".join(candidates)

    if "ct" in joined or "cts" in joined or "computed" in joined:
        return "ct"

    if "x-ray" in joined or "xray" in joined or "cxr" in joined or "chest_x" in joined:
        return "cxr"

    return "cxr"


def get_segmentation_asset_for_row(
    segmentation_assets: Dict[str, Dict[str, object]],
    row: Any,
    metadata_file: Optional[Path] = None,
) -> Tuple[object, Dict[str, object], str]:
    modality_type = infer_modality_type(row, metadata_file)

    if modality_type == "ct":
        asset = segmentation_assets.get("ct")
    else:
        asset = segmentation_assets.get("cxr")

    if asset is None:
        raise ValueError(f"No segmentation asset found for modality_type={modality_type}")

    return asset["model"], asset["postprocess_params"], modality_type


def build_global_label_mapping(
    csv_files: Iterable[Path],
    label_map_path: Path,
):
    categories = set()

    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype=str)

            if "category" in df.columns:
                clean = df["category"].dropna().str.strip()
            elif "label" in df.columns:
                clean = df["label"].dropna().str.strip()
            else:
                logger.warning("Skipping label mapping for %s: no category/label column", f.name)
                continue

            clean = clean[clean.str.lower() != "category"]
            categories.update(clean.unique())

        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)

    if not categories:
        raise ValueError("No categories found in metadata files")

    le = LabelEncoder()
    le.fit(sorted(categories))

    mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}

    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    label_map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    logger.info("Label mapping created with %s classes at %s", len(mapping), label_map_path)

    return le, mapping, categories


def normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports both:
    filepath
    image_path
    """
    df = df.copy()

    if "filepath" not in df.columns and "image_path" in df.columns:
        df["filepath"] = df["image_path"]

    if "modality" not in df.columns:
        df["modality"] = "image"

    return df


async def extract_features_parallel(
    df: pd.DataFrame,
    extractor_func: Callable,
    segmentation_assets: Dict[str, Dict[str, object]],
    output_dir: Path,
    prefix: str,
    deep_feature_extraction_model,
    metadata_file: Optional[Path] = None,
    shard_size: int = 1000,
) -> List[Path]:
    features = []
    shard_files: List[Path] = []
    shard_idx = 0

    modality_counter = {"ct": 0, "cxr": 0, "failed": 0}

    for i, row in enumerate(df.itertuples(index=False), 1):
        filepath = str(row.filepath)
        extractor_modality = getattr(row, "modality", "image")

        if not os.path.exists(filepath):
            logger.warning("Skipping missing file: %s", filepath)
            continue

        try:
            segmentation_model, postprocess_params, modality_type = get_segmentation_asset_for_row(
                segmentation_assets=segmentation_assets,
                row=row,
                metadata_file=metadata_file,
            )

            modality_counter[modality_type] = modality_counter.get(modality_type, 0) + 1

            feature = extractor_func(
                extractor_modality,
                filepath,
                segmentation_model,
                postprocess_params,
                deep_feature_extraction_model,
            )

            if feature is None:
                continue

            feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)

            if not np.isfinite(feature).all():
                logger.warning("Skipping non-finite feature for %s", filepath)
                continue

            features.append(feature)

            if len(features) >= shard_size:
                shard_array = np.vstack(features)
                shard_file = output_dir / f"{prefix}_shard_{shard_idx}.npy"
                np.save(shard_file, shard_array.astype(np.float32))
                shard_files.append(shard_file)

                logger.info("Saved shard %s with shape %s", shard_file, shard_array.shape)

                features = []
                shard_idx += 1

        except Exception as exc:
            modality_counter["failed"] += 1
            logger.error("Feature extraction failed for %s: %s", filepath, exc)

    if features:
        shard_array = np.vstack(features)
        shard_file = output_dir / f"{prefix}_shard_{shard_idx}.npy"
        np.save(shard_file, shard_array.astype(np.float32))
        shard_files.append(shard_file)
        logger.info("Saved final shard %s with shape %s", shard_file, shard_array.shape)

    logger.info("%s modality usage: %s", prefix, modality_counter)

    return shard_files


def run_incremental_pca(
    shard_files: List[Path],
    features_dim: int,
    output_dir: Path,
    prefix: str,
    explained_variance: float = 0.99,
) -> List[Path]:
    if not shard_files:
        raise ValueError("No shard files provided for PCA")

    shapes = [np.load(f, mmap_mode="r").shape for f in shard_files]

    original_dim = int(shapes[0][1])
    n_samples_total = int(sum(s[0] for s in shapes))
    min_batch_samples = int(min(s[0] for s in shapes))

    max_components = min(
        int(features_dim),
        int(original_dim),
        int(n_samples_total),
        int(min_batch_samples),
    )

    if max_components < 2:
        raise ValueError(
            f"Not enough samples for PCA: max_components={max_components}, "
            f"n_samples_total={n_samples_total}, min_batch_samples={min_batch_samples}"
        )

    logger.info(
        "Running IncrementalPCA: max_components=%s, total_samples=%s, "
        "original_dim=%s, min_batch_samples=%s",
        max_components,
        n_samples_total,
        original_dim,
        min_batch_samples,
    )

    ipca_full = IncrementalPCA(n_components=max_components)

    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
        if batch.shape[0] < max_components:
            logger.warning(
                "Skipping too-small batch for initial PCA: %s shape=%s max_components=%s",
                f,
                batch.shape,
                max_components,
            )
            continue
        ipca_full.partial_fit(batch)

    cumulative_variance = np.cumsum(ipca_full.explained_variance_ratio_)

    selected_components = int(np.searchsorted(cumulative_variance, explained_variance) + 1)
    selected_components = min(selected_components, max_components)

    logger.info(
        "PCA selected %s components to preserve %.2f%% variance",
        selected_components,
        explained_variance * 100,
    )

    ipca = IncrementalPCA(n_components=selected_components)

    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
        if batch.shape[0] < selected_components:
            logger.warning(
                "Skipping too-small batch for final PCA fit: %s shape=%s selected_components=%s",
                f,
                batch.shape,
                selected_components,
            )
            continue
        ipca.partial_fit(batch)

    reduced_files = []

    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
        reduced = ipca.transform(batch)

        out = output_dir / f"{prefix}_{f.stem}_pca.npy"
        np.save(out, reduced.astype(np.float32))
        reduced_files.append(out)

    pca_info = {
        "original_dim": int(original_dim),
        "max_components": int(max_components),
        "selected_components": int(selected_components),
        "explained_variance_target": float(explained_variance),
        "explained_variance_actual": float(cumulative_variance[selected_components - 1]),
        "total_samples": int(n_samples_total),
        "min_batch_samples": int(min_batch_samples),
        "num_shards": int(len(shard_files)),
    }

    with open(output_dir / f"{prefix}_pca_info.json", "w", encoding="utf-8") as f:
        json.dump(pca_info, f, indent=2)

    logger.info("Saved PCA info: %s", pca_info)

    return reduced_files


async def process_metadata_file(
    metadata_file: Path,
    output_dir: Path,
    extractor_func: Callable,
    segmentation_assets: Dict[str, Dict[str, object]],
    deep_feature_extraction_model,
    shard_size: int = 1000,
) -> Optional[List[Path]]:
    try:
        df = pd.read_csv(metadata_file)
    except OSError as e:
        logger.error("Cannot read metadata file %s: %s", metadata_file, e)
        return None

    df = normalize_metadata_columns(df)

    if "filepath" not in df.columns:
        logger.warning("Skipping %s: missing filepath/image_path column", metadata_file.name)
        return None

    df["filepath"] = df["filepath"].astype(str)
    df = df[df["filepath"].apply(os.path.exists)].sort_values("filepath")

    if df.empty:
        logger.warning("No valid rows in %s", metadata_file.name)
        return None

    prefix = metadata_file.stem
    logger.info("%s: %s valid rows", metadata_file.name, len(df))

    shard_files = await extract_features_parallel(
        df=df,
        extractor_func=extractor_func,
        segmentation_assets=segmentation_assets,
        output_dir=output_dir,
        prefix=prefix,
        deep_feature_extraction_model=deep_feature_extraction_model,
        metadata_file=metadata_file,
        shard_size=shard_size,
    )

    logger.info("Shard files for %s: %s", prefix, shard_files)

    return shard_files or None


async def run_pipeline_with_extract(
    metadata_dir: str,
    output_dir: str,
    features_dim: int,
    folds: int,
    extractor_func: Callable,
    segmentation_assets: Dict[str, Dict[str, object]],
    deep_feature_extraction_model,
    shard_size: int = 1000,
    explained_variance: float = 0.99,
):
    meta_path = Path(metadata_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Metadata path: %s", meta_path)
    logger.info("Output path: %s", output_path)

    metadata_files = get_metadata_files()

    if not metadata_files:
        raise RuntimeError("No metadata CSV files found")

    build_global_label_mapping(
        metadata_files,
        output_path / "label_mapping.json",
    )

    all_shard_files: List[Path] = []

    for metadata_file in metadata_files:
        shard_files = await process_metadata_file(
            metadata_file=metadata_file,
            output_dir=output_path,
            extractor_func=extractor_func,
            segmentation_assets=segmentation_assets,
            deep_feature_extraction_model=deep_feature_extraction_model,
            shard_size=shard_size,
        )

        if shard_files:
            all_shard_files.extend(shard_files)

    if not all_shard_files:
        raise RuntimeError("No features extracted")

    reduced_feature_files = run_incremental_pca(
        shard_files=all_shard_files,
        features_dim=features_dim,
        output_dir=output_path,
        prefix="global",
        explained_variance=explained_variance,
    )

    logger.info("Reduced feature files: %s", reduced_feature_files)

    return reduced_feature_files


def get_valid_segmentation_image_paths(
    extensions: Iterable[str] = ("png", "jpg", "jpeg"),
    path_column: str = "filepath",
) -> List[str]:
    if isinstance(extensions, (str, Path)):
        raise TypeError(f"`extensions` must be iterable of strings, got {type(extensions)}")

    extensions = tuple(ext.lower().lstrip(".") for ext in extensions)
    paths: List[str] = []

    for csv_file in get_metadata_files():
        try:
            df = pd.read_csv(csv_file)

            if path_column not in df.columns and "image_path" in df.columns:
                path_column = "image_path"

            mask = df[path_column].astype(str).str.lower().str.endswith(extensions)
            paths.extend(df.loc[mask, path_column].tolist())

        except Exception as exc:
            logger.warning("Skipping %s while collecting image paths: %s", csv_file, exc)

    return sorted(paths)


async def extract_1(metadata_file_path: str):
    segmentation_assets = get_segmentation_assets()
    deep_feature_extraction_model = get_densenet_extractor_model()

    output_dir = resolve_output_dir()

    await run_pipeline_with_extract(
        metadata_dir=metadata_file_path,
        output_dir=str(output_dir),
        features_dim=int(os.getenv("FEATURES_DIM", "950")),
        folds=5,
        extractor_func=Factory.extractor,
        segmentation_assets=segmentation_assets,
        deep_feature_extraction_model=deep_feature_extraction_model,
        shard_size=int(os.getenv("FEATURE_SHARD_SIZE", "1000")),
        explained_variance=float(os.getenv("PCA_EXPLAINED_VARIANCE", "0.99")),
    )