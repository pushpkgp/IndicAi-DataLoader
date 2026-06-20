from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.metadata_cache import get_metadata_files
from app.model.densnet import get_densenet_extractor
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

def build_global_label_mapping(
    csv_files: Iterable[Path],
    label_map_path: Path,
):
    categories = set()

    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["category"], dtype=str)
            clean = df["category"].dropna().str.strip()
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
    label_map_path.write_text(json.dumps(mapping), encoding="utf-8")

    logger.info("Label mapping created with %s classes at %s", len(mapping), label_map_path)

    return le, mapping, categories

async def extract_features_parallel(
    df: pd.DataFrame,
    extractor_func: Callable,
    segmentation_model,
    postprocess_params: Dict[str, object],
    output_dir: Path,
    prefix: str,
    deep_feature_extraction_model,
    shard_size: int = 500,
) -> List[Path]:
    features = []
    shard_files: List[Path] = []
    shard_idx = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        filepath = row.filepath
        modality = getattr(row, "modality", "image")

        if not os.path.exists(filepath):
            continue

        try:
            feature = extractor_func(
                modality,
                filepath,
                segmentation_model,
                postprocess_params,
                deep_feature_extraction_model
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
                np.save(shard_file, shard_array)
                shard_files.append(shard_file)
                features = []
                shard_idx += 1

        except Exception as exc:
            logger.error("Feature extraction failed for %s: %s", filepath, exc)

    if features:
        shard_array = np.vstack(features)
        shard_file = output_dir / f"{prefix}_shard_{shard_idx}.npy"
        np.save(shard_file, shard_array)
        shard_files.append(shard_file)

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

    sample = np.load(shard_files[0], mmap_mode="r")
    n_samples_total = sum(np.load(f, mmap_mode="r").shape[0] for f in shard_files)
    original_dim = sample.shape[1]

    max_components = min(features_dim, original_dim, n_samples_total)

    logger.info(
        "Running initial IncrementalPCA: max_components=%s, total_samples=%s, original_dim=%s",
        max_components,
        n_samples_total,
        original_dim,
    )

    ipca_full = IncrementalPCA(n_components=max_components)

    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
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
    }

    with open(output_dir / f"{prefix}_pca_info.json", "w") as f:
        json.dump(pca_info, f, indent=2)

    logger.info("Saved PCA info: %s", pca_info)

    return reduced_files

async def process_metadata_file(
    metadata_file: Path,
    output_dir: Path,
    extractor_func: Callable,
    segmentation_model,
    postprocess_params: Dict[str, object],
    deep_feature_extraction_model
) -> Optional[List[Path]]:
    try:
        df = pd.read_csv(metadata_file)
    except OSError as e:
        logger.error(
            "Cannot read metadata file %s: %s",
            metadata_file,
            e
        )
        return None

    if "filepath" not in df.columns:
        logger.warning("Skipping %s: missing filepath column", metadata_file.name)
        return None

    df = df[df["filepath"].apply(os.path.exists)].sort_values("filepath")

    if df.empty:
        logger.warning("No valid rows in %s", metadata_file.name)
        return None

    prefix = metadata_file.stem
    logger.info("%s: %s valid rows", metadata_file.name, len(df))

    shard_files = await extract_features_parallel(
        df=df,
        extractor_func=extractor_func,
        segmentation_model=segmentation_model,
        postprocess_params=postprocess_params,
        output_dir=output_dir,
        prefix=prefix,
        deep_feature_extraction_model=deep_feature_extraction_model
    )

    logger.info("Shard files for %s: %s", prefix, shard_files)

    return shard_files or None

async def run_pipeline_with_extract(
    metadata_dir: str,
    output_dir: str,
    features_dim: int,
    folds: int,
    extractor_func: Callable,
    segmentation_model,
    postprocess_params: Dict[str, object],
    deep_feature_extraction_model
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
            segmentation_model=segmentation_model,
            postprocess_params=postprocess_params,
            deep_feature_extraction_model=deep_feature_extraction_model
        )

        if shard_files:
            all_shard_files.extend(shard_files)

    if not all_shard_files:
        raise RuntimeError("No features extracted")

    reduced_feature_files = run_incremental_pca(
        shard_files=all_shard_files,
        features_dim=950,
        output_dir=output_path,
        prefix="global",
        explained_variance=0.99,
    )

    logger.info("Reduced feature files: %s", reduced_feature_files)

def create_stratified_folds(X, y, output_dir, n_splits=5):
    if len(X) < 2:
        logger.warning("Not enough samples")
        return

    from collections import Counter

    min_class_samples = min(Counter(y).values())

    if min_class_samples < n_splits:
        n_splits = min_class_samples

    if n_splits < 2:
        logger.warning("Not enough samples per class")
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        np.save(output_dir / f"x_train_{i}.npy", X[train_index])
        np.save(output_dir / f"y_train_{i}.npy", y[train_index])
        np.save(output_dir / f"x_test_{i}.npy", X[test_index])
        np.save(output_dir / f"y_test_{i}.npy", y[test_index])

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
            df = pd.read_csv(csv_file, usecols=[path_column])
            mask = df[path_column].astype(str).str.lower().str.endswith(extensions)
            paths.extend(df.loc[mask, path_column].tolist())
        except Exception as exc:
            logger.warning("Skipping %s while collecting image paths: %s", csv_file, exc)

    return sorted(paths)

async def extract_1(metadata_file_path: str):
    segmentation_model, postprocess_params = get_segmentation_assets()
    deep_feature_extraction_model = get_densenet_extractor()

    await run_pipeline_with_extract(
        metadata_dir=metadata_file_path,
        output_dir="/data/features/image",
        features_dim=950,
        folds=5,
        extractor_func=Factory.extractor,
        segmentation_model=segmentation_model,
        postprocess_params=postprocess_params,
        deep_feature_extraction_model=deep_feature_extraction_model
    )