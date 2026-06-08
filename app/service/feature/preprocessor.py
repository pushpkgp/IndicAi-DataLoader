import asyncio
import json
import logging
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.metadata_cache import get_metadata_files
from app.service.feature.extractor_factory import Factory
from app.service.feature.image.segmentation import get_segmentation_mask_model

# ----------------------------
# GLOBAL CONFIG (IMPORTANT)
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
pd.options.mode.copy_on_write = True

logger = logging.getLogger("feature_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------
# METADATA FILENAME PARSER
# ----------------------------
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

# ----------------------------
# LABEL MAPPING (SERIAL)
# ----------------------------
def build_global_label_mapping(
    csv_files: Iterable[Path],
    label_map_path: Path
):

    categories = set()

    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["category"], dtype=str)
            # categories.update(df["category"].dropna().unique())
            clean = (
                df["category"]
                .dropna()
                .str.strip()
            )

            clean = clean[clean.str.lower() != "category"]

            categories.update(clean.unique())
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")

    if not categories:
        raise ValueError("No categories found in metadata files")

    le = LabelEncoder()
    le.fit(sorted(categories))

    mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}

    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    label_map_path.write_text(json.dumps(mapping))
    logger.info(f"Label mapping created with {mapping} classes, File Path {label_map_path}")

    # logger.info("Label mapping created with %d classes", len(mapping))
    return le, mapping, categories

# ----------------------------
# FEATURE EXTRACTION WORKER
# ----------------------------
def extract_and_save_feature(
        row: Dict,
        extractor_func: Callable,
        output_dir: Path,
        prefix: str,
        model
) -> Optional[str]:
    filepath = row["filepath"]
    if not os.path.exists(filepath):
        return None

    try:
        logger.info(f"Pushpinder Extracting Features")
        feature = extractor_func(row.get("modality"), filepath, model)
        out_file = output_dir / f"{prefix}_{uuid.uuid4().hex}.npy"
        np.save(out_file, feature.astype(np.float32).reshape(1, -1))
        return str(out_file)
    except Exception as e:
        logger.error(f"Feature extraction failed for {filepath}: {e}")
        return None

# ----------------------------
# PARALLEL FEATURE EXTRACTION
# ----------------------------
async def extract_features_parallel(
        df: pd.DataFrame,
        extractor_func,
        model,
        output_dir: Path,
        prefix: str,
        shard_size=2000
):
    features = []
    shard_files = []
    shard_idx = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        row = row.to_dict()
        filepath = row["filepath"]

        if not os.path.exists(filepath):
            continue

        feature = extractor_func(row.get("modality"), filepath, model)

        if feature is None:
            continue

        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)

        # 🔥 CRITICAL: Remove NaN / Inf
        if not np.isfinite(feature).all():
            continue
        features.append(feature)

        if len(features) >= shard_size:
            shard_array = np.vstack(features)
            shard_file = output_dir / f"{prefix}_shard_{shard_idx}.npy"
            np.save(shard_file, shard_array)
            shard_files.append(shard_file)
            features = []
            shard_idx += 1

    # Save remaining
    if features:
        shard_array = np.vstack(features)
        shard_file = output_dir / f"{prefix}_shard_{shard_idx}.npy"
        np.save(shard_file, shard_array)
        shard_files.append(shard_file)

    return shard_files

# ----------------------------
# PCA
# ----------------------------
def run_incremental_pca(
        shard_files: List[Path],
        features_dim: int,
        output_dir: Path,
        prefix: str
) -> List[Path]:
    sample = np.load(shard_files[0], mmap_mode="r")
    print("Sample shape:", sample.shape)
    print("Sample ndim:", sample.ndim)

    n_samples_total = sum(np.load(f, mmap_mode="r").shape[0] for f in shard_files)

    n_components = min(features_dim, sample.shape[1], n_samples_total)

    ipca = IncrementalPCA(n_components=n_components)

    # ---- FIRST PASS: FIT ----
    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
        ipca.partial_fit(batch)

    # ---- SECOND PASS: TRANSFORM ----
    reduced_files = []
    for f in shard_files:
        batch = np.load(f, mmap_mode="r")
        reduced = ipca.transform(batch)
        out = output_dir / f"{prefix}_{f.stem}_pca.npy"
        np.save(out, reduced.astype(np.float32))
        reduced_files.append(out)

    return reduced_files

# ----------------------------
# PER-METADATA FILE PIPELINE
# ----------------------------
async def process_metadata_file(
        metadata_file: Path,
        output_dir: Path,
        executor: ProcessPoolExecutor,
        extractor_func,
        model,
):
    df = pd.read_csv(metadata_file)
    df = df[df["filepath"].apply(os.path.exists)].sort_values("filepath")

    if df.empty:
        logger.warning(f"No valid rows in {metadata_file.name}")
        return None

    prefix = metadata_file.stem

    logger.warning(f"{metadata_file.name}: {len(df)} rows")

    shard_files = await extract_features_parallel(
        df,
        extractor_func,
        model,
        output_dir,
        prefix
    )

    logger.info(f"Pushpinder Shard Files {shard_files} classes, Prefix {prefix}")

    if not shard_files:
        return None

    return shard_files

# ----------------------------
# PIPELINE ENTRYPOINT
# ----------------------------
async def run_pipeline_with_extract(
        metadata_dir: str,
        output_dir: str,
        features_dim: int,
        folds: int,
        extractor_func,
        model
):
    meta_path = Path(metadata_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.warning(f"Metadata Path: {meta_path}")

    metadata_files = get_metadata_files()
    if not metadata_files:
        raise RuntimeError("No metadata CSV files found")

    logger.warning(f"Metadata Files: {metadata_files}")

    label_encoder, label_mapping, labels = build_global_label_mapping(
        metadata_files,
        output_path / "label_mapping.json"
    )

    executor = ProcessPoolExecutor(1)

    logger.warning(f"Pushpinder output_path: {output_path}")
    all_shard_files = []
    for metadata_file in metadata_files:
        shard_files = await process_metadata_file(
                metadata_file,
                output_path,
                executor,
                extractor_func,
                model
            )

        if shard_files:
            all_shard_files.extend(shard_files)

        # if not shard_files:
        #     continue


        # all_shard_files.append(shard_files)

    if not all_shard_files:
        raise RuntimeError("No features extracted")

    reduced_feature_files = run_incremental_pca(all_shard_files, features_dim, output_path, "global")

    logger.info(f"Reduced Feature Files: {reduced_feature_files}")

    # labels = label_encoder.fit_transform(labels)
    #
    # logger.info(f"Reduced Feature Matrix Shape: {reduced_features.shape}")
    #
    # X = np.vstack(reduced_features)
    # y = np.array(labels)
    #
    # create_stratified_folds(X, y, output_path, folds)

def create_stratified_folds(X, y, output_dir, n_splits=5):

    if len(X) < 2:
        print("Not enough samples")
        return

    from collections import Counter
    min_class_samples = min(Counter(y).values())

    if min_class_samples < n_splits:
        n_splits = min_class_samples

    if n_splits < 2:
        print("Not enough samples per class")
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    i = 1
    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        np.save(output_dir / f"x_train_{i}.npy", X_train)
        np.save(output_dir / f"y_train_{i}.npy", y_train)
        np.save(output_dir / f"x_test_{i}.npy", X_test)
        np.save(output_dir / f"y_test_{i}.npy", y_test)

        i += 1

# ----------------------------
# UTILITY
# ----------------------------
def get_valid_segmentation_image_paths(
    extensions: Iterable[str] = ("png", "jpg", "jpeg"),
    path_column: str = "filepath",
) -> List[str]:
    """
    Collect valid image paths from cached metadata CSV files.
    """

    # Defensive check (catches this bug immediately)
    if isinstance(extensions, (str, Path)):
        raise TypeError(
            f"`extensions` must be an iterable of strings, got {type(extensions)}"
        )

    extensions = tuple(ext.lower().lstrip(".") for ext in extensions)
    paths: List[str] = []

    for csv_file in get_metadata_files():
        df = pd.read_csv(csv_file, usecols=[path_column])

        mask = (
            df[path_column]
            .astype(str)
            .str.lower()
            .str.endswith(extensions)
        )

        paths.extend(df.loc[mask, path_column].tolist())

    return sorted(paths)

# ----------------------------
# PUBLIC API
# ----------------------------
async def extract_1(metadata_file_path: str):
    model = get_segmentation_mask_model(get_valid_segmentation_image_paths())
    await run_pipeline_with_extract(
        metadata_dir=metadata_file_path,
        output_dir="/data/features/image",
        features_dim=950,
        folds=5,
        extractor_func=Factory.extractor,
        model= model
    )