import asyncio
import csv
import glob
import json
import logging
import os
import re
import uuid
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.config.config import Config
from app.service.feature.extractor_factory import Factory

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
        raise ValueError(f"Metadata filename unexpected format: {filename}")
    g = match.groupdict()
    logger.info(f"Pushpinder: Parsed file -> {filename}")
    return g["modality"], g["body_part"], g["modality_type"], g["split"], g["category"]

def parse_meta_filename_1(path: Path) -> Tuple[str, str, str, str, str]:
    """
    Parse metadata from a CSV file.
    NOTE: Despite the name, metadata is NOT parsed from filename,
    but from CSV content (single-row metadata file).
    """

    # path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path.name}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV metadata file, got: {path.name}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        try:
            row = next(reader)
        except StopIteration:
            raise ValueError(f"Metadata CSV is empty: {path.name}")

    required_fields = {
        "modality",
        "body_part",
        "modality_type",
        "split",
        "category",
    }

    missing = required_fields - row.keys()
    if missing:
        raise ValueError(
            f"Metadata CSV missing required fields {missing} in file: {path.name}"
        )

    modality = row["modality"].strip()
    body_part = row["body_part"].strip()
    modality_type = row["modality_type"].strip()
    split = row["split"].strip()
    category = row["category"].strip()

    logger.info(
        "Parsed metadata | file=%s modality=%s body_part=%s modality_type=%s split=%s category=%s",
        path.name, modality, body_part, modality_type, split, category
    )

    return modality, body_part, modality_type, split, category

def build_global_label_mapping(csv_files: Iterable[Path], label_map_path: Path) -> Tuple[LabelEncoder, dict]:
    # categories = []
    # for f in meta_dir.glob("*.csv"):
    #     try:
    #         df = pd.read_csv(f)
    #         categories.extend(df["category"].dropna().unique())
    #     except Exception as e:
    #         logger.warning(f"Failed loading {f} for label mapping: {e}")
    # categories = sorted(set(categories))
    # le = LabelEncoder()
    # le.fit(categories)
    # mapping = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    # with open(label_map_path, "w") as file:
    #     json.dump(mapping, file)
    # logger.info(f"Label mapping saved at {label_map_path}, mapping: {mapping}")
    # return le, mapping

    """
        Builds a global label mapping from CSV metadata files
        in a memory-safe and streaming manner.
        """

    unique_categories = set()

    # Use rglob with generator semantics (safer for large dirs)

    for f in csv_files:
        if not f.is_file() or f.suffix != ".csv":
            continue

        try:
            # Read ONLY the required column, no full dataframe
            for chunk in pd.read_csv(
                    f,
                    usecols=["category"],
                    chunksize=5_000,
                    engine="c",  # force low-memory engine
                    dtype=str  # avoid dtype inference
            ):
                unique_categories.update(
                    chunk["category"].dropna().unique()
                )

        except Exception as e:
            logger.warning(f"Failed loading {f} for label mapping: {e}")

    if not unique_categories:
        raise ValueError("No categories found in metadata files")

    categories = sorted(unique_categories)

    le = LabelEncoder()
    le.fit(categories)

    mapping = {
        label: int(idx)
        for idx, label in enumerate(le.classes_)
    }

    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as file:
        json.dump(mapping, file)

    logger.info(
        f"Label mapping saved at {label_map_path}, "
        f"total labels: {len(mapping)}"
    )

    return le, mapping

def get_labels_global(df: pd.DataFrame, le: LabelEncoder, mapping: dict) -> np.ndarray:
    df = df.copy()
    df["label_enc"] = df["category"].map(mapping).fillna(-1).astype(int)
    return df["label_enc"].values

def initialize_kfolds_stratified(
    folds: int, labels: np.ndarray, prefix: str, output_dir: Path
) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    indices = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(labels)), labels), 1):
        np.save(output_dir / f"{prefix}_train_idx_fold_{fold_idx}.npy", train_idx)
        np.save(output_dir / f"{prefix}_test_idx_fold_{fold_idx}.npy", test_idx)
        indices.append((train_idx, test_idx))
    logger.info(f"Stratified folds saved for {prefix}")
    return indices

def incremental_pca_fit(shard_files: List[Path], features_dim: int, batch_size: int = 1000) -> IncrementalPCA:
    sample = np.load(shard_files[0], mmap_mode="r")
    n_components = min(features_dim, sample.shape[1])
    ipca = IncrementalPCA(n_components=n_components)
    for shard_file in sorted(shard_files, key=lambda p: p.name):
        data = np.load(shard_file, mmap_mode="r")
        for start in range(0, data.shape[0], batch_size):
            ipca.partial_fit(data[start: start + batch_size])
    logger.info(f"PCA model fitted with {n_components} components")
    return ipca

def incremental_pca_transform(ipca: IncrementalPCA, shard_files: List[Path], output_dir: Path, prefix: str) -> List[Path]:
    reduced_files = []
    for shard_file in sorted(shard_files, key=lambda p: p.name):
        data = np.load(shard_file, mmap_mode="r")
        reduced = ipca.transform(data)
        out_file = output_dir / f"{prefix}_{shard_file.stem}_pca.npy"
        np.save(out_file, reduced)
        reduced_files.append(out_file)
        logger.info(f"PCA reduced shard saved: {out_file}")
    return reduced_files

def process_data(row, extractor_func: Callable[[str, str], np.ndarray], reference_images) -> Optional[Tuple[np.ndarray,str]]:
    filepath = row['filepath']
    modality = row.get("modality", None)
    identifier = row.get("id", "unknown")
    if not filepath or not os.path.exists(filepath):
        logger.warning(f"Missing file {filepath}, id {identifier}")
        return None
    try:
        feature = extractor_func(modality, filepath, reference_images)
        return feature, filepath
    except Exception as e:
        logger.error(f"Extraction failed for file {filepath}, id {identifier}: {e}")
        return None

def batch_generator(rows, batch_size: int):
    batch = []
    for idx, row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

async def save_shard_async(shard_path: Path, data: np.ndarray):
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(np.save, shard_path, data)
    logger.info(f"Saved shard {shard_path}")

async def extract_features(
        rows,
        batch_size: int,
        proc_executor: ProcessPoolExecutor,
        output_dir: Path,
        prefix: str,
        extractor_func: Callable[[str, str], np.ndarray],
        reference_images,
):
    features_files = []
    valid_paths = []
    batch_index = 0
    for batch_rows in batch_generator(rows, batch_size):
        # Map each future to its row (for matching results-to-filepath deterministically)
        futures_to_filepath = {}
        futures = []
        for row in batch_rows:
            filepath = row['filepath']
            fut = proc_executor.submit(process_data, row, extractor_func, reference_images)
            futures.append(fut)
            futures_to_filepath[fut] = filepath
        batch_feats = []
        batch_extracted_paths = []
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                feat, path = result
                batch_feats.append(feat)
                batch_extracted_paths.append(path)
        if batch_feats:
            batch_array = np.array(batch_feats, dtype=np.float32)
            shard_name = f"{prefix}_batch_{batch_index}_{uuid.uuid4().hex}.npy"
            shard_path = output_dir / shard_name
            await save_shard_async(shard_path, batch_array)
            features_files.append(shard_path)
            valid_paths.extend(batch_extracted_paths)
            batch_index += 1
    logger.info(f"features_files: {features_files}")
    return features_files, valid_paths

async def extract(
        metadata_file: Path,
        output_dir: Path,
        batch_size: int,
        proc_executor: ProcessPoolExecutor,
        extractor_func: Callable[[str, str], np.ndarray],
        reference_images,
        features_dim: int,
        folds: int,
        label_encoder: LabelEncoder,
        mapping: Dict,
        split: Optional[str],
        modality: Optional[str],
        category: Optional[str],
):
    try:
        df = pd.read_csv(metadata_file)
        logger.info(f"MetaData File Processing: {metadata_file.stem}")
    except Exception as e:
        logger.error(f"Error reading {metadata_file}: {e}")
        return []

    if split:
        df = df[df["split"] == split]

    if modality:
        df = df[df["modality"] == modality]

    if category:
        df = df[df["category"] == category]

    df = df[df["filepath"].apply(os.path.exists)]
    df = df.sort_values("filepath")

    if df.empty:
        logger.warning(f"No valid files for {metadata_file.name} after filtering")
        return []

    prefix = metadata_file.stem

    features_files, valid_paths = await extract_features(
        df.iterrows(), batch_size, proc_executor, output_dir, prefix, extractor_func, reference_images
    )
    if not features_files:
        logger.warning(f"No features extracted for {metadata_file.name}")
        return []

    ipca = incremental_pca_fit(features_files, features_dim)
    reduced_files = incremental_pca_transform(ipca, features_files, output_dir, prefix)

    valid_df = df[df["filepath"].isin(valid_paths)]
    labels = get_labels_global(valid_df, label_encoder, mapping)

    initialize_kfolds_stratified(folds, labels, prefix, output_dir)
    return reduced_files

async def run_pipeline_with_extract(
        metadata_dir: str,
        output_dir: str,
        batch_size: int,
        num_workers: int,
        features_dim: int,
        folds: int,
        max_concurrent_files: int,
        extractor_func: Callable[[str, str], np.ndarray],
        reference_images,
        retries: int = 2,
):
    semaphore = asyncio.Semaphore(max_concurrent_files)
    meta_path = Path(metadata_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    label_map_path = output_path / "label_mapping.json"

    csv_files = (
        f for f in meta_path.iterdir()
        if f.is_file() and f.suffix == ".csv"
    )

    label_encoder, mapping = build_global_label_mapping(csv_files, label_map_path)

    metadata_files = list(csv_files)

    results = []

    proc_executor = ProcessPoolExecutor(max_workers=num_workers)

    async def process_file(file_path: Path):
        nonlocal proc_executor
        async with semaphore:
            for attempt in range(1, retries + 2):
                try:
                    modality, _, _, split, category = parse_meta_filename(file_path.name)
                except ValueError as ve:
                    logger.warning(f"{ve} Skipping {file_path.name}")
                    return

                try:
                    reduced_files = await extract(
                        file_path,
                        output_path,
                        batch_size,
                        proc_executor,
                        extractor_func,
                        reference_images,
                        features_dim,
                        folds,
                        label_encoder,
                        mapping,
                        split,
                        modality,
                        category,
                    )

                    results.append({
                        "metadata_file": file_path.name,
                        "reduced_feature_files": [str(rf) for rf in reduced_files]
                    })

                    logger.info(f"Finished processing {file_path.name}")
                    return
                except Exception as e:
                    logger.error(f"Error processing {file_path.name} on attempt {attempt}: {e}")
                    if attempt <= retries:
                        backoff = 5 * (2 ** (attempt - 1))
                        logger.info(f"Retrying {file_path.name} after {backoff}s")
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"Failed permanently: {file_path.name}")

    await asyncio.gather(*[process_file(f) for f in metadata_files])
    proc_executor.shutdown()
    return results

def get_valid_segmentation_image_paths(metadata_dir, exts=('png', 'jpg', 'jpeg'), path_column='filepath'):
    # Find all metadata CSV files with 'valid' in the filename
    metadata_files = glob.glob(os.path.join(metadata_dir, '*train*.csv'))

    image_paths = []
    for meta_file in metadata_files:
        df = pd.read_csv(meta_file)
        # Filter image paths by file extension (case-insensitive)
        valid_paths = df[df[path_column].str.lower().str.endswith(exts)][path_column]
        image_paths.extend(valid_paths.tolist())
    return sorted(image_paths)

async def extract_1(metadata_file_path:str):
    await run_pipeline_with_extract(
        metadata_dir=metadata_file_path,
        output_dir="/data/features/image",
        batch_size=1000,
        num_workers=4,
        features_dim=950,
        folds=5,
        max_concurrent_files=4,
        extractor_func=Factory.extractor,
        reference_images=get_valid_segmentation_image_paths(Config().get_path("val_images"))[:10],
        retries=2,
   )