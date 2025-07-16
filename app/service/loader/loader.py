import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import IterableDataset, get_worker_info

from app.config.logging_config import logger
from app.service.generator.feature_generator import generate_image_features
from app.utils.file_utils import save

from sklearn.preprocessing import LabelEncoder

def initialize_kfolds(folds, reduced_features, labels):
    # Initialize KFold
    kf = KFold(n_splits=folds, shuffle=True)

    i = 1
    # Perform K-fold cross-validation
    for train_index, test_index in kf.split(reduced_features):
        feature_train, feature_test = reduced_features[train_index], reduced_features[test_index]
        save('feature_train_' + str(i), feature_train)
        save('feature_test_' + str(i), feature_test)

        label_train, label_test = labels[train_index], labels[test_index]
        save('label_train_' + str(i), label_train)
        save('label_test_' + str(i), label_test)
        i += 1

def get_labels():
    labels = []
    labels = LabelEncoder().fit_transform(labels)
    save('labels', labels)
    return labels

# Dimensionality Reduction
def reduce_features(features):
    pca = PCA(n_components=950)
    reduced_features = pca.fit_transform(features)
    print("Reduced Feature Matrix Shape:", reduced_features.shape)
    return reduced_features

def process_row(row):
    modality = row.get("modality")
    filepath = row.get("filepath")
    label = row.get("category")

    if not filepath or not os.path.exists(filepath):
        return None

    try:
        if modality == 'image':
            return generate_image_features(filepath)

        # elif modality == 'text':
        #     with open(filepath, 'r', encoding='utf-8') as f:
        #         text = f.read()
        #     yield text, label
        #
        # elif modality == 'audio':
        #     waveform, sample_rate = torchaudio.load(filepath)
        #     yield waveform, label
        #
        # elif modality == 'video':
        #     yield filepath, label  # Load externally
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def get_features(rows):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_row, row): row for _, row in rows}

        features = []
        for future in as_completed(futures):
            try:
                feature = future.result()
                if feature is not None:
                    features.append(feature)
            except Exception as exc:
                print(f"Row processing raised exception: {exc}")
                logger.warning("One of the rows returned None and will be skipped.")

        features = abs(np.array(features))

        save('features', features)

        return features

class IndicDataLoader(IterableDataset):
    def __init__(self, index_path, split='train', modality=None, category=None, shuffle=True):
        self.df = pd.read_csv(index_path)

        if split:
            self.df = self.df[self.df['split'] == split]

        if modality:
            self.df = self.df[self.df['modality'] == modality]

        if category:
            self.df = self.df[self.df['category'] == category]

        before = len(self.df)
        self.df = self.df[self.df["filepath"].apply(os.path.exists)]
        dropped = before - len(self.df)
        if dropped > 0:
            logger.warning(f"Skipped {dropped} missing files in dataset")

        self.shuffle = shuffle

    def __iter__(self):
        worker_info = get_worker_info()
        data = self.df.copy()

        # Optional shuffle (stream-safe)
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        # Shard rows among workers
        if worker_info is not None:
            data = data.iloc[worker_info.id::worker_info.num_workers]

        features = get_features(rows=data.iterrows())

        reduced_features = reduce_features(features=features)

        labels = get_labels()

        initialize_kfolds(folds=5, reduced_features=reduced_features, labels=labels)