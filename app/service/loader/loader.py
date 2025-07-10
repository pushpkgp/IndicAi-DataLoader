import os
import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image
import torchaudio

from app.config.logging_config import logger


class StreamingDatasetLoader(IterableDataset):
    def __init__(self, index_path, split='train', modality=None, category=None, transform=None, shuffle=True):
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

        self.transform = transform
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

        for _, row in data.iterrows():
            modality = row['modality']
            filepath = row['filepath']
            label = row['category']

            try:
                if modality == 'image':
                    img = Image.open(filepath).convert('RGB')
                    yield self.transform(img) if self.transform else img, label

                elif modality == 'text':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    yield text, label

                elif modality == 'audio':
                    waveform, sample_rate = torchaudio.load(filepath)
                    yield waveform, label

                elif modality == 'video':
                    yield filepath, label  # Load externally
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
