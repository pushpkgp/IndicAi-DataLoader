import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image

class StreamingMultiModalDatasetTorch(IterableDataset):
    def __init__(self, index_path, modality='images', transform=None):
        self.df = pd.read_csv(index_path)
        self.df = self.df[self.df['modality'] == modality]
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        data = self.df.copy()
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            data = data.iloc[worker_id::total_workers]

        for _, row in data.iterrows():
            filepath = row['filepath']
            label = row['category']
            try:
                img = Image.open(filepath).convert('RGB')
                yield (self.transform(img) if self.transform else img, label)
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")