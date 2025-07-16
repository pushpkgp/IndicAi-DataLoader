from pathlib import Path
import pandas as pd

class MetadataGenerator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def generate_and_save_metadata(self):
        base_dir = Path(self.dataset_path)
        metadata_root = Path(".") / "metadata" / "image"
        metadata_root.mkdir(parents=True, exist_ok=True)

        rows_by_split = {}

        # Efficiently iterate using pathlib's glob
        for image_path in base_dir.rglob("*.*"):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            parts = image_path.parts
            try:
                idx = parts.index("images")
                modality_type = parts[idx + 2]   # e.g., 'ct' or 'x-ray'
                split = parts[idx + 3]           # e.g., 'train'
                label_dir = Path(*parts[:idx + 5])
            except (ValueError, IndexError):
                continue  # Skip malformed path

            row = {
                "filepath": str(image_path),
                "label": str(label_dir),
                "modality": "image",
                "split": split,
                "category": "cat"
            }

            rows_by_split.setdefault(split, []).append(row)

        # Save each split to a CSV
        for split, rows in rows_by_split.items():
            df = pd.DataFrame(rows)
            output_path = metadata_root / f"{split}.csv"
            df.to_csv(output_path, sep="\t", index=False)
            print(f"✅ Metadata saved: {output_path}")