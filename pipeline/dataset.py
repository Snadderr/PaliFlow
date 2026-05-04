"""JSONLDataset — loads Roboflow paligemma-format annotations."""

import json
from pathlib import Path
from PIL import Image


class JSONLDataset:
    def __init__(self, jsonl_file_path, image_directory_path):
        self.image_directory = Path(image_directory_path)
        self.entries = []
        with open(jsonl_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = self.image_directory / entry["image"]
        image = Image.open(image_path).convert("RGB")
        return image, entry
