from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from src.data.parse_mat import read_mat

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]


def minmax_uint8(img):
    # per-image, because intensity scales differ between scanners/acquisitions
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


class BrainTumorDataset(Dataset):
    def __init__(self, manifest, data_root, split=None, transform=None, return_mask=False):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if split is not None:
            df = df[df.split == split]
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.return_mask = return_mask
        assert all(CLASS_NAMES[l - 1] == n for l, n in zip(self.df.label, self.df.label_name))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        s = read_mat(self.data_root / r.file_path)
        img = torch.from_numpy(minmax_uint8(s.image)).unsqueeze(0).repeat(3, 1, 1)
        sample = {"image": tv_tensors.Image(img)}
        if self.return_mask:
            sample["tumor_mask"] = tv_tensors.Mask(torch.from_numpy(s.tumor_mask).unsqueeze(0))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.return_mask:
            sample["tumor_mask"] = sample["tumor_mask"][0]
        sample["label"] = int(r.label) - 1
        sample["patient_id"] = r.patient_id
        sample["file_path"] = r.file_path
        return sample