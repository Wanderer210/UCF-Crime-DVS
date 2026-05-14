"""
可选：当原论文 main.py 不方便直接读取新特征时，用这个 Dataset 替换原来的 feature loader。

特征目录格式:
    features_stdp/
      Abuse001.npy
      Normal_Videos_003.npy
      ...

每个 .npy:
    [T, D]，D 默认 512

label:
    文件名包含 Normal -> 0，否则 -> 1
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class STDPFeatureBagDataset(Dataset):
    def __init__(self, feature_root: str):
        self.files = sorted(Path(feature_root).rglob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy features found under {feature_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        feat = np.load(p).astype("float32")  # [T,D]
        label = 0 if "normal" in p.stem.lower() else 1
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long), str(p)
