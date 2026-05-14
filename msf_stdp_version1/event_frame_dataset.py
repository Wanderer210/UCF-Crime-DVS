from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SUPPORTED_SUFFIX = {".npy", ".npz", ".pt", ".pth"}


def find_event_files(root: str) -> List[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIX]
    return sorted(files)


def _load_npz(path: Path):
    obj = np.load(path, allow_pickle=True)
    if "frames" in obj:
        return obj["frames"]
    if "event_frames" in obj:
        return obj["event_frames"]
    if "arr_0" in obj:
        return obj["arr_0"]
    # fallback: 取第一个数组
    keys = list(obj.keys())
    if not keys:
        raise ValueError(f"Empty npz: {path}")
    return obj[keys[0]]


def load_event_array(path: Path) -> torch.Tensor:
    """
    返回 [T, C, H, W] float tensor。

    支持常见格式:
      [T, C, H, W]
      [T, H, W, C]
      [C, T, H, W]
      [T, H, W]
    """
    if path.suffix.lower() == ".npz":
        arr = _load_npz(path)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif path.suffix.lower() in {".pt", ".pth"}:
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, dict):
            for key in ["frames", "event_frames", "data", "x"]:
                if key in arr:
                    arr = arr[key]
                    break
        if isinstance(arr, torch.Tensor):
            x = arr.float()
            return normalize_shape(x)
        arr = np.asarray(arr)
    else:
        raise ValueError(f"Unsupported file: {path}")

    x = torch.from_numpy(np.asarray(arr)).float()
    return normalize_shape(x)


def normalize_shape(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        # [T,H,W] -> [T,1,H,W]
        x = x.unsqueeze(1)
    elif x.ndim == 4:
        # [T,C,H,W]
        if x.shape[1] in (1, 2, 3, 4):
            pass
        # [T,H,W,C]
        elif x.shape[-1] in (1, 2, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        # [C,T,H,W]
        elif x.shape[0] in (1, 2, 3, 4):
            x = x.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"Cannot infer event frame layout from shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")

    # 保证 C=2。单通道复制为 ON/OFF 占位；超过2通道取前2个。
    if x.shape[1] == 1:
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.shape[1] > 2:
        x = x[:, :2]

    return x


def preprocess_frames(
    frames: torch.Tensor,
    image_size: int = 128,
    log_scale: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    frames: [T,C,H,W]
    """
    frames = frames.float()
    if log_scale:
        frames = torch.sign(frames) * torch.log1p(frames.abs())

    if image_size > 0 and (frames.shape[-1] != image_size or frames.shape[-2] != image_size):
        frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)

    if normalize:
        # per-video robust normalization
        denom = frames.flatten(1).abs().amax(dim=1).clamp_min(1e-6)
        frames = frames / denom.view(-1, 1, 1, 1)

    return frames.clamp(min=0.0)


class EventFrameVideoDataset(Dataset):
    """
    每个 item 返回一个视频的事件帧:
        frames [T,C,H,W], path
    """

    def __init__(
        self,
        root: str,
        normal_keyword: Optional[str] = None,
        max_videos: int = 0,
        image_size: int = 128,
        max_frames_per_video: int = 0,
        frame_stride: int = 1,
    ):
        self.root = Path(root)
        files = find_event_files(root)

        if normal_keyword:
            files = [p for p in files if normal_keyword.lower() in str(p).lower()]

        if max_videos and max_videos > 0:
            files = files[:max_videos]

        if not files:
            raise RuntimeError(f"No event frame files found under {root}")

        self.files = files
        self.image_size = image_size
        self.max_frames_per_video = max_frames_per_video
        self.frame_stride = max(1, frame_stride)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        frames = load_event_array(path)  # [T,C,H,W]

        frames = frames[::self.frame_stride]
        if self.max_frames_per_video and self.max_frames_per_video > 0:
            frames = frames[: self.max_frames_per_video]

        frames = preprocess_frames(frames, image_size=self.image_size)
        return frames, str(path)
