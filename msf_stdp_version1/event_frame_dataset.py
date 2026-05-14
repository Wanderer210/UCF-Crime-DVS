from pathlib import Path
from typing import Optional, List, Tuple, Iterator
import shutil
import zipfile

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
    keys = list(obj.keys())
    if not keys:
        raise ValueError(f"Empty npz: {path}")
    return obj[keys[0]]


def _npz_member_name(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".npy")]
    for name in ("frames.npy", "event_frames.npy", "arr_0.npy"):
        if name in names:
            return name
    if not names:
        raise ValueError(f"Empty npz: {path}")
    raw_event_names = {"t.npy", "x.npy", "y.npy", "p.npy"}
    if raw_event_names.issubset(set(names)):
        raise ValueError(
            f"Raw-event npz is not supported by EventFrameVideoDataset: {path}. "
            "Please convert events to frame tensors first."
        )
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"Cannot infer frame array key from npz members {names} in {path}. "
        "Expected frames/event_frames/arr_0."
    )


def _materialize_npz_member(path: Path) -> Path:
    member = _npz_member_name(path)
    cache_file = path.parent / ".npz_cache" / path.stem / Path(member).name
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf, zf.open(member) as src, open(cache_file, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    return cache_file


def _open_event_array_lazy(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=True)
    if suffix == ".npz":
        return np.load(_materialize_npz_member(path), mmap_mode="r", allow_pickle=True)
    return load_event_array(path).numpy()


def _num_frames(arr) -> int:
    if arr.ndim == 3:
        return arr.shape[0]
    if arr.ndim == 4:
        if arr.shape[1] in (1, 2, 3, 4) or arr.shape[-1] in (1, 2, 3, 4):
            return arr.shape[0]
        if arr.shape[0] in (1, 2, 3, 4):
            return arr.shape[1]
    raise ValueError(f"Cannot infer event frame layout from shape {tuple(arr.shape)}")


def _read_frame(arr, t: int) -> torch.Tensor:
    if arr.ndim == 3:
        x = torch.from_numpy(np.asarray(arr[t])).float().unsqueeze(0)
    elif arr.ndim == 4 and arr.shape[1] in (1, 2, 3, 4):
        x = torch.from_numpy(np.asarray(arr[t])).float()
    elif arr.ndim == 4 and arr.shape[-1] in (1, 2, 3, 4):
        x = torch.from_numpy(np.asarray(arr[t])).float().permute(2, 0, 1).contiguous()
    elif arr.ndim == 4 and arr.shape[0] in (1, 2, 3, 4):
        x = torch.from_numpy(np.asarray(arr[:, t])).float()
    else:
        raise ValueError(f"Cannot infer event frame layout from shape {tuple(arr.shape)}")
    if x.shape[0] == 1:
        x = torch.cat([x, torch.zeros_like(x)], dim=0)
    elif x.shape[0] > 2:
        x = x[:2]
    return x


def iter_event_frames(path: Path, image_size: int = 128, max_frames_per_video: int = 0, frame_stride: int = 1) -> Iterator[torch.Tensor]:
    arr = _open_event_array_lazy(path)
    total = _num_frames(arr)
    stride = max(1, frame_stride)
    stop = total if max_frames_per_video <= 0 else min(total, max_frames_per_video * stride)
    for t in range(0, stop, stride):
        yield preprocess_frames(
            _read_frame(arr, t).unsqueeze(0),
            image_size=image_size,
            normalize=True,
        ).squeeze(0)


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

    def get_path(self, idx) -> str:
        return str(self.files[idx])

    def iter_video_frames(self, idx) -> Iterator[torch.Tensor]:
        path = self.files[idx]
        yield from iter_event_frames(
            path,
            image_size=self.image_size,
            max_frames_per_video=self.max_frames_per_video,
            frame_stride=self.frame_stride,
        )

    def __getitem__(self, idx):
        path = self.files[idx]
        raise RuntimeError(
            "EventFrameVideoDataset no longer supports dense video loading via __getitem__(), "
            f"which would load the whole video into memory: {path}. "
            "Use get_path(idx) and iter_video_frames(idx) instead."
        )
