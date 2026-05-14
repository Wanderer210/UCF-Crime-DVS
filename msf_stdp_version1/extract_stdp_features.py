import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stdp_encoder import STDPFeatureEncoder
from event_frame_dataset import EventFrameVideoDataset


def parse_args():
    p = argparse.ArgumentParser("Extract MSF-compatible features using STDP encoder")
    p.add_argument("--event_frame_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--max_frames_per_video", type=int, default=0,
                   help="0 means use all frames")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--c1", type=int, default=128)
    p.add_argument("--c2", type=int, default=384)
    return p.parse_args()


def output_name(src_path: str) -> str:
    p = Path(src_path)
    return p.stem + ".npy"


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    c1 = int(cfg.get("c1", args.c1))
    c2 = int(cfg.get("c2", args.c2))

    model = STDPFeatureEncoder(in_channels=2, c1=c1, c2=c2).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    dataset = EventFrameVideoDataset(
        root=args.event_frame_root,
        normal_keyword=None,
        max_videos=args.max_videos,
        image_size=args.image_size,
        max_frames_per_video=args.max_frames_per_video,
        frame_stride=args.frame_stride,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Info] videos: {len(dataset)}")
    print(f"[Info] feature dim: {model.out_dim}")

    for idx in tqdm(range(len(dataset)), desc="Extract STDP features"):
        path = dataset.get_path(idx)
        model.reset_state()

        feats = []
        for frame in dataset.iter_video_frames(idx):
            x = frame.unsqueeze(0).to(device, non_blocking=True)
            feat, _ = model(x, learn_stdp=False)
            feats.append(feat.squeeze(0).detach().cpu().numpy())

        feats = np.stack(feats, axis=0).astype("float32")  # [T,D]

        # 保留相对目录结构，避免同名视频覆盖
        src = Path(path)
        try:
            rel = src.relative_to(Path(args.event_frame_root))
            dst_dir = out_root / rel.parent
        except Exception:
            dst_dir = out_root
        dst_dir.mkdir(parents=True, exist_ok=True)

        out_path = dst_dir / output_name(path)
        np.save(out_path, feats)

    print(f"[OK] Features saved to: {out_root}")


if __name__ == "__main__":
    main()
