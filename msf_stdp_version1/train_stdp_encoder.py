import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from stdp_encoder import STDPFeatureEncoder
from event_frame_dataset import EventFrameVideoDataset


def parse_args():
    p = argparse.ArgumentParser("Unsupervised STDP pretraining for event-frame encoder")
    p.add_argument("--event_frame_root", type=str, required=True)
    p.add_argument("--normal_keyword", type=str, default="Normal",
                   help="Only files containing this keyword are used for STDP pretraining. "
                        "Set empty string to use all files.")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--max_frames_per_video", type=int, default=128)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_path", type=str, default="checkpoints/stdp_encoder.pth")
    p.add_argument("--c1", type=int, default=128)
    p.add_argument("--c2", type=int, default=384)
    p.add_argument("--stdp_lr1", type=float, default=2e-4)
    p.add_argument("--stdp_lr2", type=float, default=1e-4)
    p.add_argument("--threshold1", type=float, default=0.6)
    p.add_argument("--threshold2", type=float, default=0.6)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    normal_keyword = args.normal_keyword if args.normal_keyword.strip() else None
    dataset = EventFrameVideoDataset(
        root=args.event_frame_root,
        normal_keyword=normal_keyword,
        max_videos=args.max_videos,
        image_size=args.image_size,
        max_frames_per_video=args.max_frames_per_video,
        frame_stride=args.frame_stride,
    )

    model = STDPFeatureEncoder(
        in_channels=2,
        c1=args.c1,
        c2=args.c2,
        stdp_lr1=args.stdp_lr1,
        stdp_lr2=args.stdp_lr2,
        threshold1=args.threshold1,
        threshold2=args.threshold2,
    ).to(device)

    model.train()
    print(f"[Info] STDP videos: {len(dataset)}")
    print(f"[Info] Output feature dim: {model.out_dim}")
    print(f"[Info] Device: {device}")

    for ep in range(args.epochs):
        pbar = tqdm(range(len(dataset)), desc=f"STDP epoch {ep+1}/{args.epochs}")
        running_rate1, running_rate2, n_steps = 0.0, 0.0, 0

        for idx in pbar:
            path = dataset.get_path(idx)
            model.reset_state()

            # 流式逐帧读取，避免先把整段视频加载到 CPU 内存
            for frame in dataset.iter_video_frames(idx):
                x = frame.unsqueeze(0).to(device, non_blocking=True)
                _, spikes = model(x, learn_stdp=True)
                running_rate1 += float(spikes["rate1"].mean().item())
                running_rate2 += float(spikes["rate2"].mean().item())
                n_steps += 1

            if n_steps > 0:
                pbar.set_postfix({
                    "r1": f"{running_rate1/n_steps:.4f}",
                    "r2": f"{running_rate2/n_steps:.4f}",
                    "file": Path(path).name,
                })

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "out_dim": model.out_dim,
        "config": vars(args),
    }, save_path)
    print(f"[OK] STDP encoder saved to: {save_path}")


if __name__ == "__main__":
    main()
