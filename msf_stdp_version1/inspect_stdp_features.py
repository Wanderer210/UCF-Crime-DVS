import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_root", type=str, required=True)
    args = p.parse_args()

    files = sorted(Path(args.feature_root).rglob("*.npy"))
    print(f"[Info] npy feature files: {len(files)}")
    for f in files[:20]:
        arr = np.load(f)
        print(f"{f} -> shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")


if __name__ == "__main__":
    main()
