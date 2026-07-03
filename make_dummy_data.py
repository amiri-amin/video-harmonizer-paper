# make_dummy_data.py
"""
Generate a small synthetic dataset of video clips for the Video-Harmonizer
smoke test.  Each clip is a moving structured pattern (gradients + drifting
sinusoids + a bouncing box) so reconstructions are non-trivial and PSNR/SSIM are
meaningful.  Clips are saved as (C, T, H, W) float tensors in [0,1].

Usage:
    python make_dummy_data.py --out_dir data/dummy --num_clips 8
"""
import os
import argparse
import math
import torch


def make_clip(C, T, H, W, seed):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H),
                            torch.linspace(0, 1, W), indexing="ij")
    phase = torch.rand(1, generator=g).item() * math.pi
    fx = 3 + torch.randint(0, 4, (1,), generator=g).item()
    fy = 3 + torch.randint(0, 4, (1,), generator=g).item()
    clip = torch.zeros(C, T, H, W)
    for t in range(T):
        drift = 2 * math.pi * t / T
        base = 0.5 + 0.5 * torch.sin(2 * math.pi * (fx * xx + fy * yy) + drift + phase)
        # bouncing bright box for localized high-frequency motion
        bx = int((0.5 + 0.45 * math.sin(drift)) * (W - W // 5))
        by = int((0.5 + 0.45 * math.cos(drift)) * (H - H // 5))
        box = torch.zeros(H, W)
        box[by:by + H // 5, bx:bx + W // 5] = 1.0
        frame = (base + 0.6 * box).clamp(0, 1)
        for c in range(C):
            clip[c, t] = (frame * (0.6 + 0.4 * ((c + 1) / C))).clamp(0, 1)
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/dummy")
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.num_clips):
        clip = make_clip(args.channels, args.frames, args.size, args.size, seed=1000 + i)
        torch.save(clip, os.path.join(args.out_dir, f"clip_{i:03d}.pt"))
    print(f"[make_dummy_data] wrote {args.num_clips} clips "
          f"({args.channels}x{args.frames}x{args.size}x{args.size}) to {args.out_dir}")


if __name__ == "__main__":
    main()
