# train_harmonizer.py
"""
Train Video-Harmonizer on a folder of video clips.

Example (dummy CPU run):
    python make_dummy_data.py --out_dir data/dummy --num_clips 8
    python train_harmonizer.py --data_dir data/dummy --epochs 20 --batch_size 2 \
        --lr 1e-3 --output_dir checkpoints

Gumbel-Softmax temperature is annealed 1.0 -> 0.3 across training (Sec. II-D).
Note: the paper's UHD run uses lr = 1e-7 on 2xA100; the dummy run uses a larger
lr so the loss visibly decreases within a handful of epochs on CPU.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from config import tiny_config, paper_config
from datasets import VideoClipDataset
from models import VideoHarmonizer


def anneal_tau(epoch, epochs, t0, t1):
    if epochs <= 1:
        return t1
    frac = epoch / (epochs - 1)
    return t0 + (t1 - t0) * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="checkpoints")
    ap.add_argument("--config", choices=["tiny", "paper"], default="tiny")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = tiny_config() if args.config == "tiny" else paper_config()
    device = torch.device(args.device)

    ds = VideoClipDataset(args.data_dir, cfg.num_frames, cfg.image_size, is_train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)

    model = VideoHarmonizer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] config={args.config}  params={n_params/1e6:.2f}M  "
          f"patches/clip={cfg.num_patches}  device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        tau = anneal_tau(epoch, args.epochs, cfg.gumbel_tau_start, cfg.gumbel_tau_end)
        model.set_gumbel_tau(tau)
        running = {}
        for clips in dl:
            clips = clips.to(device)
            out = model(clips)
            loss, stats = model.compute_loss(out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + float(v)
        n = len(dl)
        msg = "  ".join(f"{k}={running[k]/n:.4f}" for k in
                        ["total", "l_disc", "l_cont", "commit",
                         "ssim_disc", "ssim_cont"])
        print(f"epoch {epoch+1:03d}/{args.epochs}  tau={tau:.3f}  {msg}")

    ckpt = os.path.join(args.output_dir, "video_harmonizer_last.pth")
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckpt)
    print(f"[train] saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
