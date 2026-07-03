# test_harmonizer.py
"""
Reconstruct clips with a trained Video-Harmonizer and report PSNR / SSIM for both
the discrete and continuous streams, plus the sequence-level normalized error
(SNE%, Eq. 36).

Example:
    python test_harmonizer.py --data_dir data/dummy \
        --checkpoint checkpoints/video_harmonizer_last.pth
"""
import argparse
import torch

from config import tiny_config, paper_config
from datasets import VideoClipDataset
from models import VideoHarmonizer
from models.losses import ssim


def psnr(x, y):
    mse = torch.mean((x - y) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def sne_percent(x, y):
    # Sequence-level Normalized Error (Eq. 36)
    return float(100.0 * torch.norm(x - y) / torch.norm(x).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", choices=["tiny", "paper"], default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_clips", type=int, default=8)
    args = ap.parse_args()

    cfg = tiny_config() if args.config == "tiny" else paper_config()
    device = torch.device(args.device)

    model = VideoHarmonizer(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = VideoClipDataset(args.data_dir, cfg.num_frames, cfg.image_size, is_train=False)
    print(f"[test] {min(len(ds), args.max_clips)} clips  config={args.config}")

    agg = {"psnr_d": 0, "ssim_d": 0, "psnr_c": 0, "ssim_c": 0, "sne_d": 0, "n": 0}
    with torch.no_grad():
        for i in range(min(len(ds), args.max_clips)):
            clip = ds[i].unsqueeze(0).to(device)
            out = model(clip)
            tgt, rd, rc = out["target"], out["rec_disc"], out["rec_cont"]
            agg["psnr_d"] += psnr(rd, tgt); agg["ssim_d"] += float(ssim(rd, tgt))
            agg["psnr_c"] += psnr(rc, tgt); agg["ssim_c"] += float(ssim(rc, tgt))
            agg["sne_d"] += sne_percent(tgt, rd); agg["n"] += 1

            enc = model.encode(clip)
            n_tokens = list(enc["discrete_indices"].values())[0].numel()
            print(f"clip {i:03d}  discrete PSNR={psnr(rd,tgt):6.2f} SSIM={float(ssim(rd,tgt)):.4f}"
                  f" | continuous PSNR={psnr(rc,tgt):6.2f} SSIM={float(ssim(rc,tgt)):.4f}"
                  f" | tokens/stream={n_tokens} cont_dim={enc['continuous_tokens'].shape[-1]}")

    n = agg["n"]
    print("\n[mean] "
          f"discrete PSNR={agg['psnr_d']/n:.2f} SSIM={agg['ssim_d']/n:.4f}  "
          f"continuous PSNR={agg['psnr_c']/n:.2f} SSIM={agg['ssim_c']/n:.4f}  "
          f"SNE(disc)={agg['sne_d']/n:.2f}%")


if __name__ == "__main__":
    main()
