# smoke_test.py
"""
End-to-end architecture smoke test for Video-Harmonizer.

Verifies that every module (adaptive windowing, the four transforms, FluxHead,
dual quantization, both FluxFormer decoders, and the multi-objective loss) runs a
full forward + backward pass on random input and that shapes / gradients / loss
are all valid.  Runs in a few seconds on CPU.

    python smoke_test.py
"""
import torch
from config import tiny_config
from models import VideoHarmonizer


def main():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = VideoHarmonizer(cfg)
    B = 2
    x = torch.rand(B, cfg.in_chans, cfg.num_frames, cfg.image_size, cfg.image_size)

    print("=" * 68)
    print("Video-Harmonizer smoke test (tiny config)")
    print("=" * 68)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input           : {tuple(x.shape)}")
    print(f"parameters      : {n_params/1e6:.3f} M")
    print(f"patches/clip Np : {cfg.num_patches}  grid(nt,nh,nw)={cfg.grid}")
    print(f"aug channels    : {cfg.aug_channels} (= 14 x C={cfg.in_chans})")

    out = model(x)

    # ---- shape checks ----
    assert out["rec_disc"].shape == x.shape, out["rec_disc"].shape
    assert out["rec_cont"].shape == x.shape, out["rec_cont"].shape
    assert out["target"].shape == x.shape
    assert out["z_f"].shape == (B, cfg.num_patches, cfg.embed_dim)
    print(f"rec_disc        : {tuple(out['rec_disc'].shape)}  OK")
    print(f"rec_cont        : {tuple(out['rec_cont'].shape)}  OK")
    print(f"shared Z_f       : {tuple(out['z_f'].shape)}  OK")
    print(f"discrete codebooks: {list(out['disc_indices'].keys())}")
    print(f"cont soft-assign : {tuple(out['cont_probs'].shape)}")

    # ---- loss + backward ----
    loss, stats = model.compute_loss(out)
    assert torch.isfinite(loss), "loss is not finite!"
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_total = sum(1 for _ in model.parameters())
    print(f"loss            : {float(loss):.5f}  (finite OK)")
    print(f"  L_disc={float(stats['l_disc']):.4f}  L_cont={float(stats['l_cont']):.4f}"
          f"  commit={float(stats['commit']):.4f}")
    print(f"  SSIM disc={float(stats['ssim_disc']):.4f}  SSIM cont={float(stats['ssim_cont']):.4f}")
    print(f"gradients set    : {n_grad}/{n_total} parameter tensors have finite grads")
    assert n_grad == n_total, "some parameters received no gradient"

    # ---- encode() emits both streams ----
    enc = model.encode(x)
    assert enc["continuous_tokens"].shape == (B, cfg.num_patches, cfg.embed_dim)
    print(f"encode(): discrete streams={list(enc['discrete_indices'].keys())}, "
          f"continuous_tokens={tuple(enc['continuous_tokens'].shape)}")

    print("=" * 68)
    print("SMOKE TEST PASSED - full architecture runs forward+backward cleanly.")
    print("=" * 68)


if __name__ == "__main__":
    main()
