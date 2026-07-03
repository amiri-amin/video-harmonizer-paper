# models/losses.py
"""
Multi-objective reconstruction loss (Eqs. 30-31).

Each stream is trained against the original windowed patches Z_p with a weighted
combination of MSE and an SSIM-based structural term:

    L_* = lambda_MSE || z_recon - Z_p ||^2 + lambda_SSIM ( 1 - SSIM(z_recon, Z_p) )

with lambda_MSE = 0.75 and lambda_SSIM = 0.25.  The total loss adds the discrete
and continuous stream losses plus the discrete commitment loss (Eq. 32):

    L = L_disc + L_cont + L_commit
"""
import torch
import torch.nn.functional as F


def _gaussian_window(win_size=7, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return (g[:, None] * g[None, :])                         # (win, win)


def ssim(x, y, win_size=7, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """
    Mean SSIM between two video volumes.

    Args:
        x, y: (B, C, T, H, W) in [0,1]-ish range
    Returns:
        scalar mean SSIM.
    """
    B, C, T, H, W = x.shape
    win = min(win_size, H, W)
    if win % 2 == 0:
        win -= 1
    win = max(win, 1)
    x2 = x.reshape(B * C * T, 1, H, W)
    y2 = y.reshape(B * C * T, 1, H, W)
    if win == 1:
        # degenerate spatial size: fall back to global statistics
        mu_x, mu_y = x2.mean(), y2.mean()
        vx, vy = x2.var(), y2.var()
        vxy = ((x2 - mu_x) * (y2 - mu_y)).mean()
        s = ((2 * mu_x * mu_y + C1) * (2 * vxy + C2)) / \
            ((mu_x ** 2 + mu_y ** 2 + C1) * (vx + vy + C2))
        return s

    w = _gaussian_window(win, sigma, x.device, x.dtype).reshape(1, 1, win, win)
    pad = win // 2
    mu_x = F.conv2d(x2, w, padding=pad)
    mu_y = F.conv2d(y2, w, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x2 * x2, w, padding=pad) - mu_x2
    sig_y = F.conv2d(y2 * y2, w, padding=pad) - mu_y2
    sig_xy = F.conv2d(x2 * y2, w, padding=pad) - mu_xy
    smap = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
           ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
    return smap.mean()


def stream_loss(recon, target, lambda_mse=0.75, lambda_ssim=0.25):
    """L_* for one stream (Eq. 30 / 31). recon, target: (B,C,T,H,W)."""
    mse = F.mse_loss(recon, target)
    s = ssim(recon, target)
    return lambda_mse * mse + lambda_ssim * (1.0 - s), mse.detach(), s.detach()
