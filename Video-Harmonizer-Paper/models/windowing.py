# models/windowing.py
"""
Video input, spatio-temporal patch extraction, and Adaptive Hann-Tukey windowing.

Implements Eqs. (1)-(11) of the Video-Harmonizer paper:

  * Patch extraction              -> Eq. (2)
  * Per-patch mean / variance     -> Eqs. (3)-(4)
  * Normalized clipped variance   -> Eq. (5)
  * Hann window                   -> Eq. (6)
  * Adaptive 1D interpolation     -> Eq. (7)
  * Temporal / spatial windows    -> Eqs. (8)-(9)
  * Separable 3D window           -> Eq. (10)
  * Windowed patch  Z_p = w3D (x) -> Eq. (11)

Patches are represented channel-first as (B, Np, C, Pt, Ps, Ps).
"""
import math
import torch
import torch.nn as nn


def _hann(n: int, device, dtype) -> torch.Tensor:
    """Standard Hann window of length n (Eq. 6). Length-1 -> flat window of 1."""
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype)
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * k / (n - 1)))


def patchify(x: torch.Tensor, pt: int, ps: int) -> torch.Tensor:
    """
    Partition a video into non-overlapping spatio-temporal patches (Eq. 2).

    Args:
        x:  (B, C, T, H, W)
    Returns:
        patches: (B, Np, C, Pt, Ps, Ps)  with Np = (T/Pt)(H/Ps)(W/Ps)
    """
    B, C, T, H, W = x.shape
    nt, nh, nw = T // pt, H // ps, W // ps
    x = x.reshape(B, C, nt, pt, nh, ps, nw, ps)
    # -> (B, nt, nh, nw, C, pt, ps, ps)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.reshape(B, nt * nh * nw, C, pt, ps, ps)


def unpatchify(patches: torch.Tensor, grid, T: int, H: int, W: int) -> torch.Tensor:
    """
    Inverse of patchify: reassemble patches into a video volume.

    Args:
        patches: (B, Np, C, Pt, Ps, Ps)
        grid:    (nt, nh, nw)
    Returns:
        x: (B, C, T, H, W)
    """
    B, Np, C, pt, ps, _ = patches.shape
    nt, nh, nw = grid
    x = patches.reshape(B, nt, nh, nw, C, pt, ps, ps)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()   # (B, C, nt, pt, nh, ps, nw, ps)
    return x.reshape(B, C, T, H, W)


class AdaptiveHannTukeyWindow(nn.Module):
    """
    Variance-adaptive separable 3D windowing (Eqs. 3-11).

    High-variance patches (sharp structure) are barely attenuated (w3D ~ 1),
    while low-variance patches are smoothly tapered by a Hann profile, reducing
    seams between neighbouring patches without oversmoothing detail.
    """

    def __init__(self, pt: int, ps: int, v_ref: float = 0.02):
        super().__init__()
        self.pt = pt
        self.ps = ps
        self.v_ref = v_ref

    def forward(self, patches: torch.Tensor):
        """
        Args:
            patches: (B, Np, C, Pt, Ps, Ps)  raw patches x_ij
        Returns:
            Z_p:  (B, Np, C, Pt, Ps, Ps)  windowed patches
            vtil: (B, Np)                 normalized clipped variance (for logging)
        """
        B, Np, C, pt, ps, _ = patches.shape
        device, dtype = patches.device, patches.dtype

        # Per-patch mean/variance over all elements (Eqs. 3-4)
        flat = patches.reshape(B, Np, -1)
        mu = flat.mean(dim=-1)                      # (B, Np)   Eq. (3)
        var = flat.var(dim=-1, unbiased=False)      # (B, Np)   Eq. (4)

        # Normalized clipped variance (Eq. 5)
        vtil = torch.clamp(var / self.v_ref, 0.0, 1.0)   # (B, Np)

        # Base Hann windows (Eq. 6)
        ht = _hann(pt, device, dtype)               # (Pt,)
        hs = _hann(ps, device, dtype)               # (Ps,)

        # Adaptive 1D windows (Eqs. 7-9): w = (1 - vtil) h + vtil
        v = vtil.reshape(B, Np, 1)
        wt = (1.0 - v) * ht.reshape(1, 1, pt) + v          # (B, Np, Pt)
        ws_h = (1.0 - v) * hs.reshape(1, 1, ps) + v        # (B, Np, Ps)
        ws_w = (1.0 - v) * hs.reshape(1, 1, ps) + v        # (B, Np, Ps)

        # Separable 3D window (Eq. 10): w3D = wt * ws_h * ws_w
        w3d = (wt.reshape(B, Np, pt, 1, 1)
               * ws_h.reshape(B, Np, 1, ps, 1)
               * ws_w.reshape(B, Np, 1, 1, ps))            # (B, Np, Pt, Ps, Ps)

        # Apply elementwise (Eq. 11): broadcast over channels
        Z_p = w3d.unsqueeze(2) * patches                   # (B, Np, C, Pt, Ps, Ps)
        return Z_p, vtil
