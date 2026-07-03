# models/transforms.py
"""
Transform-based augmentation front-end (Eqs. 12-19).

Four complementary transforms are applied *in parallel* to every windowed patch
Z_p, each preserving the patch shape (C, Pt, Ps, Ps):

  * Weierstrass  (Gaussian low-pass)        -> Eqs. (12)-(13)    ->  C  channels
  * Riesz        (directional high-pass x3)  -> Eq.  (14)         -> 3C  channels
  * Gabor        (localized band-pass, G=8)  -> Eqs. (15)-(16)    -> 8C  channels
  * Mellin       (log-radial, spatial only)  -> Eq.  (17)         ->  C  channels

They are concatenated with the raw windowed patch (Eq. 19) to give
C(1 + 1 + 3 + 8 + 1) = 14C channels.

All patches enter/leave as (B, Np, C, Pt, Ps, Ps).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _freq_grids(pt, ps, device, dtype):
    """Angular-frequency grids (2*pi*fftfreq) for a Pt x Ps x Ps volume."""
    wt = 2 * math.pi * torch.fft.fftfreq(pt, device=device).to(dtype)
    wh = 2 * math.pi * torch.fft.fftfreq(ps, device=device).to(dtype)
    ww = 2 * math.pi * torch.fft.fftfreq(ps, device=device).to(dtype)
    WT, WH, WW = torch.meshgrid(wt, wh, ww, indexing="ij")
    return WT, WH, WW


class TransformBank(nn.Module):
    """Parallel Weierstrass / Riesz / Gabor / Mellin transform bank."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pt, ps = cfg.tubelet_size, cfg.patch_size

        # ---- Weierstrass Gaussian low-pass filter H_alpha(w) = exp(-alpha |w|^2)  (Eq. 13)
        WT, WH, WW = _freq_grids(pt, ps, torch.device("cpu"), torch.float32)
        w2 = WT ** 2 + WH ** 2 + WW ** 2                     # |w|^2
        self.register_buffer("weier_H", torch.exp(-cfg.weierstrass_alpha * w2))

        # ---- Riesz normalized derivative factors  w_k / |w|  (Eq. 14)
        wmag = torch.sqrt(w2)
        wmag_safe = wmag.clone()
        wmag_safe[wmag_safe == 0] = 1.0                      # avoid /0 at DC
        # store the three real multipliers r_k = w_k/|w| (the -i is applied at use)
        self.register_buffer("riesz_t", WT / wmag_safe)
        self.register_buffer("riesz_h", WH / wmag_safe)
        self.register_buffer("riesz_w", WW / wmag_safe)

        # ---- Gabor filter bank (G orientations)  (Eq. 16)
        self.gabor_kt = pt if pt % 2 == 1 else max(1, pt - 1)
        self.gabor_ks = ps if ps % 2 == 1 else max(1, ps - 1)
        gabor = self._build_gabor_bank()                     # (G, Kt, Ks, Ks)
        self.register_buffer("gabor_kernels", gabor)

        # ---- Mellin log-radial spatial filter  (Eq. 17)
        self.mellin_ks = ps if ps % 2 == 1 else max(1, ps - 1)
        self.register_buffer("mellin_kernel", self._build_mellin_kernel())  # (Ks, Ks)

    # ------------------------------------------------------------------ Gabor
    def _build_gabor_bank(self):
        cfg = self.cfg
        Kt, Ks = self.gabor_kt, self.gabor_ks
        tt = torch.arange(Kt, dtype=torch.float32) - Kt // 2
        hh = torch.arange(Ks, dtype=torch.float32) - Ks // 2
        ww = torch.arange(Ks, dtype=torch.float32) - Ks // 2
        TT, HH, WW = torch.meshgrid(tt, hh, ww, indexing="ij")
        env = torch.exp(-(TT ** 2) / (2 * cfg.gabor_sigma_t ** 2)
                        - (HH ** 2 + WW ** 2) / (2 * cfg.gabor_sigma_s ** 2))
        kernels = []
        ut = cfg.gabor_freq * 0.5                            # temporal center freq
        for o in range(cfg.gabor_orientations):
            theta = math.pi * o / cfg.gabor_orientations
            uh = cfg.gabor_freq * math.cos(theta)
            uw = cfg.gabor_freq * math.sin(theta)
            carrier = torch.cos(2 * math.pi * (ut * TT + uh * HH + uw * WW))
            g = env * carrier
            g = g - g.mean()                                 # zero-DC -> true band-pass
            kernels.append(g)
        return torch.stack(kernels, dim=0)                   # (G, Kt, Ks, Ks)

    # ----------------------------------------------------------------- Mellin
    def _build_mellin_kernel(self):
        cfg = self.cfg
        Ks = self.mellin_ks
        hh = torch.arange(Ks, dtype=torch.float32) - Ks // 2
        ww = torch.arange(Ks, dtype=torch.float32) - Ks // 2
        HH, WW = torch.meshgrid(hh, ww, indexing="ij")
        r = torch.sqrt(HH ** 2 + WW ** 2) + 1e-3
        m = torch.exp(-((torch.log(r) - cfg.mellin_mu) ** 2) / (2 * cfg.mellin_sigma ** 2))
        m = m / m.sum()                                      # normalize
        return m                                             # (Ks, Ks)

    # ------------------------------------------------------------- transforms
    def weierstrass(self, z):
        """Gaussian low-pass (Eqs. 12-13). z:(M,C,Pt,Ps,Ps) -> (M,C,Pt,Ps,Ps)."""
        Z = torch.fft.fftn(z, dim=(-3, -2, -1))
        Z = Z * self.weier_H.to(Z.dtype)
        return torch.fft.ifftn(Z, dim=(-3, -2, -1)).real

    def riesz(self, z):
        """Directional high-pass, 3 components (Eq. 14). -> (M,3C,Pt,Ps,Ps)."""
        Z = torch.fft.fftn(z, dim=(-3, -2, -1))
        outs = []
        for r in (self.riesz_t, self.riesz_h, self.riesz_w):
            # multiply by -i * (w_k/|w|)
            factor = (-1j) * r.to(Z.dtype)
            outs.append(torch.fft.ifftn(Z * factor, dim=(-3, -2, -1)).real)
        return torch.cat(outs, dim=1)                        # concat along channel

    def gabor(self, z):
        """Localized band-pass, G orientations (Eqs. 15-16). -> (M,G*C,Pt,Ps,Ps)."""
        M, C, Pt, Ps, _ = z.shape
        pad = (self.gabor_kt // 2, self.gabor_ks // 2, self.gabor_ks // 2)
        outs = []
        for o in range(self.cfg.gabor_orientations):
            k = self.gabor_kernels[o].to(z.dtype)            # (Kt,Ks,Ks)
            weight = k.reshape(1, 1, *k.shape).repeat(C, 1, 1, 1, 1)  # depthwise
            outs.append(F.conv3d(z, weight, padding=pad, groups=C))
        return torch.cat(outs, dim=1)

    def mellin(self, z):
        """Log-radial scale filter, spatial only (Eq. 17). -> (M,C,Pt,Ps,Ps)."""
        M, C, Pt, Ps, _ = z.shape
        zt = z.reshape(M * C, 1, Pt, Ps, Ps)
        # temporal kernel size 1 -> pure spatial filtering, applied per frame
        k = self.mellin_kernel.to(z.dtype)                   # (Ks,Ks)
        weight = k.reshape(1, 1, 1, *k.shape)                # (1,1,1,Ks,Ks)
        pad = (0, self.mellin_ks // 2, self.mellin_ks // 2)
        out = F.conv3d(zt, weight, padding=pad)
        return out.reshape(M, C, Pt, Ps, Ps)

    # ---------------------------------------------------------------- forward
    def forward(self, Z_p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_p: (B, Np, C, Pt, Ps, Ps) windowed patches
        Returns:
            Z_c: (B, Np, 14C, Pt, Ps, Ps) concatenated multi-transform features (Eq. 19)
        """
        B, Np, C, Pt, Ps, _ = Z_p.shape
        z = Z_p.reshape(B * Np, C, Pt, Ps, Ps)

        z_w = self.weierstrass(z)                            # C
        z_r = self.riesz(z)                                  # 3C
        z_g = self.gabor(z)                                  # 8C
        z_m = self.mellin(z)                                 # C

        Z_c = torch.cat([z, z_w, z_r, z_g, z_m], dim=1)      # (B*Np, 14C, Pt,Ps,Ps)
        return Z_c.reshape(B, Np, -1, Pt, Ps, Ps)
