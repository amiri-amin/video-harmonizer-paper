# models/fluxformer.py
"""
FluxFormer: token encoder-decoder stack that reconstructs windowed patches
(Eqs. 28-31).

Two *separate* FluxFormer instances are used -- one for the discrete stream and
one for the continuous stream -- so each can specialize (sharp/edge-preserving vs
smooth/diffusion-friendly).  Branching happens only here, after the shared
FluxHead embedding; the tokenization pathway upstream is common to both streams.

Given per-patch tokens (B, Np, d), the decoder produces per-patch pixels
(C*Pt*Ps*Ps) which are reshaped back to (B, Np, C, Pt, Ps, Ps).
"""
import torch
import torch.nn as nn
from .attention import TransformerBlock


class FluxFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_patches, d))
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.fluxformer_heads, cfg.mlp_ratio)
            for _ in range(cfg.fluxformer_depth)
        ])
        self.norm = nn.LayerNorm(d)
        patch_dim = cfg.in_chans * cfg.tubelet_size * cfg.patch_size * cfg.patch_size
        self.head = nn.Linear(d, patch_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, Np, d)
        Returns:
            patches: (B, Np, C, Pt, Ps, Ps) reconstructed windowed patches
        """
        cfg = self.cfg
        x = tokens + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)                                     # (B, Np, C*Pt*Ps*Ps)
        B, Np, _ = x.shape
        return x.reshape(B, Np, cfg.in_chans, cfg.tubelet_size,
                         cfg.patch_size, cfg.patch_size)
