# models/fluxhead.py
"""
FluxHead: streaming multi-head self-attention shared feature encoder (Eq. 20).

The fused multi-transform tensor Z_c (14C channels over Pt x Ps x Ps positions)
is processed *patch-wise* -- attention runs over the Pt*Ps*Ps spatio-temporal
positions **within** each patch rather than over full frames, which keeps memory
and compute tractable for ultra-high-resolution video.  Each patch is pooled to
a single d-dimensional shared embedding Z_f used by both quantization streams.
"""
import torch
import torch.nn as nn
from .attention import TransformerBlock


class FluxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pt, ps = cfg.tubelet_size, cfg.patch_size
        self.seq_len = pt * ps * ps                          # positions per patch
        in_ch = cfg.aug_channels                             # 14C

        self.proj = nn.Linear(in_ch, cfg.embed_dim)          # 14C -> d
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, cfg.embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.fluxhead_heads, cfg.mlp_ratio)
            for _ in range(cfg.fluxhead_depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, Z_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_c: (B, Np, 14C, Pt, Ps, Ps)
        Returns:
            Z_f: (B, Np, d)  shared per-patch embedding
        """
        B, Np, Caug, Pt, Ps, _ = Z_c.shape
        # (B*Np, L, 14C) with L = Pt*Ps*Ps spatio-temporal positions
        x = Z_c.reshape(B * Np, Caug, Pt * Ps * Ps).transpose(1, 2)
        x = self.proj(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                                    # pool positions -> (B*Np, d)
        return x.reshape(B, Np, self.cfg.embed_dim)
