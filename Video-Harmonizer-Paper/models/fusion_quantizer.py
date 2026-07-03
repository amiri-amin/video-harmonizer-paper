# models/fusion_quantizer.py
"""
FusionQuantizer: discrete quantization with residual + hierarchical codebooks
(Eqs. 21-26) plus the commitment loss (Eq. 32).

Residual path  (Eqs. 21-23):
    c^{1R} = argmin_c || Z_f - c ||^2        (codebook C^{1R})
    r^{1R} = Z_f - c^{1R}
    c^{2R} = argmin_c || r^{1R} - c ||^2     (codebook C^{2R})   ... up to L levels
Hierarchical path (Eqs. 24-25):
    c^{lH} = argmin_c || Z_f - c ||^2        (codebooks C^{lH}), each on Z_f directly
Aggregation (Eq. 26):
    z_disc = 1/2 ( c^{1R}+c^{1H}+c^{2R}+c^{2H}+...+c^{LR}+c^{LH} )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Codebook(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / size, 1.0 / size)

    def quantize(self, x):
        """x:(M,d) -> (quantized (M,d), indices (M,))  nearest-neighbour lookup."""
        w = self.embedding.weight                            # (K, d)
        d = (x.pow(2).sum(1, keepdim=True)
             - 2 * x @ w.t()
             + w.pow(2).sum(1))                              # (M, K) squared L2
        idx = d.argmin(1)
        return F.embedding(idx, w), idx


class FusionQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.beta = cfg.commit_beta
        self.residual = nn.ModuleList([_Codebook(cfg.codebook_size, d)
                                       for _ in range(cfg.residual_levels)])
        self.hier = nn.ModuleList([_Codebook(cfg.codebook_size, d)
                                   for _ in range(cfg.hier_levels)])

    def forward(self, Z_f: torch.Tensor):
        """
        Args:
            Z_f: (B, Np, d)
        Returns:
            z_disc: (B, Np, d)  discrete tokens (straight-through)
            loss:   scalar commitment loss (Eq. 32)
            indices: dict of index tensors per codebook (for token-usage logging)
        """
        B, Np, d = Z_f.shape
        z = Z_f.reshape(-1, d)                               # (M, d)

        centroids = []          # all c^{lR}, c^{lH} for aggregation
        commit = z.new_zeros(())
        indices = {}

        # --- residual path (Eqs. 21-23) ---
        residual = z
        for l, cb in enumerate(self.residual):
            q, idx = cb.quantize(residual)
            centroids.append(q)
            indices[f"res{l+1}"] = idx.reshape(B, Np)
            # commitment loss (Eq. 32): quantizing the *current residual*
            commit = commit + F.mse_loss(q, residual.detach()) \
                            + self.beta * F.mse_loss(residual, q.detach())
            residual = residual - q.detach()                 # next-level residual

        # --- hierarchical path (Eqs. 24-25): each level quantizes Z_f directly ---
        for l, cb in enumerate(self.hier):
            q, idx = cb.quantize(z)
            centroids.append(q)
            indices[f"hier{l+1}"] = idx.reshape(B, Np)
            commit = commit + F.mse_loss(q, z.detach()) \
                            + self.beta * F.mse_loss(z, q.detach())

        # --- aggregation (Eq. 26) ---
        z_disc = 0.5 * torch.stack(centroids, 0).sum(0)      # (M, d)

        # straight-through estimator so gradients reach the encoder
        z_disc = z + (z_disc - z).detach()
        return z_disc.reshape(B, Np, d), commit, indices
