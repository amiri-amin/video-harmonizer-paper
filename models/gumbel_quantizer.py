# models/gumbel_quantizer.py
"""
Continuous Gumbel-Softmax quantization (Eq. 27).

    g_m ~ Gumbel(0,1)
    alpha_m = softmax_m( (-|| Z_f - c_m ||^2 + g_m) / tau )
    z_cont  = sum_m alpha_m c_m

The temperature tau is annealed from 1.0 to 0.3 over training (Sec. II-D); it is
supplied per-step by the training loop via `set_tau`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.codebook = nn.Embedding(cfg.gumbel_codebook_size, cfg.embed_dim)
        nn.init.uniform_(self.codebook.weight,
                         -1.0 / cfg.gumbel_codebook_size,
                         1.0 / cfg.gumbel_codebook_size)
        self.tau = cfg.gumbel_tau_start

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def forward(self, Z_f: torch.Tensor):
        """
        Args:
            Z_f: (B, Np, d)
        Returns:
            z_cont: (B, Np, d) continuous soft-quantized tokens
            probs:  (B, Np, K) soft assignment weights (for logging)
        """
        B, Np, d = Z_f.shape
        z = Z_f.reshape(-1, d)                               # (M, d)
        w = self.codebook.weight                             # (K, d)

        # logits = -|| Z_f - c_m ||^2   (Eq. 27 numerator, pre-Gumbel)
        neg_dist = -(z.pow(2).sum(1, keepdim=True)
                     - 2 * z @ w.t()
                     + w.pow(2).sum(1))                       # (M, K)

        # differentiable Gumbel-Softmax sampling with temperature tau
        alpha = F.gumbel_softmax(neg_dist, tau=self.tau, hard=False, dim=-1)
        z_cont = alpha @ w                                    # (M, d)
        return z_cont.reshape(B, Np, d), alpha.reshape(B, Np, -1)
