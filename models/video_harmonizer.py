# models/video_harmonizer.py
"""
Video-Harmonizer: unified video tokenizer (paper-exact architecture).

End-to-end pipeline (Fig. 1 / Section II):

    raw video  X in R^{T x H x W x C}
      -> patch extraction                         (Eqs. 1-2)
      -> adaptive Hann-Tukey 3D windowing  Z_p     (Eqs. 3-11)
      -> parallel transforms  (Weierstrass/Riesz/Gabor/Mellin) -> concat Z_c  (Eqs. 12-19)
      -> FluxHead streaming self-attention -> shared embedding Z_f            (Eq. 20)
      -> dual quantization:
             discrete FusionQuantizer  z_disc      (Eqs. 21-26, 32)
             continuous Gumbel-Softmax z_cont       (Eq. 27)
      -> separate FluxFormer decoders  -> reconstruct windowed patches        (Eqs. 28-31)
      -> multi-objective MSE + SSIM loss                                      (Eqs. 30-31)

The model natively emits both discrete indices (for autoregressive / LLM-style
decoders) and continuous embeddings (for diffusion decoders) with no conversion
layer.
"""
import torch
import torch.nn as nn

from .windowing import patchify, unpatchify, AdaptiveHannTukeyWindow
from .transforms import TransformBank
from .fluxhead import FluxHead
from .fusion_quantizer import FusionQuantizer
from .gumbel_quantizer import GumbelQuantizer
from .fluxformer import FluxFormer
from . import losses


class VideoHarmonizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.window = AdaptiveHannTukeyWindow(cfg.tubelet_size, cfg.patch_size, cfg.v_ref)
        self.transforms = TransformBank(cfg)
        self.fluxhead = FluxHead(cfg)
        self.fusion_q = FusionQuantizer(cfg)         # discrete stream
        self.gumbel_q = GumbelQuantizer(cfg)         # continuous stream
        self.decoder_disc = FluxFormer(cfg)          # separate discrete decoder
        self.decoder_cont = FluxFormer(cfg)          # separate continuous decoder

    # ---- temperature annealing hook for Gumbel-Softmax (Sec. II-D) ----
    def set_gumbel_tau(self, tau: float):
        self.gumbel_q.set_tau(tau)

    def _shared_encode(self, x):
        """Run the shared tokenization pathway up to the FluxHead embedding Z_f."""
        B, C, T, H, W = x.shape
        patches = patchify(x, self.cfg.tubelet_size, self.cfg.patch_size)
        Z_p, vtil = self.window(patches)             # windowed patches (Eqs. 3-11)
        Z_c = self.transforms(Z_p)                   # multi-transform concat (Eqs. 12-19)
        Z_f = self.fluxhead(Z_c)                     # shared embedding (Eq. 20)
        return Z_p, Z_f, vtil, (T, H, W)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) video in ~[0,1]
        Returns dict with reconstructions, target (windowed video), losses, tokens.
        """
        cfg = self.cfg
        Z_p, Z_f, vtil, (T, H, W) = self._shared_encode(x)

        # dual quantization from the shared embedding
        z_disc, commit_loss, disc_idx = self.fusion_q(Z_f)      # Eqs. 21-26, 32
        z_cont, cont_probs = self.gumbel_q(Z_f)                 # Eq. 27

        # separate reconstruction (Eqs. 28-31)
        rec_disc_p = self.decoder_disc(z_disc)
        rec_cont_p = self.decoder_cont(z_cont)

        rec_disc = unpatchify(rec_disc_p, cfg.grid, T, H, W)
        rec_cont = unpatchify(rec_cont_p, cfg.grid, T, H, W)
        target = unpatchify(Z_p, cfg.grid, T, H, W)            # windowed target

        return {
            "rec_disc": rec_disc,
            "rec_cont": rec_cont,
            "target": target,
            "commit_loss": commit_loss,
            "disc_indices": disc_idx,
            "cont_probs": cont_probs,
            "vtil": vtil,
            "z_f": Z_f,
        }

    def compute_loss(self, out):
        """Total multi-objective loss L = L_disc + L_cont + L_commit (Eqs. 30-32)."""
        cfg = self.cfg
        l_disc, mse_d, ssim_d = losses.stream_loss(
            out["rec_disc"], out["target"], cfg.lambda_mse, cfg.lambda_ssim)
        l_cont, mse_c, ssim_c = losses.stream_loss(
            out["rec_cont"], out["target"], cfg.lambda_mse, cfg.lambda_ssim)
        total = l_disc + l_cont + out["commit_loss"]
        stats = {
            "total": total.detach(),
            "l_disc": l_disc.detach(), "l_cont": l_cont.detach(),
            "commit": out["commit_loss"].detach(),
            "mse_disc": mse_d, "ssim_disc": ssim_d,
            "mse_cont": mse_c, "ssim_cont": ssim_c,
        }
        return total, stats

    @torch.no_grad()
    def encode(self, x):
        """
        Tokenize a video, emitting BOTH token streams (no conversion layer):
            discrete_indices: dict of per-codebook index tensors (B, Np)
            continuous_tokens: (B, Np, d) continuous embeddings
        """
        self.eval()
        _, Z_f, _, _ = self._shared_encode(x)
        _, _, disc_idx = self.fusion_q(Z_f)
        z_cont, _ = self.gumbel_q(Z_f)
        return {"discrete_indices": disc_idx, "continuous_tokens": z_cont}
