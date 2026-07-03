# config.py
"""
Configuration for Video-Harmonizer (paper-exact architecture).

The DEFAULT ("paper") config documents the settings used in the paper for
ultra-high-resolution (4K/8K) video on 2xA100 GPUs.  The TINY config keeps the
*exact same architecture* but shrinks resolution / patch grid so the whole
pipeline (forward + backward) runs on a CPU in a few seconds -- this is what we
use for the local dummy-data smoke test.

Only spatial/temporal sizes and channel/embedding widths differ between the two;
every architectural component (adaptive windowing, the four transforms, FluxHead,
dual quantization, separate FluxFormer decoders, multi-objective loss) is
identical.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HarmonizerConfig:
    # ---- Video / patch geometry (Eqs. 1-2) ----
    in_chans: int = 3               # C
    num_frames: int = 8             # T
    image_size: int = 32            # H = W
    tubelet_size: int = 2           # P_t  (temporal patch size)
    patch_size: int = 8             # P_s  (spatial patch size, P_s x P_s)

    # ---- Adaptive Hann-Tukey windowing (Eqs. 5-11) ----
    v_ref: float = 0.02             # reference variance for normalization (Eq. 5)

    # ---- Transform front-end (Eqs. 12-19) ----
    weierstrass_alpha: float = 0.15  # Gaussian low-pass strength alpha (Eq. 13)
    gabor_orientations: int = 8      # Gabor filter bank orientations (Eq. 16)
    gabor_sigma_t: float = 1.0       # sigma_t   (temporal extent)
    gabor_sigma_s: float = 2.0       # sigma_s   (spatial extent)
    gabor_freq: float = 0.35         # base center frequency (cycles / pixel)
    mellin_mu: float = 1.0           # preferred log-radius mu (Eq. 17)
    mellin_sigma: float = 0.5        # log-radius bandwidth sigma (Eq. 17)

    # ---- FluxHead shared encoder (Eq. 20) ----
    embed_dim: int = 64             # d  (shared token embedding dimension)
    fluxhead_depth: int = 2
    fluxhead_heads: int = 4
    mlp_ratio: float = 4.0

    # ---- Dual quantization (Eqs. 21-27) ----
    # Discrete FusionQuantizer: residual + hierarchical codebooks
    codebook_size: int = 256        # entries per codebook
    residual_levels: int = 2        # L residual levels  (C^{1R}, C^{2R}, ...)
    hier_levels: int = 2            # L hierarchical levels (C^{1H}, C^{2H}, ...)
    commit_beta: float = 0.25       # beta in commitment loss (Eq. 32)
    # Continuous Gumbel-Softmax
    gumbel_codebook_size: int = 256
    gumbel_tau_start: float = 1.0   # temperature annealing start (Sec. II-D)
    gumbel_tau_end: float = 0.3     # temperature annealing end

    # ---- Separate FluxFormer decoders (Eqs. 28-31) ----
    fluxformer_depth: int = 2
    fluxformer_heads: int = 4

    # ---- Multi-objective loss (Eqs. 30-31) ----
    lambda_mse: float = 0.75
    lambda_ssim: float = 0.25

    # ---- derived (filled in __post_init__) ----
    num_patches: int = field(init=False, default=0)
    grid: Tuple[int, int, int] = field(init=False, default=(0, 0, 0))

    def __post_init__(self):
        assert self.num_frames % self.tubelet_size == 0, "T must be divisible by P_t"
        assert self.image_size % self.patch_size == 0, "H/W must be divisible by P_s"
        nt = self.num_frames // self.tubelet_size
        nh = self.image_size // self.patch_size
        nw = self.image_size // self.patch_size
        self.grid = (nt, nh, nw)
        self.num_patches = nt * nh * nw

    # number of channels after concatenating raw + 4 transforms (Eq. 19)
    @property
    def aug_channels(self) -> int:
        # C * (raw 1 + Weierstrass 1 + Riesz 3 + Gabor G + Mellin 1)
        return self.in_chans * (1 + 1 + 3 + self.gabor_orientations + 1)


def tiny_config() -> HarmonizerConfig:
    """Small CPU-runnable config (same architecture, tiny geometry)."""
    return HarmonizerConfig()


def paper_config() -> HarmonizerConfig:
    """
    Settings matching the paper's UHD experiments (needs A100-class GPUs).
    4K frames (2160x3840). Provided for documentation / GPU scaling; not used
    by the local dummy run.
    """
    return HarmonizerConfig(
        in_chans=3,
        num_frames=16,
        image_size=256,     # tile size; full 4K/8K frames are processed in tiles
        tubelet_size=4,     # "chunk size 4" in the paper
        patch_size=16,
        embed_dim=512,
        fluxhead_depth=6,
        fluxhead_heads=8,
        codebook_size=8192,
        gumbel_codebook_size=8192,
        fluxformer_depth=6,
        fluxformer_heads=8,
    )
