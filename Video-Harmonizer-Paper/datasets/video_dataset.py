# datasets/video_dataset.py
"""
Minimal video-clip dataset for Video-Harmonizer.

Supports two sources:
  * pre-extracted clip tensors  (*.pt / *.npy) of shape (C, T, H, W) in [0,1]
    -- used by the dummy smoke test (no video codec required)
  * video files (*.mp4/*.avi/*.mov/*.mkv) decoded with torchvision, center-cropped
    and resized to the model resolution.

Every item is returned as a (C, T, H, W) float tensor in [0,1].
"""
import os
import glob
import torch
import torch.nn.functional as F


class VideoClipDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_frames, image_size, is_train=True):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_train = is_train
        exts = ("*.pt", "*.npy", "*.mp4", "*.avi", "*.mov", "*.mkv")
        self.files = []
        for e in exts:
            self.files += glob.glob(os.path.join(root_dir, e))
        self.files.sort()
        if not self.files:
            raise FileNotFoundError(f"No clips/videos found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def _load_tensor_clip(self, path):
        if path.endswith(".pt"):
            clip = torch.load(path, map_location="cpu")
        else:  # .npy
            import numpy as np
            clip = torch.from_numpy(np.load(path))
        return clip.float()

    def _load_video(self, path):
        import torchvision.io as io
        video, _, _ = io.read_video(path, pts_unit="sec")   # (T,H,W,C) uint8
        if video.shape[0] == 0:
            raise RuntimeError(f"empty video {path}")
        return video.permute(3, 0, 1, 2).float() / 255.0     # (C,T,H,W)

    def _fit_length(self, clip):
        C, T, H, W = clip.shape
        if T >= self.num_frames:
            start = (torch.randint(0, T - self.num_frames + 1, ()).item()
                     if self.is_train else (T - self.num_frames) // 2)
            clip = clip[:, start:start + self.num_frames]
        else:  # pad by repeating last frame
            pad = self.num_frames - T
            clip = torch.cat([clip, clip[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        return clip

    def _fit_spatial(self, clip):
        C, T, H, W = clip.shape
        s = self.image_size
        if (H, W) != (s, s):
            clip = F.interpolate(clip, size=(s, s), mode="bilinear", align_corners=False)
        return clip.clamp(0, 1)

    def __getitem__(self, idx):
        path = self.files[idx]
        clip = (self._load_tensor_clip(path)
                if path.endswith((".pt", ".npy")) else self._load_video(path))
        if clip.dim() != 4:
            raise ValueError(f"expected (C,T,H,W), got {tuple(clip.shape)} for {path}")
        clip = self._fit_length(clip)
        clip = self._fit_spatial(clip)
        return clip
