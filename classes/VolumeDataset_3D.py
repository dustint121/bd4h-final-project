import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np




class VolumeDataset_3D(Dataset):
    def __init__(self, dataset_path, file_indices, depth_fraction=None):
        self.volumes = []
        self.segmentations = []
        
        for i in file_indices:
            data = np.load(f"{dataset_path}/{i}.npz")
            
            if "LITS17" in dataset_path:
                volume = data["volume"]  # (H, W, D)
                seg = data["segmentation"]
                # Reorder to (D, H, W)
                volume = np.transpose(volume, (2, 0, 1)).astype(np.int16)
                seg = np.transpose(seg, (2, 0, 1)).astype(np.int8)
            elif "KITS19" in dataset_path:
                volume = data["volume"].astype(np.int16)  # (D, H, W)
                seg = data["segmentation"].astype(np.int8)
            
            if depth_fraction is not None:
                total_depth = volume.shape[0]
                keep_slices = int(total_depth * depth_fraction)
                keep_slices = max(1, keep_slices)  # Ensure at least 1 slice
                start_idx = (total_depth - keep_slices) // 2
                end_idx = start_idx + keep_slices
                # Slice both volume and segmentation
                volume = volume[start_idx:end_idx]
                seg = seg[start_idx:end_idx]


            # Z-score normalization (paper Section 4.1)
            volume = self._normalize(volume, seg)
            
            self.volumes.append(volume[None, ...])  # Add channel dim
            self.segmentations.append(seg)



    def _normalize(self, volume, seg):
        """Paper-style normalization using foreground voxels"""
        mask = seg > 0
        if mask.sum() == 0:  # Handle empty masks
            return volume
        mean = volume[mask].mean()
        std = volume[mask].std()
        return ((volume - mean) / (std + 1e-8)).astype(np.float16)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = torch.FloatTensor(self.volumes[idx])  # (1, D, H, W)
        seg = torch.LongTensor(self.segmentations[idx])  # (D, H, W)
        
        # Pad all spatial dims to multiples of 8
        d, h, w = volume.shape[1], volume.shape[2], volume.shape[3]
        pad_d = (8 - (d % 8)) % 8
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        
        volume_padded = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d, 0, 0))  # (1, D+pd, H+ph, W+pw)
        seg_padded = F.pad(seg, (0, pad_w, 0, pad_h, 0, pad_d))  # (D+pd, H+ph, W+pw)
        return volume_padded, seg_padded