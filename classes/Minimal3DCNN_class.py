import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class MedicalVolumeDataset_3D(Dataset):
    def __init__(self, dataset_path, file_indices):
        self.volumes = []
        self.segmentations = []
        
        for i in file_indices:
            data = np.load(f"{dataset_path}/{i}.npz")
            
            if "LITS17" in dataset_path:
                volume = data["volume"]  # (H, W, D)
                seg = data["segmentation"]
                # Reorder to (D, H, W)
                volume = np.transpose(volume, (2, 0, 1)).astype(np.int32)
                seg = np.transpose(seg, (2, 0, 1))
            elif "KITS19" in dataset_path:
                volume = data["volume"].astype(np.int32)  # (D, H, W)
                seg = data["segmentation"]
            
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
        return (volume - mean) / (std + 1e-8)

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











# Minimal 3D CNN Model
class Minimal3DCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        print("here1")
        x = self.conv(x)
        print("here2")
        x = torch.relu(x)
        print("here3")
        x = self.pool(x)  # shape: (batch, 4, 1, 1, 1)
        x = x.view(x.size(0), -1)
        print("here4")
        return self.fc(x)
