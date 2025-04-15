import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np




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
    







class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)







class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv3D(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))
        

        # Decoder (Fixed with output_padding)
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv3D(256, 128)  # 128 + 128 = 256
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv3D(128, 64)   # 64 + 64 = 128
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv3D(64, 32)    # 32 + 32 = 64
        
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        print("here")
        x1 = self.inc(x)    # (B,32,D,H,W)
        x2 = self.down1(x1) # (B,64,D/2,H/2,W/2)
        x3 = self.down2(x2) # (B,128,D/4,H/4,W/4)
        x4 = self.down3(x3) # (B,256,D/8,H/8,W/8)
        
        # Decoder
        x = self.up1(x4)    # (B,128,D/4,H/4,W/4)
        x = torch.cat([x, x3], dim=1)  # (B,256,D/4,H/4,W/4)
        x = self.conv1(x)   # (B,128,D/4,H/4,W/4)
        
        x = self.up2(x)     # (B,64,D/2,H/2,W/2)
        x = torch.cat([x, x2], dim=1)  # (B,128,D/2,H/2,W/2)
        x = self.conv2(x)   # (B,64,D/2,H/2,W/2)
        
        x = self.up3(x)     # (B,32,D,H,W)
        x = torch.cat([x, x1], dim=1)  # (B,64,D,H,W)
        x = self.conv3(x)   # (B,32,D,H,W)
        
        return self.outc(x)  # (B,n_classes,D,H,W)