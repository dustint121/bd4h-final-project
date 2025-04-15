from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalVolumeDataset2_5D(Dataset):
    def __init__(self, dataset_path, file_indices, num_slices=3):
        self.num_slices = num_slices
        self.half_slices = num_slices // 2
        self.volumes = []
        self.segmentations = []
        
        for i in file_indices:
            data = np.load(f"{dataset_path}/{i}.npz")
            volume, seg = None, None
            # volume = np.transpose(data["volume"], (2, 0, 1))  # (D, H, W)
            # seg = np.transpose(data["segmentation"], (2, 0, 1))
            if "LITS17" in dataset_path:
                volume = data["volume"]  # (H, W, D)
                seg = data["segmentation"]
                # Reorder to (D, H, W)
                volume = np.transpose(volume, (2, 0, 1)).astype(np.int32)
                seg = np.transpose(seg, (2, 0, 1))
            elif "KITS19" in dataset_path:
                volume = data["volume"].astype(np.int32)  # (D, H, W)
                seg = data["segmentation"]

            # Normalize using foreground voxels
            volume = self._normalize(volume, seg)
            
            self.volumes.append(volume)
            self.segmentations.append(seg)

    def _normalize(self, volume, seg):
        mask = seg > 0
        if mask.sum() == 0:
            return volume
        mean = volume[mask].mean()
        std = volume[mask].std()
        return (volume - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = self.volumes[idx]  # (D, H, W)
        seg = self.segmentations[idx]
        
        # Create 2.5D slices with adjacent slices as channels
        slices = []
        masks = []
        for z in range(volume.shape[0]):
            # Handle edge cases by replicating first/last slice
            start = max(0, z - self.half_slices)
            end = min(volume.shape[0], z + self.half_slices + 1)
            slice_stack = []
            
            # Pad with replicated slices if needed
            while len(slice_stack) < self.num_slices:
                if start < 0:
                    slice_stack.append(volume[0])
                    start += 1
                elif end >= volume.shape[0]:
                    slice_stack.append(volume[-1])
                    end -= 1
                else:
                    slice_stack.append(volume[start])
                    start += 1
            
            slice_stack = np.stack(slice_stack)  # (C, H, W)
            slices.append(slice_stack)
            masks.append(seg[z])  # Current slice mask
        
        volume_2_5d = np.stack(slices)  # (D, C, H, W)
        masks = np.stack(masks)         # (D, H, W)
        
        return torch.FloatTensor(volume_2_5d), torch.LongTensor(masks)

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet2_5D(nn.Module):
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv2D(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv2D(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv2D(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv2D(64, 32)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x1 = self.inc(x)    # (B,32,H,W)
        x2 = self.down1(x1) # (B,64,H/2,W/2)
        x3 = self.down2(x2) # (B,128,H/4,W/4)
        x4 = self.down3(x3) # (B,256,H/8,W/8)
        
        x = self.up1(x4)    # (B,128,H/4,W/4)
        x = torch.cat([x, x3], dim=1) # (B,256,H/4,W/4)
        x = self.conv1(x)   # (B,128,H/4,W/4)
        
        x = self.up2(x)     # (B,64,H/2,W/2)
        x = torch.cat([x, x2], dim=1) # (B,128,H/2,W/2)
        x = self.conv2(x)   # (B,64,H/2,W/2)
        
        x = self.up3(x)     # (B,32,H,W)
        x = torch.cat([x, x1], dim=1) # (B,64,H,W)
        x = self.conv3(x)   # (B,32,H,W)
        
        return self.outc(x) # (B,n_classes,H,W)

