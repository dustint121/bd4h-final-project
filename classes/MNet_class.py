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
            volume = np.transpose(data["volume"], (2, 0, 1)).astype(np.int32)
            seg = np.transpose(data["segmentation"], (2, 0, 1))
            self.volumes.append(volume[None, ...])  # Add channel dim
            self.segmentations.append(seg)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = torch.FloatTensor(self.volumes[idx])
        seg = torch.LongTensor(self.segmentations[idx])
        d, h, w = volume.shape[1:]
        pad_d = (8 - (d % 8)) % 8
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        volume_padded = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d, 0, 0))
        seg_padded = F.pad(seg, (0, pad_w, 0, pad_h, 0, pad_d))
        return volume_padded, seg_padded





class MeshBlock(nn.Module):
    """
    MNet basic block: parallel 2D and 3D convolutions, then feature fusion.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3D conv branch
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        # 2D conv branch (applied slice-wise along depth)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        # Fusion
        self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        out3d = self.conv3d(x)
        # Apply 2D conv to each slice along D
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        out2d = self.conv2d(x2d)
        out2d = out2d.reshape(b, d, -1, h, w).permute(0, 2, 1, 3, 4)
        # Concatenate along channel
        out = torch.cat([out3d, out2d], dim=1)
        out = self.fuse(out)
        return out
    


class MNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super().__init__()
        # Encoder
        self.enc1 = MeshBlock(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), MeshBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), MeshBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), MeshBlock(128, 256))
        # Decoder
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = MeshBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = MeshBlock(128, 64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = MeshBlock(64, 32)
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)
        return self.outc(x)