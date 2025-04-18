import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class MedicalVolumeDataset_3D(Dataset):
    def __init__(self, dataset_path, file_indices):
        self.volumes = []
        self.segmentations = []
        for i in file_indices:
            data = np.load(f"{dataset_path}/{i}.npz")
            volume = np.transpose(data["volume"], (2, 0, 1)).astype(np.float32)  # (D, H, W)
            seg = np.transpose(data["segmentation"], (2, 0, 1))
            # Z-score normalization using foreground voxels
            mask = seg > 0
            if mask.sum() > 0:
                mean = volume[mask].mean()
                std = volume[mask].std()
                volume = (volume - mean) / (std + 1e-8)
            self.volumes.append(volume[None, ...])  # Add channel dim
            self.segmentations.append(seg)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = torch.FloatTensor(self.volumes[idx])  # (1, D, H, W)
        seg = torch.LongTensor(self.segmentations[idx])  # (D, H, W)
        # Pad to multiples of 8 (for UNet compatibility)
        d, h, w = volume.shape[1:]
        pad_d = (8 - d % 8) % 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
        seg = F.pad(seg, (0, pad_w, 0, pad_h, 0, pad_d))
        return volume, seg




class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (for odd sizes)
        diffD = skip.shape[2] - x.shape[2]
        diffH = skip.shape[3] - x.shape[3]
        diffW = skip.shape[4] - x.shape[4]
        x = F.pad(x, [0, diffW, 0, diffH, 0, diffD])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class nnUNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, base_filters=32, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]
        # Encoder
        self.inc = ConvBlock3D(n_channels, filters[0])
        self.down1 = DownBlock3D(filters[0], filters[1])
        self.down2 = DownBlock3D(filters[1], filters[2])
        self.down3 = DownBlock3D(filters[2], filters[3])
        self.down4 = DownBlock3D(filters[3], filters[4])
        # Decoder
        self.up1 = UpBlock3D(filters[4], filters[3])
        self.up2 = UpBlock3D(filters[3], filters[2])
        self.up3 = UpBlock3D(filters[2], filters[1])
        self.up4 = UpBlock3D(filters[1], filters[0])
        # Output layers (deep supervision)
        self.outc = nn.Conv3d(filters[0], n_classes, 1)
        if self.deep_supervision:
            self.ds2 = nn.Conv3d(filters[1], n_classes, 1)
            self.ds3 = nn.Conv3d(filters[2], n_classes, 1)
            self.ds4 = nn.Conv3d(filters[3], n_classes, 1)

    def forward(self, x):
        print("here0")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        print("here00001")
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        out = self.outc(d1)
        print("here1")
        if self.deep_supervision and self.training:
            ds_out2 = self.ds2(d2)
            ds_out3 = self.ds3(d3)
            ds_out4 = self.ds4(d4)
            # Upsample to match out shape
            ds_out2 = F.interpolate(ds_out2, size=out.shape[2:], mode='trilinear', align_corners=False)
            ds_out3 = F.interpolate(ds_out3, size=out.shape[2:], mode='trilinear', align_corners=False)
            ds_out4 = F.interpolate(ds_out4, size=out.shape[2:], mode='trilinear', align_corners=False)
            print("here2")
            return [out, ds_out2, ds_out3, ds_out4]
        print("here3")
        return out
