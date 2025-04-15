import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


"""
MedicalSliceDataset_2D converts 3D arrays(volumes and segmentations) into lists of 2D arrays

In model training/testing:
    self.volumes = inputs 
    self.segmentations = targets
"""
class MedicalSliceDataset_2D(Dataset):
    def __init__(self, dataset_path, file_indices, max_slices=512):
        self.volumes = [] #will be a list of 2D arrays
        self.segmentations = [] #will be a list of 2D arrays
        
        
        for i in file_indices:
            data = np.load(f"{dataset_path}/{i}.npz")
            volume, seg = None, None
            if "LITS17" in dataset_path:
                volume = data["volume"].astype(np.int32)
                seg = data["segmentation"] #int8
  
            elif "KITS19" in dataset_path:
                volume = np.transpose(data["volume"], (1, 2, 0)).astype(np.int32)
                seg = np.transpose(data["segmentation"], (1, 2, 0))

            #Using z-score
            volume = (volume - volume.mean()) / (volume.std() + 1e-8) #float64
            print(volume.dtype, volume.max(), volume.min())
            # Add all slices as individual samples
            for d in range(volume.shape[2]): #for each layer of "depth"
                self.volumes.append(volume[..., d]) #add the "height x weight" 2D data of the layer to the self.volume
                self.segmentations.append(seg[..., d])

    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        # Remove channel dim from masks & convert to int64
        slice = torch.FloatTensor(self.volumes[idx][None, ...])  # (1,512,512)
        mask = torch.LongTensor(self.segmentations[idx])  # (512,512) NOT (1,512,512)
        return slice, mask




"""
DoubleConv:
Strenghts:
Deeper Feature Extraction: Two convolutional layers instead of one (like in ConvNet), 
                            enabling richer hierarchical feature learning.
Instance Normalization: Better for medical imaging with small batch sizes compared to batch norm.
Leaky ReLU: Avoids "dead neurons" by allowing small gradients for negative inputs (unlike basic ReLU in ConvNet).

Weaknesses:
Higher computational cost due to dual convolutions.
Requires more memory for intermediate feature maps.
"""
class DoubleConv(nn.Module):
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

class UNet2D(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)  # Output 3 channels

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        return self.outc(x) 