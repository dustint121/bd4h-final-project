import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class MeshBlock(nn.Module):
    """Paper-accurate block with absolute subtraction fusion"""
    def __init__(self, in_channels, out_channels, module_type='both'):
        super().__init__()
        self.module_type = module_type
        
        # 3D branch
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        ) if module_type in ['3d', 'both'] else None
        
        # 2D branch
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        ) if module_type in ['2d', 'both'] else None
        
        # Paper's feature merging unit (FMU)
        self.fmu = nn.Sequential(
            nn.Conv3d(out_channels*2, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        ) if module_type == 'both' else None

    def forward(self, x):
        if self.module_type == 'both':
            out3d = self.conv3d(x)
            
            # Process 2D slices
            b, c, d, h, w = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w)
            out2d = self.conv2d(x2d)
            out2d = out2d.reshape(b, d, -1, h, w).permute(0, 2, 1, 3, 4)
            
            # Paper's fusion: abs(subtraction)
            merged = torch.abs(out3d - out2d)
            return self.fmu(torch.cat([out3d, merged], dim=1))
        
        elif self.module_type == '2d':
            b, c, d, h, w = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w)
            out2d = self.conv2d(x2d)
            return out2d.reshape(b, d, -1, h, w).permute(0, 2, 1, 3, 4)       
            
        elif self.module_type == '3d':
            return self.conv3d(x)
        else:  # 2D module
            return x  # 2D processing handled at network level

class FMU(nn.Module):
    """Feature Merging Unit from paper"""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch*2, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.LeakyReLU()
        )
        
    def forward(self, x2d, x3d):
        merged = torch.abs(x3d - x2d)  # Reverse subtraction order
        return self.conv(torch.cat([x3d, merged], dim=1))  # x3d first






class MNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super().__init__()
        self.grid_size = 5  # 5x5 mesh from paper
        self.depths = [1,2,3,4,5]
        
        # Create 5x5 mesh grid
        self.modules_grid = nn.ModuleList()
        for row in range(self.grid_size):
            row_modules = nn.ModuleList()
            for col in range(self.grid_size):
                # Determine module type per paper Figure 2a
                if col == 0:  # First column: 3D only
                    module_type = '3d'
                elif row == 0:  # First row: 2D only
                    module_type = '2d'
                else:  # Both branches
                    module_type = 'both'
                
                # Filter growth formula from paper Eq(1)
                current_depth = max(row+1, col+1)  # Convert 0-based indices to depth
                in_ch = 32 + 16*(current_depth-1)
                out_ch = 32 + 16*current_depth
                
                row_modules.append(MeshBlock(in_ch, out_ch, module_type))
            self.modules_grid.append(row_modules)
        
        # Deep supervision outputs (6 branches)
        self.output_branches = nn.ModuleList([
            nn.Conv3d(32 + 16*(i-1), n_classes, 1) for i in range(1,6)
        ] + [nn.Conv3d(112, n_classes, 1)])  # Final output
        
        # Skip connections with FMU
        self.fmus = nn.ModuleList([
            FMU(32 + 16*i) for i in self.depths  # i=1 → 48, i=2 → 64 etc.
        ])

    def forward(self, x):
        features = [[None]*self.grid_size for _ in range(self.grid_size)]
        outputs = []
        
        # Initialize first module (depth=1)
        features[0][0] = self.modules_grid[0][0](x)
        
        # Build mesh connections
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if row == 0 and col == 0:
                    continue
                
                current_depth = max(row+1, col+1)  # Paper's depth definition
                inputs = []
                
                # Get left feature (same row, previous column)
                if col > 0:
                    left = features[row][col-1]
                    inputs.append(left)
                    
                # Get top feature (previous row, same column)
                if row > 0:
                    top = features[row-1][col]
                    inputs.append(top)
                
                # Fuse features using depth-appropriate FMU
                if len(inputs) == 2:
                    fmu_idx = current_depth - 1  # Correct FMU indexing
                    merged = self.fmus[fmu_idx](inputs[0], inputs[1])
                elif len(inputs) == 1:
                    merged = inputs[0]
                
                # Process through current module
                features[row][col] = self.modules_grid[row][col](merged)
                
                # Add deep supervision outputs at network edges
                if row == self.grid_size-1 or col == self.grid_size-1:
                    branch_idx = current_depth - 1  # Correct output branch indexing
                    outputs.append(self.output_branches[branch_idx](features[row][col]))
        
        # Return main output + 5 auxiliary outputs
        return outputs[-1], outputs[:-1]