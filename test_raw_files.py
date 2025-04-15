import nibabel as nib
import requests
import tempfile
import io
import os
import numpy as np
import time

#NOTE: all segmentation files are ok in regard to being in range [0, 1, 2]
    #LITS17 has some [0, 1] only


#average volume values range: [-1024.0 - 3071.0]

'''
# 6-7 times more than valid range
KITS19 : 97
         -612.3834975747501 -6573.0 18558.0
'''
         

lits_17_list = []
kits_19_list = []

# For LITS17 dataset: 131 samples [0-130]
for i in range(131):
    print(f"LITS17 : {i}")
    volume = nib.load(f"original files/LITS17/volume/volume-{i}.nii").get_fdata()
    if np.max(volume) > 3100 or np.min(volume) < -1030:
        lits_17_list.append(i)
    # print("\t", np.mean(volume), np.min(volume), np.max(volume))
    # if int(np.max(volume)) > 3100:
    #     print("HU value max is too big")
    # segmentation = nib.load(f"original files/LITS17/segmentations/segmentation-{i}.nii").get_fdata()
    # print("\t", np.unique(segmentation))




# # #for KITS19 dataset
for i in range(210):
    print(f"KITS19 : {i}")
    volume = nib.load(f"original files/KITS19/case_{i:05d}/imaging.nii.gz").get_fdata()
    if np.max(volume) > 3100 or np.min(volume) < -1030:
        kits_19_list.append(i)
#     print("\t", np.mean(volume), np.min(volume), np.max(volume))
#     if int(np.max(volume)) > 3100:
#         print("HU value max is too big")

#     segmentation = nib.load(f"original files/KITS19/case_{i:05d}/segmentation.nii.gz").get_fdata()
#     print("\t", np.unique(segmentation))


print(lits_17_list)
print(kits_19_list)

"""
data with irregular data
lits_17_list = [0, 1, 4, 16, 17, 18, 22, 35, 37, 38, 48, 50, 52, 53, 54, 55, 57, 63, 65, 68, 69, 70, 71, 72, 74, 76, 77, 
                78, 80, 81, 82, 87, 88, 89, 90, 91, 92, 93, 95, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117]

kits_19_list = [15, 18, 19, 23, 25, 31, 32, 40, 43, 45, 48, 50, 61, 64, 65, 66, 81, 85, 86, 94, 97, 99, 107, 109, 111, 117,
                121, 123, 124, 128, 131, 133, 150, 163, 166, 167, 168, 169, 172, 180, 185, 191, 192, 193, 194, 199, 202]
"""