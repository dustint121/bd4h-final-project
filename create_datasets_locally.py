import nibabel as nib
import requests
import tempfile
import io
import os
import numpy as np
import time


def load_nii_from_url(url):
    """
    Load a NIfTI file from a URL into a NumPy array, with robust cleanup for Windows.
    """
    response = requests.get(url)
    response.raise_for_status()

    # Determine file extension from URL
    if url.endswith('.nii.gz'):
        suffix = '.nii.gz'
    elif url.endswith('.nii'):
        suffix = '.nii'
    else:
        raise ValueError("URL must end with .nii or .nii.gz")

    # Write to temporary file (auto-closes after block)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    try:
        # Load the NIfTI file
        nii = nib.load(tmp_path)
        return nii.get_fdata()
    finally:
        # Retry deletion to handle Windows file locking
        max_retries = 3
        for _ in range(max_retries):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.1)  # Wait 100ms before retrying
        else:
            print(f"⚠️ Failed to delete temporary file: {tmp_path}")


def multi_process_lits_dataset():
    pass


if __name__ == "__main__":
    
    os.makedirs("data/KITS19/", exist_ok=True)
    os.makedirs("data/LITS17/", exist_ok=True)

    max_val = 0

    file_average = 0


    # For LITS17 dataset: 131 samples [0-130]
    # for i in range(131):
    #     # print(f"LITS17 : {i}")
    #     volume_url = f"https://bd4h-final-project-data.s3.us-west-1.amazonaws.com/LITS17/volume/volume-{i}.nii"
    #     segmentation_url = f"https://bd4h-final-project-data.s3.us-west-1.amazonaws.com/LITS17/segmentations/segmentation-{i}.nii"        # print(segmentation_url, volume_url)
    #     # Load volume and segmentation from URLs and store into one file
    #     if not os.path.exists(f"data/LITS17/{i}.npz"):
    #         volume = load_nii_from_url(volume_url)  # Shape: (H, W, D)
    #         segmentation = load_nii_from_url(segmentation_url)  # Shape: (H, W, D)
    #         np.savez_compressed(f"data/LITS17/{i}",
    #                             volume=volume.astype(np.float16),
    #                             segmentation=segmentation.astype(np.uint8)
    #                             )
            
    # # Load
    # for i in range(131):
    #     data = np.load(f"data/LITS17/{i}.npz")
    #     volume = data["volume"]
    #     segmentation = data["segmentation"]
    #     print(f"LITS17 : {i}   : ",volume.shape, segmentation.shape) #Shape (H, W, D) [512, 512, D] D is varying
    #     if volume.shape != segmentation.shape:
    #         print(f"\tLITS17 : {i} has issues")
    #     print("\t", np.mean(volume), np.min(volume), np.max(volume))
    #     if np.max(volume) > max_val:
    #         max_val = np.max(volume)
        



    # # #for KITS19 dataset
    # for i in range(210):
    #     print(f"KITS19 : {i}")
    #     volume_url = f"https://bd4h-final-project-data.s3.us-west-1.amazonaws.com/KITS19/case_{i:05d}/imaging.nii.gz"
    #     segmentation_url = f"https://bd4h-final-project-data.s3.us-west-1.amazonaws.com/KITS19/case_{i:05d}/segmentation.nii.gz"
    #     # print(segmentation_url, volume_url)
    #     # Load volume and segmentation from URLs
    #     if not os.path.exists(f"data/KITS19/{i}.npz"):
    #         volume = load_nii_from_url(volume_url)  
    #         segmentation = load_nii_from_url(segmentation_url)  
    #         np.savez_compressed(f"data/KITS19/{i}",
    #                             volume=volume.astype(np.float16),
    #                             segmentation=segmentation.astype(np.uint8)
    #                             )





    # # Load
    # for i in range(210):
    #     data = np.load(f"data/KITS19/{i}.npz")  #need to convert: (num_slices, height, width) -> height, width, num_slices
    #     volume = np.transpose(data["volume"], (1, 2, 0))
    #     segmentation = np.transpose(data["segmentation"], (1, 2, 0))
    #     print(f"KITS17 : {i}   : ",volume.shape, segmentation.shape) 
    #     if volume.shape != segmentation.shape:
    #         print(f"\tKITS19 : {i} has issues")
    #     if np.max(volume) > max_val:
    #         max_val = np.max(volume)
    #     print("\t", np.mean(volume), np.min(volume), np.max(volume))
#8:47 - 8:54 for loading everything


#11:35 - 11:43


print("aaaaaaaaaaaaaaa")
print(max_val)