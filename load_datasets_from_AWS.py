import requests
import tempfile
import numpy as np
import os
import time

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_npz_from_url(base_url, save_dir, num_cases, dataset_name, max_workers=8, max_cases=None):
    """
    Download NPZ files from sequential URLs and save locally.
    
    Args:
        base_url (str): Base URL pattern containing {dataset} and {case} placeholders
        save_dir (str): Local directory to save files
        num_cases (int): Total number of cases to download
        dataset_name (str): Name of dataset (KITS19 or LITS17)
        max_workers (int): Number of parallel threads
        max_cases (int): Optional limit for testing
    """
    os.makedirs(save_dir, exist_ok=True)
    
    def download_case(case):
        try:
            url = base_url.format(dataset=dataset_name, case=case)
            local_path = os.path.join(save_dir, f"{case}.npz")
            
            # if os.path.exists(local_path):
            #     return f"Skipped {case} (exists)"
                
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(response.content)
                
            return f"Downloaded {case}"
            
        except Exception as e:
            return f"Error {case}: {str(e)}"

    # Create case numbers with optional limit
    cases = range(num_cases)
    if max_cases:
        cases = cases[:max_cases]

    # Use parallel processing with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(download_case, cases), total=len(cases)))
    
    # Print summary
    for result in results:
        print(result)




if __name__ == "__main__":
#9:07 -

    numpy_data_folder = "data"

    base_url = "https://bd4h-final-project-data.s3.us-west-1.amazonaws.com/numpy_files/{dataset}/{case}.npz"
    
    # KITS19 (210 cases)
    download_npz_from_url(
        base_url=base_url,
        save_dir=f"{numpy_data_folder}/KITS19/",
        num_cases=210,
        dataset_name="KITS19",
        max_workers=8,
        max_cases=None  # Set to small number (e.g., 5) for testing
    )

    #9:28 - 11:10 (2 hours for loading everything)
    
    # LITS17 (131 cases)
    download_npz_from_url(
        base_url=base_url,
        save_dir=f"{numpy_data_folder}/LITS17/",
        num_cases=131,
        dataset_name="LITS17",
        max_workers=8,
        max_cases=None
    )