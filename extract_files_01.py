import os
import gzip
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "data"

def extract_tar_gz(tar_path, output_dir):
    """Extract .tar.gz file containing welsh dataset"""
    print(f"\nExtracting {os.path.basename(tar_path)} :")
    print(f"Output: {output_dir}")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get list of members for progress bar
            members = tar.getmembers()
            # Extract with progress bar
            for member in tqdm(members, desc="Extracting", unit="files"):
                tar.extract(member, output_dir)
        
        print(f"Welsh dataset extraction completed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def extract_gz(gz_path, output_path):
    """Extract .gz file containing english dataset"""
    print(f"\nExtracting {os.path.basename(gz_path)} :")
    print(f"Output: {output_path}")
    
    try:
        # Get file size for progress bar
        file_size = os.path.getsize(gz_path)
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                # Read and write in chunks with progress bar
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    while True:
                        chunk = f_in.read(1024 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"English dataset extraction completed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Extracting files...")
    
    if not os.path.exists(DATA_DIR):
        print(f"\nError: '{DATA_DIR}' directory not found!")
        return
    
    # File 1: cv-corpus (Welsh data - tar.gz)
    welsh_tar = os.path.join(DATA_DIR, "cv-corpus-22.0-2025-06-20.gz")
    welsh_output = os.path.join(DATA_DIR, "welsh")
    
    # File 2: cvss (English data - tar.gz)
    english_tar = os.path.join(DATA_DIR, "cvss_c_cy_en_v1.0.tar.gz")
    english_output = os.path.join(DATA_DIR, "english")
    
    # Create output directories
    os.makedirs(welsh_output, exist_ok=True)
    os.makedirs(english_output, exist_ok=True)
    
    # Extract Welsh data
    if os.path.exists(welsh_tar):
        try:
            extract_tar_gz(welsh_tar, welsh_output)
        except:
            # If tar fails, try as plain gzip
            output_file = os.path.join(welsh_output, "cv-corpus-22.0-2025-06-20")
            extract_gz(welsh_tar, output_file)
    else:
        print(f"Error: {welsh_tar} not found")
    
    # Extract English data
    if os.path.exists(english_tar):
        extract_tar_gz(english_tar, english_output)
    else:
        print(f"Error: {english_tar} not found")
    
    print("\nExtraction complete")

if __name__ == "__main__":
    main()