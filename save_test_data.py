import torch
from torch.utils.data import random_split
import os
import shutil
import albumentations as A

# Import your dataset logic
from src.dataset import LungDataset

# --- CONFIG ---
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
DEST_DIR = "data/test_samples" # New folder for test images

def save_test_files():
    print(f"--- 📂 ARCHIVING UNSEEN TEST DATA ---")
    
    # 1. Setup Dataset (Logic matches train.py)
    dataset = LungDataset(IMG_DIR, MASK_DIR, transform=None)
    
    # 2. Replicate the Seed (Crucial)
    generator = torch.Generator().manual_seed(42)
    
    # 3. Calculate Split Sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # 4. Perform the Split
    _, _, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # 5. Create Destination Folder
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created folder: {DEST_DIR}")
    else:
        print(f"Folder exists: {DEST_DIR} (files will be overwritten)")

    print(f"Copying {len(test_ds)} unseen images to '{DEST_DIR}'...")
    
    count = 0
    for idx in test_ds.indices:
        # Get filename from original dataset
        filename = dataset.images[idx]
        
        # Source Path
        src_path = os.path.join(IMG_DIR, filename)
        
        # Dest Path
        dst_path = os.path.join(DEST_DIR, filename)
        
        # Copy
        shutil.copy2(src_path, dst_path)
        count += 1

    print(f"✅ Success! Copied {count} images.")
    print(f"👉 Go to 'data/test_samples' to pick images for your Streamlit App.")

if __name__ == "__main__":
    save_test_files()