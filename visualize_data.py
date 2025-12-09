import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from src.dataset import LungDataset

# --- CONSTANTS ---
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

def visualize_batch():
    # 1. Define Basic Transform (Just Resize for now)
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Rotate(limit=35, p=1.0), # Heavy rotation to check if mask follows!
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # 2. Setup Dataset
    dataset = LungDataset(IMAGE_DIR, MASK_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Get a batch
    print(f"Checking for images in: {os.path.abspath(IMAGE_DIR)}")
    try:
        images, masks = next(iter(loader))
    except FileNotFoundError:
        print("ERROR: Could not find files. Check your folder structure!")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 4. Plot
    print(f"Batch Shape: Images {images.shape}, Masks {masks.shape}")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Image (Top Row)
        img = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title("Augmented X-Ray")
        axes[0, i].axis("off")
        
        # Mask (Bottom Row)
        msk = masks[i].numpy()
        axes[1, i].imshow(msk, cmap="gray")
        axes[1, i].set_title("Matching Mask")
        axes[1, i].axis("off")
        
    plt.tight_layout()
    plt.show()
    print("If the Masks (bottom) match the X-Rays (top) perfectly, we are good to go!")

if __name__ == "__main__":
    visualize_batch()