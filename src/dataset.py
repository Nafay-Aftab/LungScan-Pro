import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # 1. Load all potential images
        all_images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 2. Filter: Only keep images that actually have a matching mask
        self.images = []
        missing_count = 0
        
        for img_name in all_images:
            mask_path = self._get_mask_path(img_name)
            
            if os.path.exists(mask_path):
                self.images.append(img_name)
            else:
                # Debugging: Print first few missing files to help you see the pattern
                if missing_count < 3:
                    print(f"⚠️ Warning: Excluding {img_name} (Mask not found at: {mask_path})")
                missing_count += 1
                
        print(f"✅ Dataset Loaded: {len(self.images)} valid pairs. (Excluded {missing_count} files without masks)")

    def _get_mask_path(self, img_filename):
        """Helper to find the mask path logic"""
        root, ext = os.path.splitext(img_filename)
        
        # Try naming convention 1: "image_mask.png"
        mask_name_v1 = f"{root}_mask{ext}"
        path_v1 = os.path.join(self.mask_dir, mask_name_v1)
        if os.path.exists(path_v1):
            return path_v1
            
        # Try naming convention 2: Exact match "image.png"
        path_v2 = os.path.join(self.mask_dir, img_filename)
        return path_v2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._get_mask_path(img_filename)

        # Load Files
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Normalize Mask (0.0 or 1.0)
        mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask