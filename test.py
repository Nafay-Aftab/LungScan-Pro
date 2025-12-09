import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import time
import os

# Custom Modules
from src.model import UNET
from src.dataset import LungDataset
from src.metrics import calculate_metrics

# --- CONFIG ---
# Must match train.py exactly to ensure the same split
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/my_checkpoint.pth.tar"
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

def run_clinical_audit():
    print("--- 🏥 STARTING FINAL CLINICAL AUDIT (HELD-OUT TEST SET) ---")
    
    # 1. REPLICATE THE SPLIT LOGIC
    # We must use the exact same seed (42) so the random split is identical to train.py
    # This ensures we get the *exact* 10% of data that train.py threw away.
    torch.manual_seed(42)
    
    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    full_dataset = LungDataset(IMG_DIR, MASK_DIR, transform=test_transform)
    
    # Recalculate sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split
    # We only care about 'test_ds' here
    _, _, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"🔒 Locked Test Set Size: {len(test_ds)} Patients")
    
    # 2. LOAD THE TRAINED BRAIN
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Train the model first!")
        return

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print("✅ Best Model Loaded. Running inference...")

    # 3. RUN AUDIT
    dice_scores = []
    iou_scores = []
    inference_times = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            
            # Timer Start
            start = time.time()
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Timer End
            end = time.time()
            inference_times.append((end - start) * 1000) # Convert to ms
            
            d, i, _, _ = calculate_metrics(preds, y)
            dice_scores.append(d)
            iou_scores.append(i)

    # 4. GENERATE REPORT
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_speed = np.mean(inference_times)
    
    print("\n" + "="*45)
    print("       🧬 FINAL PERFORMANCE REPORT")
    print("="*45)
    print(f"Model: U-Net (Fine-Tuned)")
    print(f"Data:  {len(test_ds)} Unseen Chest X-Rays")
    print("-" * 45)
    print(f"🟢 Dice Coefficient (F1 Score): {avg_dice*100:.2f}%")
    print(f"🔵 IoU (Jaccard Index):       {avg_iou*100:.2f}%")
    print(f"⚡ Avg Inference Latency:     {avg_speed:.2f} ms")
    print("-" * 45)
    
    if avg_dice > 0.90:
        print("🏆 STATUS: PASSED (Clinical Grade)")
    else:
        print("⚠️ STATUS: NEEDS OPTIMIZATION")
    print("="*45)

if __name__ == "__main__":
    run_clinical_audit()