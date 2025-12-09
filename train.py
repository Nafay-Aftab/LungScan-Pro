import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split

# Custom Modules
from src.model import UNET
from src.dataset import LungDataset

# --- HYPERPARAMETERS ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 0
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/images"
TRAIN_MASK_DIR = "data/masks"

# --- METRIC HELPER ---
def get_validation_metrics(loader, model, device="cuda"):
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    model.train()
    return dice_score / len(loader)

# --- TRAINING LOOP ---
def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        with torch.amp.autocast('cuda' if DEVICE == "cuda" else 'cpu'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

def main():
    torch.manual_seed(42)
    print(f"--- 🚀 Starting Clinical-Grade Training on {DEVICE} ---")

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    full_dataset = LungDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=transform,
    )

    # Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    print(f"📊 Split Strategy: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- FIX IS HERE ---
    # Removed 'verbose=True' to comply with newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    scaler = torch.amp.GradScaler('cuda' if DEVICE == "cuda" else 'cpu')

    best_dice = 0.0
    if not os.path.exists("saved_models"): os.makedirs("saved_models")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        
        val_dice = get_validation_metrics(val_loader, model, device=DEVICE)
        print(f"🧪 Validation Dice Score: {val_dice:.4f}")
        
        scheduler.step(val_dice)

        if val_dice > best_dice:
            print(f"💾 Improvement detected ({best_dice:.4f} -> {val_dice:.4f}). Saving model...")
            best_dice = val_dice
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, "saved_models/my_checkpoint.pth.tar")
        else:
            print(f"⏳ No improvement. (Current Best: {best_dice:.4f})")

    print(f"\n✅ Training Complete. Best Validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()