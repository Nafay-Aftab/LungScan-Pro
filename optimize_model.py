import torch
import os
from src.model import UNET

INPUT_PATH = "saved_models/my_checkpoint.pth.tar"
OUTPUT_PATH = "saved_models/lung_model_deploy.pth"

def optimize():
    if not os.path.exists(INPUT_PATH):
        print("❌ Error: Checkpoint not found.")
        return
        
    print("📉 Optimizing model...")
    checkpoint = torch.load(INPUT_PATH, map_location="cpu")
    
    model = UNET(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Convert to Half Precision (FP16) to cut size by 50%
    model.half()
    
    # Save weights only (Strip optimizer data)
    torch.save(model.state_dict(), OUTPUT_PATH)
    
    print(f"✅ Done! Saved to: {OUTPUT_PATH}")
    print(f"New Size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    optimize()