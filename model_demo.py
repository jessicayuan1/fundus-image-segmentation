"""
This script contains functionality to produce 4 binary masks from a trained model and a sample
as 4 images with black backgrounds for negative pixels, and white foreground for positive
pixels.
The produces images are used in our paper.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from HydraLANet_Definition.model.hydralanet import HydraLANet

# Configs
MODEL_PATH = "../../best_model.pt"
IMAGE_PATH = "model_training/utils/assets/TJDR_test_004.png"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD = 0.35

def apply_green_clahe(image_rgb, clip_limit = 2.0, tile_grid_size = (8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )
    out = image_rgb.copy()
    out[:, :, 1] = clahe.apply(out[:, :, 1])
    return out

# Load Existing Model
model = HydraLANet(
    snapshot = MODEL_PATH,
)

model.to(DEVICE)
model.eval()

# Load Image
image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_rgb = apply_green_clahe(image_rgb)

image = image_rgb.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype = np.float32)

image = (image - mean) / std

image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(image)

# Expected shape: (1, 4, H, W)
output = output.squeeze(0)

# Apply thresholding
binary_masks = (output > THRESHOLD).float()
masks = binary_masks.cpu().numpy()

# Save original image
plt.figure(figsize = (6, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig("original.png", dpi = 600, bbox_inches = "tight", pad_inches = 0)
plt.close()

# Save each mask separately
class_names = ["EX", "HE", "MA", "SE"]

for i, name in enumerate(class_names):
    mask = (masks[i] * 255).astype(np.uint8)

    plt.figure(figsize = (6, 6))
    plt.imshow(mask, cmap = "gray", vmin = 0, vmax = 255)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(f"mask_{name}.png", dpi = 600, bbox_inches = "tight", pad_inches = 0)
    plt.close()
