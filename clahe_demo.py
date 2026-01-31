"""
This script provides a CLAHE demonstration with a sample image, 
which is used in our paper.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_preprocessing(
    image_rgb,
    clip_limit = 2.0,
    tile_grid_size = (8, 8)
):
    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )

    out = image_rgb.copy()
    out[:, :, 1] = clahe.apply(out[:, :, 1])  # Green Channel Only
    return out


# Image Path
image_path = "IDRID/Original_Images/train/IDRiD_25.jpg"

image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

green_clahe = apply_preprocessing(image_rgb)


# Center Crop
h, w, _ = image_rgb.shape
min_dim = min(h, w)

start_x = (w - min_dim) // 2
start_y = (h - min_dim) // 2

image_rgb_sq = image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]
green_clahe_sq = green_clahe[start_y:start_y + min_dim, start_x:start_x + min_dim]


# Plot & Export
fig, axes = plt.subplots(1, 2, figsize = (8, 4))

for ax, img in zip(axes, [image_rgb_sq, green_clahe_sq]):
    ax.imshow(img)
    ax.axis("off")

plt.subplots_adjust(wspace = 0, hspace = 0)
plt.tight_layout(pad = 0)

plt.savefig(
    "clahe_green_comparison.png",
    bbox_inches = "tight",
    pad_inches = 0,
    dpi = 300
)

plt.close()
