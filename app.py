import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from HydraLANet_Definition.model.hydralanet import HydraLANet
CLASS_NAMES = ["EX (Hard Exudates)", "HE (Hemorrhages)", "MA (Microaneurysms)", "SE (Soft Exudates)"]
CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# def apply_green_clahe(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
def apply_green_clahe(image_rgb, clip_limit=1.5, tile_grid_size=(8, 8)):
    """Apply CLAHE to green channel only for contrast enhancement"""
    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )
    out = image_rgb.copy()
    out[:, :, 1] = clahe.apply(image_rgb[:, :, 1])
    return out

def preprocess_image(image_rgb):
    """Preprocess image with CLAHE and ImageNet normalization"""
    image_rgb = apply_green_clahe(image_rgb)

    image = image_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image_tensor, image_rgb

@st.cache_resource
def load_model():
    """Load HydraLANet model with trained weights"""
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else \
             "cpu"

    weights_path = project_root / "runs" / "4B" / "best_model.pt"

    if weights_path.exists():
        model = HydraLANet(snapshot=str(weights_path))
        st.success(f"✅ Loaded trained model")
    else:
        st.error(f"❌ Model weights not found at {weights_path}. Please ensure weights are placed in runs/4B/best_model.pt")
        st.stop()

    model.to(device)
    model.eval()

    return model, device

def create_overlay(image_rgb, masks, threshold=0.35, alpha=0.4):
    """Create visualization with colored masks overlaid on original image"""
    overlay = image_rgb.copy()

    for i, (mask, color) in enumerate(zip(masks, CLASS_COLORS)):
        binary_mask = (mask > threshold).astype(np.uint8)
        colored_mask = np.zeros_like(image_rgb)
        colored_mask[binary_mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

    return overlay


st.set_page_config(
    page_title="HydraLA-Net Fundus Segmentation",
    page_icon="👁️",
    layout="wide"
)
st.title("👁️ HydraLA-Net Fundus Image Segmentation")
st.markdown("""
This application performs semantic segmentation of diabetic retinopathy lesions in fundus images using **HydraLA-Net**.
""")

# Device info and settings
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"🖥️ Using device: **{device.upper()}**")

threshold = st.slider("Detection Threshold", 0.01, 1.0, 0.35, 0.05)

uploaded_file = st.file_uploader(
    "Choose a fundus image",
    type=["png", "jpg", "jpeg"],
    help="Upload a retinal fundus image for segmentation"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image.convert('RGB'))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Original Image")
        st.image(image_rgb, caption="Original Image", width='stretch')

    with col2:
        st.subheader("📊 Segmentation Results")

    if st.button("🔍 Segment Image", type="primary"):
        with st.spinner("Loading model and processing image..."):
            model, device = load_model()
            image_tensor, processed_rgb = preprocess_image(image_rgb)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                output = model(image_tensor)

            output = output.squeeze(0).cpu().numpy()

        with col2:
            overlay = create_overlay(image_rgb, output, threshold)
            st.image(overlay, caption="All Lesions Overlay", width='stretch')

        st.subheader("🎯 Individual Lesion Masks")
        mask_cols = st.columns(4)

        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            with mask_cols[i]:
                mask = output[i]
                binary_mask = (mask > threshold).astype(np.uint8)

                colored_mask = np.zeros_like(image_rgb)
                colored_mask[binary_mask == 1] = color
                mask_overlay = cv2.addWeighted(image_rgb, 0.6, colored_mask, 0.4, 0)

                # st.image(mask_overlay, caption=name, use_container_width=True)
                st.image(mask_overlay, caption=name, width='stretch')

                positive_pixels = binary_mask.sum()
                total_pixels = binary_mask.size
                percentage = (positive_pixels / total_pixels) * 100
                st.metric("Coverage", f"{percentage:.2f}%")
else:
    st.info("👆 Please upload a fundus image to begin segmentation")

st.markdown("---")
st.markdown("""
### About HydraLA-Net
HydraLA-Net is an adapted version of LANet for diabetic retinopathy segmentation, featuring:
- **ResNet-50 Backbone** for robust feature extraction
- **Feature Fusion Blocks (FFB)** for multi-scale feature integration  
- **Lesion-Aware Modules (LAM)** for enhanced lesion detection
- **Hydra Segmentation Head** with 4 independent class-specific branches

**Lesion Types:**
- **EX (Hard Exudates)**: Yellow lipid deposits - shown in Red
- **HE (Hemorrhages)**: Blood vessel leakage - shown in Green  
- **MA (Microaneurysms)**: Tiny vessel bulges - shown in Blue
- **SE (Soft Exudates)**: Cotton-wool spots - shown in Yellow

For more information, see the [README](https://github.com/jessicayuan1/fundus-image-segmentation).
""")