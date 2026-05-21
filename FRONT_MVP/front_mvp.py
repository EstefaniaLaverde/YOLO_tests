import streamlit as st
import cv2
import numpy as np
import os
import random
from ultralytics import YOLO, RTDETR
import matplotlib.pyplot as plt
import kagglehub

# configure streamlit page
st.set_page_config(page_title="X-Ray Baggage Anomaly Detection", layout="wide")
st.title("X-Ray Baggage Anomaly Detection - Visual Evaluation")
st.markdown("Select a model to evaluate its performance against ground truth annotations. The correct preprocessing pipeline is automatically selected based on the model.")

# define model and images path
MODELS_DIR = "BEST_MODELS"  # Folder containing trained model .pt files
try:
    # kagglehub automatically detects if the dataset is downloaded and returns the correct path
    base_kaggle_path = kagglehub.dataset_download("orvile/x-ray-baggage-anomaly-detection")
    
    # construct paths pointing directly to the kagglehub cache directory
    TEST_IMAGES_DIR = os.path.join(base_kaggle_path, "test_processed", "images")
    TEST_LABELS_DIR = os.path.join(base_kaggle_path, "test_processed", "labels")
    
except Exception as e:
    st.error(f"Error connecting to Kagglehub: {e}")
    TEST_IMAGES_DIR = None
    TEST_LABELS_DIR = None

# class names mapping
CLASS_NAMES = {0: "Scissors", 1: "Knife", 2: "Pliers", 3: "Wrench", 4: "Corkscrew"}

# bounding box colors per class
CLASS_COLORS = {
    0: (0, 0, 255), # red
    1: (255, 165, 0), # orange
    2: (255, 255, 0), # yellow
    3: (0, 255, 0), # green
    4: (255, 0, 255) # magenta
}

@st.cache_resource
def load_detection_model(path):
    """
    Dynamically loads either an RT-DETR or a YOLO model checkpoint based on the file name string pattern.
    """
    file_name = os.path.basename(path).lower()
    if "rtdetr" in file_name:
        return RTDETR(path)
    return YOLO(path)

# preprocessing pipelines 
def preprocess_pipeline_1(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # sharpening the image
    gaussian_blur = cv2.GaussianBlur(img_clahe, (9, 9), 10.0)
    img_sharp = cv2.addWeighted(img_clahe, 1.5, gaussian_blur, -0.5, 0)
    
    return img_sharp

def preprocess_pipeline_2(img):
    # use CLAHE to enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # apply a bilateral filter to smooth noise while preserving edges as much as possible
    filtered = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # highlight contours using morphological gradient - enhances objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

    # combine the filtered image with edges - 90% weight to clean image and 10% to pure contours
    img_final = cv2.addWeighted(filtered, 0.9, gradient, 0.1, 0)

    return img_final

# annotate ground truth boxes on the original image
def draw_ground_truth(img_path, label_path):
    """Reads the .txt file in YOLO format and draws the ground truth boxes on the image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                # convert YOLO normalized coordinates (x_center, y_center, width, height) to pixels
                x_c, y_c, bbox_w, bbox_h = map(float, parts[1:])
                x1 = int((x_c - bbox_w / 2) * w)
                y1 = int((y_c - bbox_h / 2) * h)
                x2 = int((x_c + bbox_w / 2) * w)
                y2 = int((y_c + bbox_h / 2) * h)
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_predictions(processed_img, model_instance):
    """Runs inference on the preprocessed image and draws the predicted boxes."""
    img_canvas = processed_img.copy()
    
    # inference (Ultralytics engines accept BGR arrays natively)
    results = model_instance.predict(img_canvas, verbose=False)[0]
    
    # draw predicted boxes manually to maintain visual consistency with ground truth
    if results.boxes is not None:
        for box in results.boxes:
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords)
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label_text = f"{CLASS_NAMES.get(cls_id, f'Class {cls_id}')} {conf:.2f}"
            
            cv2.rectangle(img_canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_canvas, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)

# Streamlit sidebar for controls and selections
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # model selection
    st.subheader("🤖 Model Weights")
    if os.path.exists(MODELS_DIR):
        available_models = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith('.pt')]
        if available_models:
            selected_model_name = st.selectbox(
                "Choose the architecture model to evaluate:", 
                sorted(available_models)
            )
            model_full_path = os.path.join(MODELS_DIR, selected_model_name)
            model = load_detection_model(model_full_path)
        else:
            st.error(f"No .pt files found inside the directory '{MODELS_DIR}'")
            st.stop()
    else:
        st.error(f"The directory '{MODELS_DIR}' does not exist in the root folder.")
        st.stop()
        
    st.markdown("---")

    # AUTOMATIC PIPELINE SELECTION BASED ON MODEL FILENAME
    st.subheader("🛠️ Preprocessing Pipeline")
    if "clahe_gaussianblur" in selected_model_name.lower():
        pipeline_selected = "Pipeline 1"
        st.success("⚙️ Automatically selected: **Pipeline 1** (CLAHE + Sharpening) to match model training criteria.")
    else:
        pipeline_selected = "Pipeline 2"
        st.info("⚙️ Automatically selected: **Pipeline 2** (Advanced X-Ray Edge Enhancement) to match model training criteria.")
    
    st.markdown("---")
    
    # test image selection 
    st.subheader("📁 Image Selection")
    img_src_path = None
    label_src_path = None
    
    if os.path.exists(TEST_IMAGES_DIR):
        available_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if available_images:
            # 1. Initialize the image index in session state if it does not exist
            if "img_index" not in st.session_state:
                st.session_state.img_index = 0

            # 2. Callbacks to handle state updates safely before rendering
            def go_back():
                if st.session_state.img_index > 0:
                    st.session_state.img_index -= 1

            def go_forward():
                if st.session_state.img_index < len(available_images) - 1:
                    st.session_state.img_index += 1

            def pick_random():
                if len(available_images) > 1:
                    st.session_state.img_index = random.randint(0, len(available_images) - 1)

            # Callback triggered when the user manually switches the selectbox option
            def on_selector_change():
                if "img_selector" in st.session_state:
                    st.session_state.img_index = available_images.index(st.session_state.img_selector)

            # 3. Render side-by-side navigation and random selection buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button("⬅️ Previous", on_click=go_back, use_container_width=True)
            with btn_col2:
                st.button("Next ➡️", on_click=go_forward, use_container_width=True)
                
            # Full-width Random button underneath the step counters
            st.button("🎲 Random Image", on_click=pick_random, use_container_width=True)

            # 4. Render the selectbox synchronized with the current state index
            selected_img_name = st.selectbox(
                "Select an image from the test set:", 
                available_images,
                index=st.session_state.img_index,
                key="img_selector",
                on_change=on_selector_change  # Synchronizes index if the dropdown is used
            )
            
            selected_img_name = available_images[st.session_state.img_index]
            
            img_src_path = os.path.join(TEST_IMAGES_DIR, selected_img_name)
            base_name = os.path.splitext(selected_img_name)[0]
            label_src_path = os.path.join(TEST_LABELS_DIR, f"{base_name}.txt")
            
            # Display current dataset progression counter
            st.caption(f"Image {st.session_state.img_index + 1} of {len(available_images)}")
        else:
            st.warning("No images found in the test directory.")
    else:
        st.error(f"Could not access the directory: {TEST_IMAGES_DIR}")

    # Allow uploading an external image if desired
    uploaded_file = st.file_uploader("Or upload an external image (.jpg, .png):", type=["jpg", "jpeg", "png"])

# ==============================================================================
# SIDE-BY-SIDE VISUAL PROCESSING AND DYNAMIC LAYOUT
# ==============================================================================
if uploaded_file is not None:
    # Process manually uploaded image (no Ground Truth available)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.info("ℹ️ Loading external image. Ground Truth annotations are not available.")
    
    # Dynamic pipeline application routing based on verified variable
    if pipeline_selected == "Pipeline 1":
        processed_img = preprocess_pipeline_1(opencv_img)
    else:
        processed_img = preprocess_pipeline_2(opencv_img)
        
    pred_display = draw_predictions(processed_img, model)

    # DYNAMIC LAYOUT: Show only the predictions canvas full-width when an image is uploaded
    st.subheader("Predictions")
    st.image(pred_display, use_container_width=True, caption=f"Inference using model: {selected_model_name} with {pipeline_selected}")

elif img_src_path is not None:
    # Process selected image from the test folder with its corresponding real Ground Truth
    opencv_img = cv2.imread(img_src_path)
    gt_display = draw_ground_truth(img_src_path, label_src_path)
    
    # Dynamic pipeline application routing based on verified variable
    if pipeline_selected == "Pipeline 1":
        processed_img = preprocess_pipeline_1(opencv_img)
    else:
        processed_img = preprocess_pipeline_2(opencv_img)
        
    pred_display = draw_predictions(processed_img, model)

    # DYNAMIC LAYOUT: Fallback to side-by-side columns view for local test directory navigation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ground Truth")
        st.image(gt_display, use_container_width=True, caption="Original annotations recorded in the dataset")

    with col2:
        st.subheader("Predictions")
        st.image(pred_display, use_container_width=True, caption=f"Inference using model: {selected_model_name} with {pipeline_selected}")
else:
    st.info("Please configure the dataset paths or upload an image to begin.")
    st.stop()