import streamlit as st
import cv2
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO, RTDETR
import kagglehub

# ==============================================================================
# PAGE CONFIGURATION & STYLES
# ==============================================================================
st.set_page_config(page_title="X-Ray Baggage Inspection App", layout="wide")
st.title("🧳 X-Ray Baggage Inspection - Multi-Model Evaluation")

# ==============================================================================
# PATHS & DATASET CONFIGURATION
# ==============================================================================
MODELS_DIR = "BEST_MODELS"  
try:
    base_kaggle_path = kagglehub.dataset_download("orvile/x-ray-baggage-anomaly-detection")
    TEST_IMAGES_DIR = os.path.join(base_kaggle_path, "test_processed", "images")
    TEST_LABELS_DIR = os.path.join(base_kaggle_path, "test_processed", "labels")
except Exception as e:
    st.error(f"Error connecting to Kagglehub: {e}")
    TEST_IMAGES_DIR = None
    TEST_LABELS_DIR = None

# CORRECT Class Mapping (Alphabetical order: 0 is Scissors)
CLASS_NAMES = {
    0: "Scissors",
    1: "Gun",
    2: "Knife",
    3: "Pliers",
    4: "Wrench"
}

# Box colors in BGR format for OpenCV
CLASS_COLORS = {
    0: (0, 255, 0),     # Green -> Scissors
    1: (0, 0, 255),     # Red -> Gun
    2: (0, 165, 255),   # Orange -> Knife
    3: (0, 255, 255),   # Yellow -> Pliers
    4: (255, 0, 255)    # Magenta -> Wrench
}

# ==============================================================================
# MODEL LOADING (CACHED)
# ==============================================================================
@st.cache_resource
def load_detection_model(path):
    """Dynamically loads RT-DETR or YOLO based on the file name pattern."""
    file_name = os.path.basename(path).lower()
    if "rtdetr" in file_name:
        return RTDETR(path)
    return YOLO(path)

@st.cache_resource
def load_clip_model():
    """Loads and caches OpenAI's CLIP model with global attention tracking enabled."""
    from transformers import CLIPConfig
    config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    config.vision_config.output_attentions = True
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=config)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# ==============================================================================
# PREPROCESSING PIPELINES
# ==============================================================================
def preprocess_pipeline_1(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    gaussian_blur = cv2.GaussianBlur(img_clahe, (9, 9), 10.0)
    img_sharp = cv2.addWeighted(img_clahe, 1.5, gaussian_blur, -0.5, 0)
    return img_sharp

def preprocess_pipeline_2(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    filtered = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)
    
    img_final = cv2.addWeighted(filtered, 0.9, gradient, 0.1, 0)
    return img_final

# ==============================================================================
# DRAWING & INFERENCE FUNCTIONS
# ==============================================================================
def draw_ground_truth(img_path, label_path):
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
                x_c, y_c, bbox_w, bbox_h = map(float, parts[1:])
                x1 = int((x_c - bbox_w / 2) * w)
                y1 = int((y_c - bbox_h / 2) * h)
                x2 = int((x_c + bbox_w / 2) * w)
                y2 = int((y_c + bbox_h / 2) * h)
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_predictions(processed_img, model_instance):
    img_canvas = processed_img.copy()
    results = model_instance.predict(img_canvas, verbose=False)[0]
    
    if results.boxes is not None:
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label_text = f"{CLASS_NAMES.get(cls_id, f'Class {cls_id}')} {conf:.2f}"
            
            cv2.rectangle(img_canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_canvas, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    return cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)

def analyze_with_clip(opencv_image, candidate_labels):
    model, processor = load_clip_model()
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    inputs = processor(text=candidate_labels, images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    results = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

def generate_clip_heatmap(opencv_image):
    model, processor = load_clip_model()
    h, w, _ = opencv_image.shape
    
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
    
    attentions = vision_outputs.attentions[-1] 
    avg_attention = attentions.mean(dim=1)[0]
    cls_attention = avg_attention[0, 1:] 
    
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    heatmap_grid = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    heatmap_normalized = (heatmap_grid - heatmap_grid.min()) / (heatmap_grid.max() - heatmap_grid.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap_normalized, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay_output = cv2.addWeighted(opencv_image, 0.6, color_heatmap, 0.4, 0)
    return cv2.cvtColor(overlay_output, cv2.COLOR_BGR2RGB)

# ==============================================================================
# GLOBAL NAVIGATION COMPONENT STATE (SYNCHRONIZED)
# ==============================================================================
if os.path.exists(TEST_IMAGES_DIR):
    available_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
else:
    available_images = []

# Persistent Initialization
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
if "current_heatmap" not in st.session_state:
    st.session_state.current_heatmap = None
if "clip_fixed_results" not in st.session_state:
    st.session_state.clip_fixed_results = None
if "clip_free_results" not in st.session_state:
    st.session_state.clip_free_results = None

def clear_cached_inferences():
    """Wipes computed assets whenever the active target image is altered."""
    st.session_state.current_heatmap = None
    st.session_state.clip_fixed_results = None
    st.session_state.clip_free_results = None

# Component Sync Actions
def handle_prev():
    if st.session_state.img_index > 0:
        st.session_state.img_index -= 1
        clear_cached_inferences()

def handle_next():
    if st.session_state.img_index < len(available_images) - 1:
        st.session_state.img_index += 1
        clear_cached_inferences()

def handle_random():
    if len(available_images) > 1:
        st.session_state.img_index = random.randint(0, len(available_images) - 1)
        clear_cached_inferences()

def handle_selectbox_change(key):
    """Callback linked to selectbox interactions."""
    selected_value = st.session_state[key]
    new_index = available_images.index(selected_value)
    if new_index != st.session_state.img_index:
        st.session_state.img_index = new_index
        clear_cached_inferences()

def render_navigation_block(key_suffix):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.button("⬅️ Previous", key=f"btn_prev_{key_suffix}", on_click=handle_prev, use_container_width=True)
    with c2:
        st.button("Next ➡️", key=f"btn_next_{key_suffix}", on_click=handle_next, use_container_width=True)
    with c3:
        st.button("🎲 Random Image", key=f"btn_rand_{key_suffix}", on_click=handle_random, use_container_width=True)
    
    # Ensure selectbox stays locked onto the uniform index state
    st.selectbox(
        "Select Image from Test Dataset:", 
        available_images, 
        index=st.session_state.img_index, 
        key=f"sb_select_{key_suffix}",
        on_change=handle_selectbox_change,
        args=(f"sb_select_{key_suffix}",)
    )
    st.caption(f"Showing Image {st.session_state.img_index + 1} of {len(available_images)}")

# ==============================================================================
# APP TABS SYSTEM (NAVIGATION)
# ==============================================================================
tab_yolo, tab_clip = st.tabs(["🔍 Object Detection (YOLO / RT-DETR)", "🔮 CLIP Laboratory (Zero-Shot)"])

# ==============================================================================
# TAB 1: OBJECT DETECTION
# ==============================================================================
with tab_yolo:
    st.header("Bounding Boxes Analysis")
    st.markdown("Evaluate trained models along with their optimal pre-processing pipeline criteria.")

    with st.sidebar:
        st.subheader("🤖 YOLO Configuration")
        if os.path.exists(MODELS_DIR) and available_images:
            available_models = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith('.pt')]
            if available_models:
                selected_model_name = st.selectbox("Select Model Architecture:", sorted(available_models), key="yolo_model_select")
                model_full_path = os.path.join(MODELS_DIR, selected_model_name)
                model = load_detection_model(model_full_path)
            else:
                st.error("No checkpoint files (.pt) found.")
                st.stop()
        
            if "clahe_gaussianblur" in selected_model_name.lower():
                pipeline_selected = "Pipeline 1"
                st.success("Automatically linked: **Pipeline 1**")
            else:
                pipeline_selected = "Pipeline 2"
                st.info("Automatically linked: **Pipeline 2**")
        st.markdown("---")

    st.subheader("📁 Image Selection for Detection")
    render_navigation_block("yolo")
    
    uploaded_file_yolo = st.file_uploader("Or upload an external image for YOLO inference:", type=["jpg", "jpeg", "png"], key="file_yolo")

    if uploaded_file_yolo is not None:
        file_bytes = np.asarray(bytearray(uploaded_file_yolo.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.info("ℹ️ External image loaded (Ground Truth layout bypassed).")
        
        processed = preprocess_pipeline_1(img_raw) if pipeline_selected == "Pipeline 1" else preprocess_pipeline_2(img_raw)
        pred_img = draw_predictions(processed, model)
        st.image(pred_img, use_container_width=True, caption="Model Preds Viewport")
    else:
        if available_images:
            path_img = os.path.join(TEST_IMAGES_DIR, available_images[st.session_state.img_index])
            path_lbl = os.path.join(TEST_LABELS_DIR, f"{os.path.splitext(available_images[st.session_state.img_index])[0]}.txt")
            
            img_raw = cv2.imread(path_img)
            gt_img = draw_ground_truth(path_img, path_lbl)
            processed = preprocess_pipeline_1(img_raw) if pipeline_selected == "Pipeline 1" else preprocess_pipeline_2(img_raw)
            pred_img = draw_predictions(processed, model)
            
            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown("#### Ground Truth Labels")
                st.image(gt_img, use_container_width=True)
            with col_pred:
                st.markdown("#### Model Predictions")
                st.image(pred_img, use_container_width=True)

# ==============================================================================
# TAB 2: INTERACTIVE CLIP LABORATORY WITH GLOBAL VISUAL ATTENTION HEATMAP
# ==============================================================================
with tab_clip:
    st.header("🔮 CLIP Laboratory (Zero-Shot Classification)")
    st.markdown("Test open-vocabulary natural language concepts to evaluate CLIP's vision-text mapping features.")

    origen_imagen = st.radio("Image Source for CLIP Input:", ["Use Image selected from Test Dataset", "Upload a new external file"], horizontal=True, key="origen_clip", on_change=clear_cached_inferences)

    opencv_img_clip = None
    if origen_imagen == "Upload a new external file":
        uploaded_file_clip = st.file_uploader("Upload an image file for CLIP:", type=["jpg", "jpeg", "png"], key="file_clip")
        if uploaded_file_clip is not None:
            file_bytes = np.asarray(bytearray(uploaded_file_clip.read()), dtype=np.uint8)
            opencv_img_clip = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        st.subheader("📁 Image Selection for CLIP")
        render_navigation_block("clip")
        if available_images:
            path_img_clip = os.path.join(TEST_IMAGES_DIR, available_images[st.session_state.img_index])
            opencv_img_clip = cv2.imread(path_img_clip)

    if opencv_img_clip is not None:
        col_visualizer, col_modules = st.columns([1, 1.2])
        
        with col_visualizer:
            st.markdown("#### Target Input View")
            st.image(cv2.cvtColor(opencv_img_clip, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🧠 CLIP Explainable AI (Transformer Attention Map)")
            
            if st.button("🗺️ Generate Attention Heatmap", use_container_width=True):
                with st.spinner("Extracting multi-head self-attention token weights..."):
                    st.session_state.current_heatmap = generate_clip_heatmap(opencv_img_clip)
            
            if st.session_state.current_heatmap is not None:
                st.image(st.session_state.current_heatmap, use_container_width=True, caption="Warm colors (Red/Yellow) highlight where CLIP is focusing.")
            else:
                st.caption("Click the button above to visually project CLIP's neural focus grid mapping.")
            
        with col_modules:
            modulo_seleccionado = st.radio(
                "Select CLIP Evaluation Modality:",
                ["📊 1. Automatic Dataset Class Identification", "💬 2. Custom Free-Form Prompting Query"]
            )
            st.markdown("---")
            
            # ------------------------------------------------------------------
            # SUBMODULE 1: FIXED DATASET CLASSES IDENTIFICATION
            # ------------------------------------------------------------------
            if modulo_seleccionado == "📊 1. Automatic Dataset Class Identification":
                st.subheader("Standard Multi-Class Assessment")
                st.write("CLIP will score the likelihood of core threats based on global contextual patterns.")
                
                prompts_fijos = [
                    "An X-ray image containing scissors",
                    "An X-ray image containing a gun",
                    "An X-ray image containing a knife",
                    "An X-ray image containing pliers",
                    "An X-ray image containing a wrench",
                    "A clean X-ray image with no threats"
                ]
                
                if st.button("🔍 Run Fixed Analysis", use_container_width=True):
                    with st.spinner("Processing embeddings matching..."):
                        st.session_state.clip_fixed_results = analyze_with_clip(opencv_img_clip, prompts_fijos)
                
                # PERSISTENT RENDERING
                if st.session_state.clip_fixed_results is not None:
                    for prompt, score in st.session_state.clip_fixed_results.items():
                        st.write(f"🔹 `{prompt}`")
                        st.progress(score)
                        st.caption(f"Confidence score: **{score * 100:.2f}%**")

            # ------------------------------------------------------------------
            # SUBMODULE 2: FREE-FORM PROMPTS QUERY
            # ------------------------------------------------------------------
            else:
                st.subheader("Free-Text Interaction Console")
                st.write("Ask open-ended queries or set custom hypothesis variables.")
                
                st.info(
                    "💡 **CLIP Operation Guide:** CLIP calculates probability by contrasting options against each other. "
                    "You must provide **at least 2 different target categories/answers** below (separated by commas) "
                    "for the model to run and distribute scores accurately."
                )
                
                pregunta_usuario = st.text_input(
                    "Context / Query prompt question:", 
                    value="What elements can you identify in this x-ray image of a bag?"
                )
                
                st.markdown("##### Target Prediction Answers to contrast:")
                
                opciones_por_defecto = "A sharp pair of scissors, A handgun weapon, A wrench tool, Normal clothing and items"
                user_options_input = st.text_area("Hypothesis Candidate Answers:", value=opciones_por_defecto, height=80)
                
                lista_opciones = [opt.strip() for opt in user_options_input.split(",") if opt.strip()]
                
                if st.button("🚀 Evaluate Free Prompt", use_container_width=True):
                    if len(lista_opciones) < 2:
                        st.warning("⚠️ Action blocked. Please enter at least 2 categories separated by commas.")
                    else:
                        with st.spinner("CLIP processing open semantics..."):
                            st.session_state.clip_free_results = analyze_with_clip(opencv_img_clip, lista_opciones)
                
                # PERSISTENT RENDERING
                if st.session_state.clip_free_results is not None:
                    st.markdown(f"**Evaluation results for query:** *{pregunta_usuario}*")
                    for opcion, score in st.session_state.clip_free_results.items():
                        st.write(f"👉 **{opcion}**")
                        st.progress(score)
                        st.caption(f"Match Probability: **{score * 100:.2f}%**")
    else:
        st.info("Configure dataset images or upload a target file to enable CLIP modules.")