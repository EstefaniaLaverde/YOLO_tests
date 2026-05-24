

import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import open_clip
import random
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from ultralytics import YOLO, RTDETR
import kagglehub
import torchvision.models as models
import torchvision.transforms as T

# config page settings and custom CSS for a dark-themed security console aesthetic
st.set_page_config(
    page_title="X-Ray Security Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# THIS WAS GENERATED WITH CLAUDE SONNET - I DONT KNOW CSS XD
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

/* ── Root tokens ──────────────────────────────────────────────────── */
:root {
    --bg:        #0a0c0f;
    --bg-panel:  #111418;
    --bg-card:   #161b22;
    --border:    #1f2a35;
    --accent:    #00ff88;
    --accent-dim:#00aa55;
    --warn:      #ffb800;
    --danger:    #ff3b5c;
    --text:      #c8d6e5;
    --text-dim:  #5a7080;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow Condensed', sans-serif;
}

/* ── Global reset ─────────────────────────────────────────────────── */
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
* { font-family: var(--sans) !important; letter-spacing: 0.02em; font-size: 16px; }

/* ── Hide Streamlit chrome ────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarToggleButton"] { 
    visibility: visible !important; 
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 12px 28px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg) !important;
    padding-top: 20px;
}

/* ── Cards / panels ───────────────────────────────────────────────── */
.scan-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.scan-card-accent { border-left: 3px solid var(--accent); }
.scan-card-warn   { border-left: 3px solid var(--warn); }

/* ── Section headers ──────────────────────────────────────────────── */
.section-header {
    font-family: var(--sans) !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── App title ────────────────────────────────────────────────────── */
.app-title {
    font-family: var(--mono) !important;
    font-size: 22px;
    color: var(--accent);
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.app-subtitle {
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.18em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 0;
}

/* ── Pipeline badge ───────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: var(--mono) !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
}
.badge-p1  { background: #0d2035; color: #4db8ff; border: 1px solid #1a4a70; }
.badge-p2  { background: #0d2010; color: var(--accent); border: 1px solid #1a5030; }
.badge-none{ background: #201810; color: var(--warn); border: 1px solid #503010; }

/* ── Metric chips ─────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.metric-chip {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 8px 14px;
    text-align: center;
    min-width: 90px;
}
.metric-chip .val {
    font-family: var(--mono) !important;
    font-size: 20px;
    color: var(--accent);
    display: block;
}
.metric-chip .lbl {
    font-size: 10px;
    letter-spacing: 0.14em;
    color: var(--text-dim);
    text-transform: uppercase;
}

/* ── Image frame ──────────────────────────────────────────────────── */
.img-frame-label {
    font-size: 13px;
    letter-spacing: 0.18em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 6px;
    font-weight: 700;
}

/* ── Probability bar ──────────────────────────────────────────────── */
.prob-row { margin: 8px 0; }
.prob-label {
    font-family: var(--mono) !important;
    font-size: 15px;
    color: var(--text);
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.prob-bar-bg {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent-dim), var(--accent));
    transition: width 0.4s ease;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 3px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Selectbox / inputs ───────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 15px !important;
}

/* ── File uploader ────────────────────────────────────────────────── */
.stFileUploader > div {
    background: var(--bg-panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* ── Caption / info ───────────────────────────────────────────────── */
.stCaption { color: var(--text-dim) !important; font-size: 13px !important; }
.stInfo    { background: #0d1a2e !important; border: 1px solid #1a3a60 !important; color: #80b4d4 !important; border-radius: 3px !important; }
.stSuccess { background: #0d2018 !important; border: 1px solid #1a5030 !important; color: var(--accent) !important; border-radius: 3px !important; }
.stWarning { background: #201a08 !important; border: 1px solid #50400a !important; color: var(--warn) !important; border-radius: 3px !important; }

/* ── Radio ────────────────────────────────────────────────────────── */
.stRadio label { font-size: 15px !important; color: var(--text) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text-dim) !important; font-size: 11px; }

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

#  set some constants for model loading, class names, and colors
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "BEST_MODELS")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
NUM_CLASSES = 5
CLASS_NAMES = {0: "Scissors", 1: "Gun", 2: "Knife", 3: "Pliers", 4: "Wrench"}
CLASS_COLORS = {
    0: (0, 255, 0), # green – Scissors
    1: (0, 0, 255), # red – Gun
    2: (0, 165, 255), # orange – Knife
    3: (0, 255, 255), # yellow – Pliers
    4: (255, 0, 255), # magenta – Wrench
}

# map model filename to its corresponding preprocessing pipeline - inference images must be processed with the same pipeline used during training for best results.
def infer_pipeline(model_filename: str) -> str:
    """Return 'P1', 'P2', or 'None' based on naming convention."""
    name = model_filename.lower()
    if any(k in name for k in ["p1", "pipeline1", "clahe_gauss", "unsharp"]):
        return "P1"
    if any(k in name for k in ["p2", "pipeline2", "bilateral", "morph"]):
        return "P2"
    # baseline .pth files always use Pipeline 1
    if name.endswith(".pth"):
        return "P1"
    return "P2"   # safe default for unlabeled pretrained .pt

# prettier badge display for the UI :)
PIPELINE_BADGE = {
    "P1": '<span class="badge badge-p1">Pipeline 1 — Unsharp Mask</span>',
    "P2": '<span class="badge badge-p2">Pipeline 2 — Bilateral + Gradient</span>',
    "None": '<span class="badge badge-none">No preprocessing</span>',
}

# load dataset paths to cache to avoid repeated downloads
@st.cache_resource
def load_dataset_paths():
    try:
        base = kagglehub.dataset_download("orvile/x-ray-baggage-anomaly-detection")
        return (
            os.path.join(base, "test", "images"),
            os.path.join(base, "test", "labels"),
        )
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None

TEST_IMG_DIR, TEST_LBL_DIR = load_dataset_paths()

# load and cache the list of test images
@st.cache_data
def get_image_list():
    if TEST_IMG_DIR and os.path.exists(TEST_IMG_DIR):
        return sorted([
            f for f in os.listdir(TEST_IMG_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    return []

IMAGES = get_image_list()

# preprocessing pipeline used in training
def preprocess_p1(img: np.ndarray) -> np.ndarray:
    """CLAHE + Unsharp Mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img_clahe, (9, 9), 10.0)
    return cv2.addWeighted(img_clahe, 1.5, blur, -0.5, 0)

def preprocess_p2(img: np.ndarray) -> np.ndarray:
    """CLAHE + Bilateral Filter + Morphological Gradient."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    filtered  = cv2.bilateralFilter(img_clahe, 9, 75, 75)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient  = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)
    return cv2.addWeighted(filtered, 0.9, gradient, 0.1, 0)

def apply_pipeline(img: np.ndarray, pipeline: str) -> np.ndarray:
    if pipeline == "P1":
        return preprocess_p1(img)
    if pipeline == "P2":
        return preprocess_p2(img)
    return img

# load and cache models to avoid repeated disk I/O and GPU warmup
@st.cache_resource
def load_pretrained_model(path: str):
    name = os.path.basename(path).lower()
    return RTDETR(path) if "rtdetr" in name else YOLO(path)

# the baseline model is a custom ResNet. We need to reconstruct the architecture in code to load the .pth checkpoint properly.
@st.cache_resource
def load_baseline_model(path: str):
    """Load the custom ResNet-18 baseline saved as .pth.

    Architecture reconstructed from the actual checkpoint keys:
      backbone - ResNet-18 conv layers (Sequential)
      spatial_reduction - Sequential(Conv2d, BN, ReLU, AdaptiveAvgPool)
      classifier - Sequential(Flatten, Linear, ReLU, Linear)
      box_regressor - Sequential(Flatten, Linear, ReLU, Linear, Sigmoid)
    """
    import torch.nn as nn

    class BaselineDetector(nn.Module):
        def __init__(self, nc=6):
            super().__init__()
            # set backbone 
            base = models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])

            # spacial reduction of feature maps to a fixed size for the classifier and regressor heads
            self.spatial_reduction = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )

            feat_dim = 128 * 4 * 4

            # classification head - outputs class logits (including background class 0)
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, nc),
            )

            # regression head - outputs normalized box coordinates (xc, yc, bw, bh) in [0, 1] relative to input image size
            self.box_regressor = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 4),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # apply reduction to the backbone features before feeding into the heads
            f = self.spatial_reduction(self.backbone(x))

            # flatten the feature maps to a vector for the fully connected layers
            f = f.flatten(1)

            # return both class logits and box coordinates
            return self.classifier(f), self.box_regressor(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = BaselineDetector(nc=NUM_CLASSES + 1)
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    state  = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

# load CLIP for zero-shot classification of detected objects
@st.cache_resource
def load_clip_model():
    config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    config.vision_config.output_attentions = True
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=config)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def load_openclip_retrieval_model():
    """Load the same OpenCLIP model used to create the stored crop embeddings."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED,
        device=device,
    )
    model.eval()
    return model, preprocess, device

@st.cache_data
def load_retrieval_assets():
    """Load normalized embeddings and row-aligned metadata generated offline."""
    emb_path = os.path.join(OUTPUTS_DIR, "embeddings.npy")
    meta_path = os.path.join(OUTPUTS_DIR, "embeddings_metadata.csv")
    embeddings = np.load(emb_path).astype("float32")
    metadata = pd.read_csv(meta_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Embeddings and metadata have different row counts.")
    return embeddings, metadata

# some inference helper functions to run the models and draw boxes on the images
def predict_pretrained(model, img_bgr: np.ndarray):
    """Run YOLO / RT-DETR on a preprocessed BGR image. Returns annotated RGB."""
    canvas  = img_bgr.copy()
    results = model.predict(canvas, verbose=False)[0]
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label= f"{CLASS_NAMES.get(cls_id, str(cls_id))}  {conf:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def extract_pretrained_regions(model, detector_img_bgr: np.ndarray, crop_img_bgr: np.ndarray, min_conf: float):
    """Run the pretrained detector and return its detected regions as BGR crops."""
    canvas  = detector_img_bgr.copy()
    results = model.predict(canvas, verbose=False)[0]
    regions = []

    if results.boxes is not None:
        h, w = crop_img_bgr.shape[:2]
        for box in results.boxes:
            conf = float(box.conf[0].item())
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(box.cls[0].item())
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))}  {conf:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            regions.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "crop_bgr": crop_img_bgr[y1:y2, x1:x2].copy(),
            })

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), regions

def encode_openclip_crop(crop_bgr: np.ndarray) -> np.ndarray:
    model, preprocess, device = load_openclip_retrieval_model()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    image = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype("float32")

def retrieve_similar_crops(query_embedding: np.ndarray, top_k: int = 3):
    embeddings, metadata = load_retrieval_assets()
    scores = embeddings @ query_embedding
    top_idx = np.argsort(-scores)[:top_k]
    rows = metadata.iloc[top_idx].copy()
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    rows["score"] = scores[top_idx]
    return rows

def resolve_metadata_crop_path(row) -> str:
    crop_path = str(row.get("crop_path", "")).strip()
    crop_path = crop_path.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(crop_path):
        return os.path.normpath(crop_path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, crop_path))

def resolve_metadata_crop(row) -> np.ndarray | None:
    """Return an RGB crop from the crop_path stored in metadata."""
    crop_abs = resolve_metadata_crop_path(row)
    if os.path.exists(crop_abs):
        img = cv2.imread(crop_abs)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    return None

# predict with the custom baseline model - since it outputs a single box and class per image, we just draw that one box if the predicted class is not background (0)
def predict_baseline(model, img_bgr: np.ndarray):
    """Run the custom baseline model. Returns annotated RGB."""
    device = next(model.parameters()).device
    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp   = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, boxes = model(inp)

    cls_id = int(logits.argmax(dim=1).item())
    conf   = float(F.softmax(logits, dim=1).max().item())
    canvas = cv2.resize(img_bgr, (416, 416)).copy()
    h, w   = canvas.shape[:2]

    # class 0 = background; skip drawing if background predicted
    if cls_id > 0:
        xc, yc, bw, bh = boxes[0].tolist()
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = CLASS_COLORS.get(cls_id - 1, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id - 1, str(cls_id))}  {conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), cls_id, conf

def draw_ground_truth(img_path: str, lbl_path: str) -> np.ndarray:
    """Draw GT boxes on image. Returns RGB."""
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    if os.path.exists(lbl_path):
        for line in open(lbl_path).readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASS_NAMES.get(cls_id, str(cls_id)),
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clip_classify(opencv_img: np.ndarray, labels: list[str]) -> dict:
    model, processor = load_clip_model()
    pil = Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    inputs = processor(text=labels, images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    return dict(sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True))

def clip_heatmap(opencv_img: np.ndarray) -> np.ndarray:
    model, processor = load_clip_model()
    h, w, _ = opencv_img.shape
    pil = Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        vis_out = model.vision_model(**inputs)
    attn = vis_out.attentions[-1].mean(dim=1)[0]
    cls_attn = attn[0, 1:]
    grid = int(np.sqrt(cls_attn.shape[0]))
    hmap = cls_attn.reshape(grid, grid).cpu().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    hmap = np.uint8(255 * cv2.resize(hmap, (w, h)))
    overlay = cv2.addWeighted(opencv_img, 0.55,
                              cv2.applyColorMap(hmap, cv2.COLORMAP_JET), 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# some iu helper functions to render probability bars and other UI elements in the Streamlit app (again i dont know css)
def prob_bars_html(results: dict) -> str:
    rows = ""
    for label, score in results.items():
        pct   = score * 100
        width = f"{pct:.1f}%"
        color = "var(--accent)" if pct == max(v * 100 for v in results.values()) else "var(--accent-dim)"
        rows += f"""
        <div class="prob-row">
          <div class="prob-label">
            <span>{label}</span>
            <span style="font-family:var(--mono);color:{color}">{pct:.1f}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{width};background:{color}"></div>
          </div>
        </div>"""
    return rows

# image index to always show the same image across all tabs - it makes it easier to compare model outputs
_GLOBAL_IDX = "global_img_idx"

def _nav_prev(_k=_GLOBAL_IDX):
    st.session_state[_k] = max(0, st.session_state[_k] - 1)

def _nav_next(_k=_GLOBAL_IDX):
    st.session_state[_k] = min(len(IMAGES) - 1, st.session_state[_k] + 1)

def _nav_rand(_k=_GLOBAL_IDX):
    st.session_state[_k] = random.randint(0, len(IMAGES) - 1)

def _nav_sync_sb(sb_key, _k=_GLOBAL_IDX):
    st.session_state[_k] = IMAGES.index(st.session_state[sb_key])

def render_image_navigator(key: str):
    """Navigation bar that reads/writes the single global image index.

    All three tabs share _GLOBAL_IDX so switching tabs always shows the same
    image. The `key` argument is used only to give Streamlit unique widget IDs.
    """
    if not IMAGES:
        st.warning("No test images found.")
        return None

    if _GLOBAL_IDX not in st.session_state:
        st.session_state[_GLOBAL_IDX] = 0

    sb_key = f"sb_{key}"

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.button("◀  Prev",   key=f"prev_{key}", on_click=_nav_prev, use_container_width=True)
    with c2:
        st.button("Next  ▶",   key=f"next_{key}", on_click=_nav_next, use_container_width=True)
    with c3:
        st.button("⊞  Random", key=f"rand_{key}", on_click=_nav_rand, use_container_width=True)

    st.selectbox(
        "Select image",
        IMAGES,
        index=st.session_state[_GLOBAL_IDX],
        key=sb_key,
        on_change=_nav_sync_sb,
        kwargs={"sb_key": sb_key},
        label_visibility="collapsed",
    )
    st.caption(f"Image {st.session_state[_GLOBAL_IDX] + 1} / {len(IMAGES)}")
    return IMAGES[st.session_state[_GLOBAL_IDX]]

def render_upload_toggle(key: str):
    """Radio to choose between test dataset or uploaded file."""
    return st.radio(
        "Image source",
        ["📂  Test dataset", "⬆  Upload image"],
        horizontal=True,
        key=f"src_{key}",
        label_visibility="collapsed",
    )

def load_image_from_upload(key: str):
    """File uploader → BGR numpy. Returns None if nothing uploaded."""
    f = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], key=f"up_{key}")
    if f:
        arr = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None

# set sidebar content with app title, system status indicators, and threat class legend
with st.sidebar:
    st.markdown("""
    <div class="app-title">XRAY-SEC</div>
    <div class="app-subtitle">Baggage Inspection Console v1.0</div>
    <hr style="border-color:#1f2a35;margin:16px 0">
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)
    dataset_ok = TEST_IMG_DIR is not None and os.path.exists(TEST_IMG_DIR)
    models_ok  = os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0
    st.markdown(f"""
    <div class="scan-card" style="padding:12px 16px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="color:{'#00ff88' if dataset_ok else '#ff3b5c'}">{'●' if dataset_ok else '○'}</span>
        <span style="font-size:15px">Dataset {'linked' if dataset_ok else 'not found'}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="color:{'#00ff88' if models_ok else '#ff3b5c'}">{'●' if models_ok else '○'}</span>
        <span style="font-size:15px">Models dir {'ready' if models_ok else 'missing'}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:20px">Threat Classes</p>',
                unsafe_allow_html=True)
    color_hex = {0:"#00ff00", 1:"#ff3b5c", 2:"#ff8c00", 3:"#ffff00", 4:"#ff00ff"}
    for cid, name in CLASS_NAMES.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{color_hex[cid]};display:inline-block"></span>'
            f'<span style="font-family:var(--mono);font-size:15px">{name}</span></div>',
            unsafe_allow_html=True,
        )

# tabs
tab_pre, tab_base, tab_clip_ret, tab_clip = st.tabs([
    "  PRETRAINED MODELS  ",
    "  BASELINE MODEL  ",
    "  CLIP RETRIEVAL  ",
    "  CLIP ZERO-SHOT  ",
])

# tab 1 design - pretrained YOLO / RT-DETR models with selectable preprocessing pipelines
with tab_pre:
    st.markdown('<p class="section-header">Pretrained Detector — YOLO / RT-DETR</p>',
                unsafe_allow_html=True)

    # Model selector
    pt_files = []
    if os.path.exists(MODELS_DIR):
        pt_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".pt")])

    if not pt_files:
        st.warning("No `.pt` model files found in BEST_MODELS/")
        st.stop()

    sel_model = st.selectbox("Model checkpoint", pt_files, key="pre_model")
    pipeline  = infer_pipeline(sel_model)
    st.markdown(PIPELINE_BADGE[pipeline], unsafe_allow_html=True)

    model_pre = load_pretrained_model(os.path.join(MODELS_DIR, sel_model))

    st.markdown("---")

    # Source toggle
    src = render_upload_toggle("pre")
    uploaded_pre = None

    if "⬆" in src:
        uploaded_pre = load_image_from_upload("pre")
        if uploaded_pre is not None:
            proc = apply_pipeline(uploaded_pre, pipeline)
            pred = predict_pretrained(model_pre, proc)
            st.markdown('<p class="img-frame-label">Model Prediction</p>',
                        unsafe_allow_html=True)
            st.image(pred, use_container_width=True)
        else:
            st.info("Upload an image to run inference.")
    else:
        sel_img = render_image_navigator("pre")
        if sel_img:
            img_path = os.path.join(TEST_IMG_DIR, sel_img)
            lbl_path = os.path.join(TEST_LBL_DIR,
                                    os.path.splitext(sel_img)[0] + ".txt")
            img_bgr  = cv2.imread(img_path)
            proc     = apply_pipeline(img_bgr, pipeline)

            col_gt, col_pred = st.columns(2, gap="medium")
            with col_gt:
                st.markdown('<p class="img-frame-label">Ground Truth</p>',
                            unsafe_allow_html=True)
                st.image(draw_ground_truth(img_path, lbl_path),
                         use_container_width=True)
            with col_pred:
                st.markdown('<p class="img-frame-label">Model Prediction</p>',
                            unsafe_allow_html=True)
                st.image(predict_pretrained(model_pre, proc),
                         use_container_width=True)

# tab 2 design - baseline 
with tab_base:
    st.markdown('<p class="section-header">Baseline Detector — ResNet-18 CNN</p>',
                unsafe_allow_html=True)

    pth_files = []
    if os.path.exists(MODELS_DIR):
        pth_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".pth")])

    if not pth_files:
        st.warning("No `.pth` model files found in BEST_MODELS/")
    else:
        sel_base = st.selectbox("Baseline checkpoint", pth_files, key="base_model")
        st.markdown(PIPELINE_BADGE['P1'], unsafe_allow_html=True)

        model_base = load_baseline_model(os.path.join(MODELS_DIR, sel_base))

        st.markdown('<div class="scan-card scan-card-accent" style="font-size:12px;color:#5a7080">'
                    'Baseline model. Images are preprocessed with Pipeline 1 (CLAHE + Unsharp Mask) '
                    'before inference. ResNet-18 backbone with classification and box regression heads.</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        src_b = render_upload_toggle("base")

        if "⬆" in src_b:
            uploaded_base = load_image_from_upload("base")
            if uploaded_base is not None:
                uploaded_base_proc = preprocess_p1(uploaded_base)
                pred_rgb, cls_id, conf = predict_baseline(model_base, uploaded_base_proc)
                st.markdown('<p class="img-frame-label">Model Prediction</p>',
                            unsafe_allow_html=True)
                st.image(pred_rgb, use_container_width=True)
                cls_name = CLASS_NAMES.get(cls_id - 1, "Background") if cls_id > 0 else "Background"
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-chip">
                    <span class="val">{cls_name}</span>
                    <span class="lbl">Predicted Class</span>
                  </div>
                  <div class="metric-chip">
                    <span class="val">{conf*100:.1f}%</span>
                    <span class="lbl">Confidence</span>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Upload an image to run inference.")
        else:
            sel_img_b = render_image_navigator("base")
            if sel_img_b:
                img_path_b = os.path.join(TEST_IMG_DIR, sel_img_b)
                lbl_path_b = os.path.join(TEST_LBL_DIR,
                                          os.path.splitext(sel_img_b)[0] + ".txt")
                img_bgr_b  = cv2.imread(img_path_b)
                img_bgr_b_proc = preprocess_p1(img_bgr_b)

                pred_rgb, cls_id, conf = predict_baseline(model_base, img_bgr_b_proc)
                col_gt_b, col_pred_b = st.columns(2, gap="medium")
                with col_gt_b:
                    st.markdown('<p class="img-frame-label">Ground Truth</p>',
                                unsafe_allow_html=True)
                    st.image(draw_ground_truth(img_path_b, lbl_path_b),
                             use_container_width=True)
                with col_pred_b:
                    st.markdown('<p class="img-frame-label">Model Prediction</p>',
                                unsafe_allow_html=True)
                    st.image(pred_rgb, use_container_width=True)
                    cls_name = CLASS_NAMES.get(cls_id - 1, "Background") if cls_id > 0 else "Background"
                    st.markdown(f"""
                    <div class="metric-row" style="margin-top:10px">
                      <div class="metric-chip">
                        <span class="val">{cls_name}</span>
                        <span class="lbl">Predicted Class</span>
                      </div>
                      <div class="metric-chip">
                        <span class="val">{conf*100:.1f}%</span>
                        <span class="lbl">Confidence</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

# tab 3 design - CLIP retrieval over stored OpenCLIP crop embeddings
with tab_clip_ret:
    st.markdown('<p class="section-header">CLIP Retrieval — Detected Region Similarity</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="scan-card scan-card-accent" style="font-size:12px;color:#5a7080">'
        f'Uses detected regions from the selected pretrained model, encodes each crop with '
        f'OpenCLIP {OPENCLIP_MODEL_NAME} / {OPENCLIP_PRETRAINED}, and retrieves the top 3 '
        'nearest stored crop embeddings.</div>',
        unsafe_allow_html=True,
    )

    retrieval_ready = (
        os.path.exists(os.path.join(OUTPUTS_DIR, "embeddings.npy"))
        and os.path.exists(os.path.join(OUTPUTS_DIR, "embeddings_metadata.csv"))
    )
    if not retrieval_ready:
        st.warning("Missing retrieval assets in outputs/: embeddings.npy and embeddings_metadata.csv")
    else:
        st.markdown(PIPELINE_BADGE[pipeline], unsafe_allow_html=True)
        st.caption(f"Detector: {sel_model}")
        st.markdown("---")

        src_r = render_upload_toggle("clip_ret")
        opencv_ret = None

        if "â¬†" in src_r:
            opencv_ret = load_image_from_upload("clip_ret")
            if opencv_ret is None:
                st.info("Upload an image to run CLIP retrieval.")
        else:
            sel_img_r = render_image_navigator("clip_ret")
            if sel_img_r:
                opencv_ret = cv2.imread(os.path.join(TEST_IMG_DIR, sel_img_r))

        if opencv_ret is not None:
            min_conf = st.slider(
                "Minimum detector confidence",
                0.05, 0.95, 0.25, 0.05,
                key="clip_ret_min_conf",
            )

            proc_ret = apply_pipeline(opencv_ret, pipeline)
            annotated_ret, regions = extract_pretrained_regions(
                model_pre,
                proc_ret,
                opencv_ret,
                min_conf,
            )

            col_det, col_res = st.columns([1, 1.5], gap="large")
            with col_det:
                st.markdown('<p class="img-frame-label">Detected Regions</p>',
                            unsafe_allow_html=True)
                st.image(annotated_ret, use_container_width=True)
                st.caption(f"{len(regions)} region(s) above confidence threshold.")

            with col_res:
                if not regions:
                    st.info("No regions detected. Lower the confidence threshold or try another image.")
                else:
                    for region_idx, region in enumerate(regions, start=1):
                        st.markdown(
                            f'<p class="section-header">Region {region_idx} — '
                            f'{region["class_name"]} {region["confidence"]:.2f}</p>',
                            unsafe_allow_html=True,
                        )

                        query_embedding = encode_openclip_crop(region["crop_bgr"])
                        top_rows = retrieve_similar_crops(query_embedding, top_k=3)

                        q_col, r1, r2, r3 = st.columns([1, 1, 1, 1], gap="small")
                        with q_col:
                            st.image(cv2.cvtColor(region["crop_bgr"], cv2.COLOR_BGR2RGB),
                                     use_container_width=True)
                            st.caption("Query crop")

                        for col, (_, row) in zip([r1, r2, r3], top_rows.iterrows()):
                            with col:
                                match_img = resolve_metadata_crop(row)
                                if match_img is not None:
                                    st.image(match_img, use_container_width=True)
                                else:
                                    st.markdown(
                                        '<div style="background:var(--bg-panel);border:1px dashed var(--border);'
                                        'border-radius:4px;height:120px;display:flex;align-items:center;'
                                        'justify-content:center;color:var(--text-dim);font-size:11px;'
                                        'letter-spacing:0.12em">CROP NOT FOUND</div>',
                                        unsafe_allow_html=True,
                                    )
                                st.caption(
                                    f"#{int(row['rank'])} {row['class_name']} | "
                                    f"{row['split']} | score={row['score']:.3f}"
                                )

# tab 4 design - CLIP zero-shot classification with attention heatmap visualization
with tab_clip:
    st.markdown('<p class="section-header">CLIP — Zero-Shot Classification & Attention</p>',
                unsafe_allow_html=True)

    src_c = render_upload_toggle("clip")
    opencv_clip = None

    if "⬆" in src_c:
        opencv_clip = load_image_from_upload("clip")
        if opencv_clip is None:
            st.info("Upload an image to activate CLIP modules.")
    else:
        sel_img_c = render_image_navigator("clip")
        if sel_img_c:
            opencv_clip = cv2.imread(os.path.join(TEST_IMG_DIR, sel_img_c))

    if opencv_clip is not None:
        st.markdown("---")
        col_img_c, col_mods = st.columns([1, 1.3], gap="large")

        with col_img_c:
            st.markdown('<p class="img-frame-label">Input Image</p>',
                        unsafe_allow_html=True)
            st.image(cv2.cvtColor(opencv_clip, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

            st.markdown('<p class="section-header" style="margin-top:18px">Attention Heatmap</p>',
                        unsafe_allow_html=True)
            if st.button("Generate heatmap", key="gen_heatmap", use_container_width=True):
                with st.spinner("Extracting transformer attention weights…"):
                    st.session_state["clip_heatmap"] = clip_heatmap(opencv_clip)

            if "clip_heatmap" in st.session_state and st.session_state["clip_heatmap"] is not None:
                st.image(st.session_state["clip_heatmap"], use_container_width=True)
                st.caption("Warm tones mark regions with highest attention weight.")
            else:
                st.markdown(
                    '<div style="background:var(--bg-panel);border:1px dashed var(--border);'
                    'border-radius:4px;height:140px;display:flex;align-items:center;'
                    'justify-content:center;color:var(--text-dim);font-size:11px;'
                    'letter-spacing:0.12em">HEATMAP NOT GENERATED</div>',
                    unsafe_allow_html=True,
                )

        with col_mods:
            mode = st.radio(
                "Evaluation mode",
                ["Standard class identification", "Free-form prompting"],
                key="clip_mode",
                label_visibility="collapsed",
            )
            st.markdown("---")

            # mode 1: standard zero-shot classification with fixed prompts for the 5 threat classes + no-threat baseline
            if mode == "Standard class identification":
                st.markdown('<p class="section-header">Standard 5-Class Assessment</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:var(--text-dim);margin-bottom:14px">'
                    'CLIP scores the likelihood of each threat category and a no-threat baseline '
                    'using zero-shot contrastive matching.</div>',
                    unsafe_allow_html=True,
                )
                FIXED_PROMPTS = [
                    "An X-ray image containing scissors",
                    "An X-ray image containing a gun",
                    "An X-ray image containing a knife",
                    "An X-ray image containing pliers",
                    "An X-ray image containing a wrench",
                    "A clean X-ray image with no threats",
                ]
                if st.button("▶  Run Standard Analysis", key="clip_fixed_run",
                             use_container_width=True):
                    with st.spinner("Matching embeddings…"):
                        st.session_state["clip_fixed"] = clip_classify(opencv_clip, FIXED_PROMPTS)

                if "clip_fixed" in st.session_state and st.session_state["clip_fixed"]:
                    st.markdown(
                        prob_bars_html(st.session_state["clip_fixed"]),
                        unsafe_allow_html=True,
                    )

            # mode 2: free-form
            else:
                st.markdown('<p class="section-header">Free-Form Query</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div class="scan-card" style="font-size:12px;color:var(--text-dim);padding:10px 14px;margin-bottom:12px">'
                    '⚠ CLIP distributes probability across all candidates. '
                    'Enter <b>at least 2</b> categories separated by commas.</div>',
                    unsafe_allow_html=True,
                )
                user_opts = st.text_area(
                    "Candidate categories (comma-separated)",
                    value="A pair of scissors, A firearm, A knife, A wrench, No dangerous object",
                    height=80,
                    key="clip_free_opts",
                )
                candidates = [o.strip() for o in user_opts.split(",") if o.strip()]

                if st.button("▶  Evaluate", key="clip_free_run", use_container_width=True):
                    if len(candidates) < 2:
                        st.warning("Enter at least 2 candidates.")
                    else:
                        with st.spinner("Processing open semantics…"):
                            st.session_state["clip_free"] = clip_classify(opencv_clip, candidates)

                if "clip_free" in st.session_state and st.session_state["clip_free"]:
                    st.markdown(
                        prob_bars_html(st.session_state["clip_free"]),
                        unsafe_allow_html=True,
                    )
