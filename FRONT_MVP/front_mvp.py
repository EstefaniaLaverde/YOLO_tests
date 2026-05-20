import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import kagglehub

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(page_title="X-Ray Baggage Anomaly Detection", layout="wide")
st.title("X-Ray Baggage Anomaly Detection - Visual Evaluation")
st.markdown("Selecciona un modelo, un pipeline de preprocesamiento y evalúa su rendimiento frente a las anotaciones reales.")

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y SELECCIÓN DINÁMICA DE MODELOS
# ==============================================================================
MODELS_DIR = "BEST_MODELS"  # carpeta con archivos .pt de modelos entrenados
try:
    # kagglehub detecta automáticamente si ya está descargado y devuelve la ruta absoluta correcta
    base_kaggle_path = kagglehub.dataset_download("orvile/x-ray-baggage-anomaly-detection")
    
    # Construimos las rutas apuntando directas a la caché de kagglehub
    TEST_IMAGES_DIR = os.path.join(base_kaggle_path, "test_processed", "images")
    TEST_LABELS_DIR = os.path.join(base_kaggle_path, "test_processed", "labels")
    
except Exception as e:
    st.error(f"Error al conectar con Kagglehub: {e}")
    TEST_IMAGES_DIR = None
    TEST_LABELS_DIR = None

# Mapeo de nombres de clases (Ajusta los nombres según tus IDs 0, 1, 2, 3, 4)
CLASS_NAMES = {0: "Scissors", 1: "Knife", 2: "Pliers", 3: "Gun", 4: "Wrench"}

# Colores fijos para cada clase en formato BGR
CLASS_COLORS = {
    0: (0, 0, 255),    # Rojo
    1: (255, 165, 0),  # Naranja
    2: (255, 255, 0),  # Amarillo
    3: (0, 255, 0),    # Verde
    4: (255, 0, 255)   # Magenta
}

@st.cache_resource
def load_yolo_model(path):
    """Carga el modelo YOLO y lo mantiene en caché para evitar recargas lentas."""
    return YOLO(path)

# ==============================================================================
# PIPELINES DE PREPROCESAMIENTO
# ==============================================================================
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
    # usar CLAHE para mejorar el contraste
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # luego aplicar un filtro bilateral para suavizar el ruido y mantener los bordes lo mejor posible
    filtered = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # resaltar los contornos mediante gradiente morfologico - exhalta los objetos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

    # combinar la imagen filtrada con los bordes - darle un 90% de peso a la imagen limpia y un 10% a los bordes puros
    img_final = cv2.addWeighted(filtered, 0.9, gradient, 0.1, 0)

    return img_final

# ==============================================================================
# FUNCIONES AUXILIARES DE DIBUJO DE ANOTACIONES
# ==============================================================================
def draw_ground_truth(img_path, label_path):
    """Lee el archivo .txt en formato YOLO y dibuja las cajas reales en la imagen."""
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
                # Convertir coordenadas YOLO normalized (x_center, y_center, width, height) a píxeles
                x_c, y_c, bbox_w, bbox_h = map(float, parts[1:])
                x1 = int((x_c - bbox_w / 2) * w)
                y1 = int((y_c - bbox_h / 2) * h)
                x2 = int((x_c + bbox_w / 2) * w)
                y2 = int((y_c + bbox_h / 2) * h)
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label_text = CLASS_NAMES.get(cls_id, f"Clase {cls_id}")
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_predictions(processed_img, model_instance):
    """Ejecuta la inferencia de YOLO sobre la imagen preprocesada y dibuja los resultados."""
    img_canvas = processed_img.copy()
    
    # Inferencia (YOLO acepta arrays en BGR)
    results = model_instance.predict(img_canvas, verbose=False)[0]
    
    # Dibujar las cajas predichas manualmente para mantener consistencia visual con el GT
    if results.boxes is not None:
        for box in results.boxes:
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords)
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label_text = f"{CLASS_NAMES.get(cls_id, f'Clase {cls_id}')} {conf:.2f}"
            
            cv2.rectangle(img_canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_canvas, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)

# ==============================================================================
# INTERFAZ DE STREAMLIT (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # 1. Selección Dinámica de Pesos del Modelo (.pt)
    st.subheader("🤖 Pesos del Modelo")
    if os.path.exists(MODELS_DIR):
        available_models = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith('.pt')]
        if available_models:
            selected_model_name = st.selectbox(
                "Escoge el modelo YOLO para evaluar:", 
                sorted(available_models)
            )
            model_full_path = os.path.join(MODELS_DIR, selected_model_name)
            model = load_yolo_model(model_full_path)
        else:
            st.error(f"No se encontraron archivos .pt dentro de la carpeta '{MODELS_DIR}'")
            st.stop()
    else:
        st.error(f"La carpeta '{MODELS_DIR}' no existe en el directorio raíz.")
        st.stop()
        
    st.markdown("---")

    # 2. Selección del Pipeline de preprocesamiento
    pipeline_option = st.selectbox(
        "Selecciona el Pipeline de Preprocesamiento:",
        ("Pipeline 1 (CLAHE + Sharpening)", "Pipeline 2 (Advanced X-Ray Edge Enhancement)")
    )
    
    # 3. Selección de la imagen de Test con Navegación Avanzada
    st.subheader("📁 Selección de Imagen")
    img_src_path = None
    label_src_path = None
    
    if os.path.exists(TEST_IMAGES_DIR):
        available_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if available_images:
            # 1. Inicializar el índice en el state si no existe
            if "img_index" not in st.session_state:
                st.session_state.img_index = 0

            # 2. Callbacks para manejar el cambio de estado de forma segura antes del render
            def ir_atras():
                if st.session_state.img_index > 0:
                    st.session_state.img_index -= 1

            def ir_adelante():
                if st.session_state.img_index < len(available_images) - 1:
                    st.session_state.img_index += 1

            # Callback cuando el usuario cambia manualmente el selectbox
            def al_cambiar_selector():
                if "img_selector" in st.session_state:
                    # Buscamos el string seleccionado y actualizamos el índice numérico
                    st.session_state.img_index = available_images.index(st.session_state.img_selector)

            # 3. Renderizar botones lado a lado con sus respectivos callbacks
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button("⬅️ Anterior", on_click=ir_atras, use_container_width=True)
            with btn_col2:
                st.button("Siguiente ➡️", on_click=ir_adelante, use_container_width=True)

            # 4. Renderizar el selector sincronizado con el índice del state
            selected_img_name = st.selectbox(
                "Elige una imagen de la carpeta de test:", 
                available_images,
                index=st.session_state.img_index,
                key="img_selector",
                on_change=al_cambiar_selector  # Sincroniza si el usuario usa el dropdown
            )
            
            # Asegurar consistencia de la variable para el resto del script
            selected_img_name = available_images[st.session_state.img_index]
            
            img_src_path = os.path.join(TEST_IMAGES_DIR, selected_img_name)
            base_name = os.path.splitext(selected_img_name)[0]
            label_src_path = os.path.join(TEST_LABELS_DIR, f"{base_name}.txt")
            
            # Muestra el contador de progreso actual
            st.caption(f"Imagen {st.session_state.img_index + 1} de {len(available_images)}")
        else:
            st.warning("No se encontraron imágenes en la ruta de test.")
    else:
        st.error(f"No se pudo acceder al directorio: {TEST_IMAGES_DIR}")

    # Permitir subir una imagen externa si se desea
    uploaded_file = st.file_uploader("O sube una imagen externa (.jpg, .png):", type=["jpg", "jpeg", "png"])

# ==============================================================================
# PROCESAMIENTO VISUAL LADO A LADO
# ==============================================================================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gt_display = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    st.info("ℹ️ Cargando imagen externa. No se dispone de anotaciones reales (Ground Truth).")
    
    if "Pipeline 1" in pipeline_option:
        processed_img = preprocess_pipeline_1(opencv_img)
    else:
        processed_img = preprocess_pipeline_2(opencv_img)
        
    pred_display = draw_predictions(processed_img, model)

elif img_src_path is not None:
    opencv_img = cv2.imread(img_src_path)
    gt_display = draw_ground_truth(img_src_path, label_src_path)
    
    if "Pipeline 1" in pipeline_option:
        processed_img = preprocess_pipeline_1(opencv_img)
    else:
        processed_img = preprocess_pipeline_2(opencv_img)
        
    pred_display = draw_predictions(processed_img, model)
else:
    st.info("Por favor, configura las rutas del dataset o añade imágenes para comenzar.")
    st.stop()

# Desplegar imágenes lado a lado usando el layout de columnas de Streamlit
col1, col2 = st.columns(2)

with col1:
    st.subheader("Objetos Reales")
    st.image(gt_display, use_container_width=True, caption="Etiquetas originales grabadas en el dataset")

with col2:
    st.subheader("Predicciones")
    st.image(pred_display, use_container_width=True, caption=f"Inferencia con el modelo: {selected_model_name} usando {pipeline_option}")