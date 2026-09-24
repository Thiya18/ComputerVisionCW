"""
streamlit_app.py
================
Streamlit UI for Diabetic Retinopathy Stage Detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

import src.config as cfg
from src.preprocessing import preprocess_image

st.set_page_config(page_title="DR Stage Detection", page_icon="👁️", layout="centered")

# The model was trained with flow_from_directory using these alphabetical folder names:
MODEL_CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferative_DR", "Severe"]

@st.cache_resource
def load_dr_model():
    """Load the final model from config path."""
    return tf.keras.models.load_model(str(cfg.FINAL_MODEL_PATH))

model = load_dr_model()

st.title("👁️ Diabetic Retinopathy Stage Detection")
st.markdown("""
Upload a retinal fundus image to predict the severity of Diabetic Retinopathy.
The image will be preprocessed (resized, CLAHE, denoised, sharpened) automatically.
""")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Uploaded Image")
    
    # Read via PIL and display
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    st.image(image_pil, use_column_width=True, caption="Original Upload")
    
    with st.spinner("Processing image and predicting..."):
        # Save uploaded image temporarily so we can pass its path to preprocess_image()
        temp_path = Path("temp_upload.png")
        cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        try:
            # Re-use the exact pipeline from src/preprocessing.py
            processed_img = preprocess_image(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
        # Model expects batch dimension: shape (1, 224, 224, 3)
        input_tensor = np.expand_dims(processed_img, axis=0)
        
        # Predict
        probs = model.predict(input_tensor)[0]
        pred_idx = np.argmax(probs)
        pred_class = MODEL_CLASS_NAMES[pred_idx]
        
    st.subheader("Prediction Results")
    st.success(f"**Predicted Stage:** {pred_class.replace('_', ' ')}")
    
    st.write("**Confidence Scores:**")
    for i, class_name in enumerate(MODEL_CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{class_name.replace('_', ' ')}: {probs[i]:.2%}")
