"""
streamlit_app.py
================
Streamlit UI for Diabetic Retinopathy Stage Detection,
with Grad-CAM explainability, pre-processing preview, and a chatbot Q&A.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown

import src.config as cfg
from src.preprocessing import preprocess_image

st.set_page_config(page_title="DR Stage Detection", page_icon="👁️", layout="centered")

# ── Grad-CAM Explainability ───────────────────────────────────────────────────

def make_gradcam_heatmap(img_tensor: np.ndarray, model: tf.keras.Model, pred_index: int = None) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap by isolating the EfficientNetB3 base and custom head.
    Dynamically finds the classification layers instead of relying on hardcoded names.
    """
    # 1. Extract the base model
    base_model = model.get_layer("efficientnetb3")
    
    # Find the index of the base model in the full model's layers
    base_model_idx = -1
    for i, layer in enumerate(model.layers):
        if layer.name == "efficientnetb3":
            base_model_idx = i
            break
            
    if base_model_idx == -1:
        raise ValueError("Could not find 'efficientnetb3' layer in the model.")
        
    # 2. Build a sub-model for the classification head using all subsequent layers
    top_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = top_input
    
    # Iterate through all layers that come after the base model
    for layer in model.layers[base_model_idx + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.Model(top_input, x)

    # 3. Compute gradients of the top predicted class wrt the base model's output feature map
    with tf.GradientTape() as tape:
        # Forward pass through base model
        conv_outputs = base_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        
        # Forward pass through classifier
        preds = classifier_model(conv_outputs, training=False)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over the spatial dimensions (H, W)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map channels by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays the Grad-CAM heatmap onto the original image.
    img: original image array (RGB, uint8)
    heatmap: 2D array of heatmap values in [0, 1]
    """
    # Rescale heatmap to a range 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    
    # Apply colormap and convert back to RGB for display
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) 
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    return superimposed_img

# ── Class mapping (alphabetical — matches flow_from_directory order) ──────────
MODEL_CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferative_DR", "Severe"]

# ── Per-class educational Q&A responses ──────────────────────────────────────
CLASS_INFO = {
    "No_DR": {
        "what_does_it_mean": (
            "**No Diabetic Retinopathy (No DR)** means no visible signs of diabetic "
            "eye disease were detected in this image. The blood vessels in the retina "
            "appear healthy with no microaneurysms, haemorrhages, or other lesions "
            "associated with DR."
        ),
        "what_next": (
            "Continue routine eye examinations as recommended by your doctor — typically "
            "once a year for people with diabetes. Maintain good control of blood sugar, "
            "blood pressure, and cholesterol, as these are the primary risk factors for "
            "developing DR in the future."
        ),
        "how_serious": (
            "This is the most favourable finding. No current retinal damage is detected. "
            "However, people with diabetes remain at risk of developing DR over time, so "
            "regular screening is still essential."
        ),
    },
    "Mild": {
        "what_does_it_mean": (
            "**Mild Non-Proliferative DR (NPDR)** is the earliest stage of diabetic "
            "retinopathy. Small areas of swelling (microaneurysms) appear in the blood "
            "vessels of the retina. Vision is usually not yet affected at this stage."
        ),
        "what_next": (
            "Discuss this finding with your doctor or optometrist. Increased monitoring "
            "(e.g. 6–12 monthly eye reviews) is typically recommended. Tightening control "
            "of blood glucose (HbA1c), blood pressure, and blood lipids can slow or halt "
            "progression."
        ),
        "how_serious": (
            "Mild DR is an early warning sign. While vision is generally unaffected at "
            "this stage, it indicates that the diabetes is beginning to affect the retinal "
            "microvasculature. Early intervention through lifestyle and medical management "
            "can be very effective."
        ),
    },
    "Moderate": {
        "what_does_it_mean": (
            "**Moderate Non-Proliferative DR (NPDR)** indicates more widespread damage to "
            "retinal blood vessels. In addition to microaneurysms, there may be dot-and-blot "
            "haemorrhages, hard exudates, and cotton-wool spots visible on the retina."
        ),
        "what_next": (
            "A referral to an ophthalmologist (eye specialist) is advisable. More frequent "
            "monitoring — often every 3–6 months — is typically recommended. Your doctor "
            "will review your diabetes management plan. In some cases, treatment such as "
            "anti-VEGF injections or laser therapy may be discussed."
        ),
        "how_serious": (
            "Moderate DR represents a meaningful increase in risk compared to Mild DR. "
            "There is a higher chance of progressing to vision-threatening stages if the "
            "underlying diabetes is not well controlled. However, with appropriate care, "
            "progression can often be slowed significantly."
        ),
    },
    "Severe": {
        "what_does_it_mean": (
            "**Severe Non-Proliferative DR (NPDR)** means a large number of blood vessels "
            "in the retina are blocked or damaged. The retina signals the body to grow new "
            "blood vessels, which sets the stage for the most advanced form of DR. "
            "Significant haemorrhages are present in all four quadrants of the retina."
        ),
        "what_next": (
            "Prompt referral to a specialist ophthalmologist or retinal surgeon is strongly "
            "recommended. Treatment options such as pan-retinal photocoagulation (laser) or "
            "anti-VEGF injections may be indicated. Do not delay in seeking specialist review."
        ),
        "how_serious": (
            "Severe DR carries a high risk (around 50% within one year) of progressing to "
            "Proliferative DR, the most advanced and sight-threatening stage. Timely "
            "specialist intervention is important to preserve vision."
        ),
    },
    "Proliferative_DR": {
        "what_does_it_mean": (
            "**Proliferative DR (PDR)** is the most advanced stage of diabetic retinopathy. "
            "Abnormal new blood vessels (neovascularisation) grow on the surface of the retina "
            "or into the vitreous. These fragile vessels can bleed, causing severe vision loss, "
            "and can lead to tractional retinal detachment."
        ),
        "what_next": (
            "Urgent referral to a retinal specialist is needed. Treatment — which may include "
            "anti-VEGF injections, extensive laser photocoagulation, or vitrectomy surgery — "
            "should be arranged as soon as possible. Do not wait. This is a sight-threatening "
            "condition requiring immediate specialist care."
        ),
        "how_serious": (
            "Proliferative DR is the most serious stage and a leading cause of blindness in "
            "working-age adults. Without treatment, vision loss can be rapid and permanent. "
            "With prompt and appropriate treatment, significant vision can often be preserved."
        ),
    },
}

QUESTION_KEYS = {
    "What does this stage mean?": "what_does_it_mean",
    "What should I do next?":     "what_next",
    "How serious is this?":       "how_serious",
}

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_dr_model():
    """Load the final model from config path, downloading from Google Drive if missing."""
    model_path = Path(cfg.FINAL_MODEL_PATH)
    
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        file_id = "1PlfXoaK7Tf3KxVRd0CqgfNLf3VEsAWmi"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        with st.spinner("Downloading model weights from Google Drive... (this may take a minute)"):
            gdown.download(url, str(model_path), quiet=False)
            
    return tf.keras.models.load_model(str(model_path))

model = load_dr_model()

# ── Session state initialisation ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (question_label, answer_text)
if "last_pred_class" not in st.session_state:
    st.session_state.last_pred_class = None

# ── Page header ───────────────────────────────────────────────────────────────
st.title("👁️ Diabetic Retinopathy Stage Detection")
st.markdown(
    "Upload a retinal fundus image to predict the severity of Diabetic Retinopathy. "
    "The image will be preprocessed (CLAHE, resize, normalise) automatically."
)

# ── Image upload & prediction ─────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Image Preprocessing Preview")
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np  = np.array(image_pil)
    
    # Show before and after side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, use_container_width=True, caption="1. Original Upload")

    with col2:
        with st.spinner("Processing image and predicting..."):
            temp_path = Path("temp_upload.png")
            cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            try:
                processed_img = preprocess_image(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            # Reverse ImageNet normalization to display the preprocessed image properly
            display_processed = (processed_img * cfg.IMAGENET_STD + cfg.IMAGENET_MEAN) * 255
            display_processed = np.clip(display_processed, 0, 255).astype(np.uint8)
            st.image(display_processed, use_container_width=True, caption="2. Preprocessed (CLAHE + Denoise + Sharpen)")

            # Prediction
            input_tensor = np.expand_dims(processed_img, axis=0)
            probs        = model.predict(input_tensor)[0]
            pred_idx     = int(np.argmax(probs))
            pred_class   = MODEL_CLASS_NAMES[pred_idx]
            max_prob     = float(probs[pred_idx])

    # Reset chat history when a new image produces a different prediction
    if pred_class != st.session_state.last_pred_class:
        st.session_state.chat_history    = []
        st.session_state.last_pred_class = pred_class

    # ── Prediction results & Grad-CAM ─────────────────────────────────────────
    st.divider()
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Prediction Results")
        # Safety flag for low confidence predictions
        if max_prob < 0.5:
            st.warning(
                f"⚠️ **Low confidence prediction ({max_prob:.1%})** — "
                "manual specialist review is strongly recommended."
            )
            st.write(f"**Tentative Stage:** {pred_class.replace('_', ' ')}")
        else:
            st.success(f"**Predicted Stage:** {pred_class.replace('_', ' ')}")
            
        st.write("**Confidence Scores:**")
        for i, class_name in enumerate(MODEL_CLASS_NAMES):
            st.progress(float(probs[i]), text=f"{class_name.replace('_', ' ')}: {probs[i]:.2%}")

    with col4:
        st.subheader("Grad-CAM Explainability")
        st.caption("Heatmap showing regions that most influenced this prediction.")
        
        try:
            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(input_tensor, model, pred_idx)
            
            # Overlay heatmap on the original image (resized to model dimensions)
            orig_resized = cv2.resize(image_np, cfg.IMAGE_SIZE)
            gradcam_img = overlay_heatmap(orig_resized, heatmap)
            
            st.image(gradcam_img, use_container_width=True, caption="Grad-CAM Overlay")
        except Exception as e:
            # Fallback if Grad-CAM generation fails for any reason
            st.error(f"Grad-CAM unavailable: {str(e)}")

    # ── Chatbot section ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask About This Result")
    st.caption(
        "⚠️ **Disclaimer:** This information is for educational purposes only and is "
        "**not** a substitute for professional medical diagnosis, advice, or treatment. "
        "Always consult a qualified healthcare professional."
    )

    # Quick-question buttons rendered side-by-side
    cols = st.columns(len(QUESTION_KEYS))
    for col, question_label in zip(cols, QUESTION_KEYS):
        if col.button(question_label, key=f"btn_{question_label}"):
            answer_key = QUESTION_KEYS[question_label]
            answer     = CLASS_INFO[pred_class][answer_key]
            # Avoid duplicating the last question if clicked twice in a row
            if not st.session_state.chat_history or \
               st.session_state.chat_history[-1][0] != question_label:
                st.session_state.chat_history.append((question_label, answer))

    # Render accumulated chat history as message bubbles
    if st.session_state.chat_history:
        st.markdown("---")
        for q_label, a_text in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q_label)
            with st.chat_message("assistant", avatar="👁️"):
                st.markdown(a_text)
