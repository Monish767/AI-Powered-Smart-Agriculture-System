import os
import json
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import onnxruntime as ort

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
ONNX_NAME = "PlantCareAI_model.onnx"
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
IMAGE_SIZE = (224, 224)

# --- UI Configuration ---
PAGE_CONFIG = {
    "page_title": "🌿 PlantCare AI - Disease Classifier",
    "page_icon": "🌱",
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}

# --- Main Functions ---
@st.cache_resource
def load_model(class_indices):
    """Loads the model from ONNX file using ONNX Runtime."""
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found: {MODEL_DIR}")
        return None
    
    # Check for ONNX model
    onnx_candidates = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.onnx')]
    
    if not onnx_candidates:
        st.error("""
        ❌ **ONNX model not found!**
        
        You need to convert your Keras H5 model to ONNX format first.
        
        **Option 1: Convert using Python 3.11/3.12 (one-time conversion)**
        1. Install Python 3.11 or 3.12
        2. Install: `pip install tensorflow tf2onnx`
        3. Run: `python convert_to_onnx.py`
        4. Copy the .onnx file to trained_model folder
        
        **Option 2: Use online converter or ask someone with Python 3.11/3.12 to convert it**
        """)
        return None
    
    preferred = [f for f in onnx_candidates if f == ONNX_NAME]
    onnx_file = preferred[0] if preferred else sorted(onnx_candidates)[-1]
    onnx_path = os.path.join(MODEL_DIR, onnx_file)
    
    try:
        with st.spinner("Loading model with ONNX Runtime..."):
            # Create ONNX Runtime inference session
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            st.success("✅ Model loaded successfully with ONNX Runtime")
            return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def load_class_indices(json_path):
    """Loads class indices from a JSON file."""
    if not os.path.exists(json_path):
        st.error(f"Class indices file not found at: {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading class indices: {e}")
        return None

def preprocess_image(image):
    """Preprocesses an image for model prediction."""
    # Resize to model input size
    image = image.resize(IMAGE_SIZE)
    # Convert to array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_class_name(class_name):
    """Formats PlantVillage class names for better readability."""
    if not class_name:
        return "Unknown"
    
    # Replace underscores with spaces and format
    formatted = class_name.replace("___", " - ").replace("_", " ")
    
    # Capitalize first letter of each word
    words = formatted.split()
    formatted = " ".join(word.capitalize() for word in words)
    
    return formatted

def predict(session, image, class_indices, top_k=5):
    """Predicts the class of an image and returns top-k predictions."""
    if session is None or class_indices is None:
        return None, None, None

    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get input name from the model
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: processed_image})
        predictions = outputs[0][0]  # Get first (and only) batch item
        
        # Apply softmax
        exp_predictions = np.exp(predictions - np.max(predictions))
        probabilities = exp_predictions / np.sum(exp_predictions)
        
        # Get top prediction
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_k_probs = probabilities[top_k_indices]
        
        # Format top-k results
        top_predictions = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            class_name = class_indices.get(str(int(idx)), f"Class_{int(idx)}")
            formatted_name = format_class_name(class_name)
            top_predictions.append({
                'class': formatted_name,
                'original_class': class_name,
                'confidence': float(prob),
                'index': int(idx)
            })

        # Get the predicted class name
        predicted_class = class_indices.get(str(predicted_index), f"Class_{predicted_index}")
        formatted_class = format_class_name(predicted_class)
        
        return formatted_class, confidence, top_predictions
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

# --- UI Components ---
def set_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)

def display_background_pattern():
    """Displays the background pattern based on user selection."""
    style = st.session_state.get("bg_style", "Leaf Pattern")
    if style == "Soft Gradient":
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif style == "Leafy Gradient":
        st.markdown(
            """
            <style>
            .stApp {
                background-image: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%),
                                  url('https://www.transparenttextures.com/patterns/leafy-green.png');
                background-blend-mode: overlay;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: url('https://www.transparenttextures.com/patterns/leafy-green.png');
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

def display_sidebar(session, class_indices):
    """Displays the sidebar with instructions and model status."""
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
            1. **Upload** a clear image of a plant leaf
            2. **Click** the 'Analyze Image' button
            3. **Review** the diagnosis and confidence score
            4. **Check** top predictions for alternative diagnoses
        """)
        
        st.markdown("---")
        st.header("🔧 Model Status")
        if session:
            st.success("✅ Model Ready")
            st.caption("Backend: ONNX Runtime (Python 3.14 compatible)")
        else:
            st.error("❌ Model Not Loaded")
        
        if class_indices:
            st.success(f"✅ Classes Loaded ({len(class_indices)} classes)")
        else:
            st.error("❌ Classes Not Loaded")
        
        st.markdown("---")
        st.header("🎨 Background")
        st.session_state.bg_style = st.selectbox(
            "Background Style",
            ["Leaf Pattern", "Soft Gradient", "Leafy Gradient"],
            index=["Leaf Pattern", "Soft Gradient", "Leafy Gradient"].index(
                st.session_state.get("bg_style", "Leaf Pattern")
            ),
        )
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.caption("PlantCare AI uses deep learning to identify plant diseases from leaf images. Trained on the PlantVillage dataset.")

def display_results(predicted_class, confidence, top_predictions=None):
    """Displays the prediction results with top-k predictions."""
    if predicted_class and confidence is not None:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main prediction
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🎯 Diagnosis")
            st.markdown(f"**{predicted_class}**")
        with col2:
            st.markdown(f"### 📈 Confidence")
            # Color code confidence
            if confidence >= 0.8:
                st.markdown(f"<h3 style='color: #4CAF50;'>{confidence:.2%}</h3>", unsafe_allow_html=True)
            elif confidence >= 0.5:
                st.markdown(f"<h3 style='color: #FF9800;'>{confidence:.2%}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: #F44336;'>{confidence:.2%}</h3>", unsafe_allow_html=True)
        
        # Status message
        if "healthy" in predicted_class.lower():
            st.success("🌱 Your plant appears to be healthy!")
        else:
            st.warning(f"⚠️ Potential disease detected: {predicted_class}")
        
        # Top-k predictions
        if top_predictions and len(top_predictions) > 1:
            st.markdown("---")
            st.markdown("### 🔍 Top Predictions")
            for i, pred in enumerate(top_predictions, 1):
                st.markdown(f"**{i}. {pred['class']}**")
                st.progress(pred['confidence'])
                st.caption(f"Confidence: {pred['confidence']:.2%}")
                if i < len(top_predictions):
                    st.markdown("---")

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    set_page_config()
    st.title("🌿 PlantCare AI")
    st.markdown("Upload a plant leaf image to detect potential diseases.")

    class_indices = load_class_indices(CLASS_INDICES_PATH)
    session = load_model(class_indices)

    display_sidebar(session, class_indices)
    display_background_pattern()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, top_predictions = predict(session, image, class_indices, top_k=5)
                    if predicted_class is not None:
                        display_results(predicted_class, confidence, top_predictions)
                    else:
                        st.error("Failed to make prediction. Please check the model and class indices.")

        except UnidentifiedImageError:
            st.error("Cannot identify image file. It may be corrupted or in an unsupported format.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


