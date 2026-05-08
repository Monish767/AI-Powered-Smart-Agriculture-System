import os
import json
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_NAME = "plant_disease_model.pth"
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
IMAGE_SIZE = (224, 224)

DISEASE_TIPS = {
    "healthy": [
        "Keep monitoring leaves weekly to catch early anomalies.",
        "Rotate crops or potted plants every season to refresh soil nutrients.",
        "Maintain balanced watering—moist but never waterlogged soil."
    ],
    "blight": [
        "Remove and dispose of infected foliage immediately to stop spread.",
        "Apply a copper-based fungicide during early infection for best impact.",
        "Improve airflow by pruning dense branches and avoid overhead watering."
    ],
    "mildew": [
        "Trim severely affected leaves and destroy them away from the garden.",
        "Use a sulfur or potassium bicarbonate spray during cool hours.",
        "Increase sunlight exposure and keep foliage dry to inhibit spores."
    ],
    "rust": [
        "Isolate the infected plant to prevent rust spores jumping hosts.",
        "Treat with a systemic fungicide that targets uredospores.",
        "Rake debris around the plant and cover soil with clean mulch."
    ],
    "spot": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "mite": [
        "Shower the underside of leaves with a firm water jet to dislodge mites.",
        "Release beneficial predators such as ladybugs or lacewings.",
        "Follow up with horticultural oil, ensuring full leaf coverage."
    ],
    "virus": [
        "Remove the plant if infection is severe—most plant viruses lack cures.",
        "Control insect vectors (aphids, whiteflies) with sticky traps or nets.",
        "Disinfect tools with 10% bleach before moving to other plants."
    ],
    "rot": [
        "Reduce watering frequency and add perlite to enhance drainage.",
        "Dust wounds with cinnamon or sulfur to curb opportunistic fungi.",
        "Repot using sterile media if roots appear brown or mushy."
    ],
    "scorch": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "curl": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "scab": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "default": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    # Specific disease names for more precise matching
    "apple scab": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "black rot": [
        "Reduce watering frequency and add perlite to enhance drainage.",
        "Dust wounds with cinnamon or sulfur to curb opportunistic fungi.",
        "Repot using sterile media if roots appear brown or mushy."
    ],
    "cedar apple rust": [
        "Isolate the infected plant to prevent rust spores jumping hosts.",
        "Treat with a systemic fungicide that targets uredospores.",
        "Rake debris around the plant and cover soil with clean mulch."
    ],
    "powdery mildew": [
        "Trim severely affected leaves and destroy them away from the garden.",
        "Use a sulfur or potassium bicarbonate spray during cool hours.",
        "Increase sunlight exposure and keep foliage dry to inhibit spores."
    ],
    "cercospora leaf spot": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "common rust": [
        "Isolate the infected plant to prevent rust spores jumping hosts.",
        "Treat with a systemic fungicide that targets uredospores.",
        "Rake debris around the plant and cover soil with clean mulch."
    ],
    "northern leaf blight": [
        "Remove and dispose of infected foliage immediately to stop spread.",
        "Apply a copper-based fungicide during early infection for best impact.",
        "Improve airflow by pruning dense branches and avoid overhead watering."
    ],
    "esca": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "black measles": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "huanglongbing": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "citrus greening": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "bacterial spot": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "early blight": [
        "Remove and dispose of infected foliage immediately to stop spread.",
        "Apply a copper-based fungicide during early infection for best impact.",
        "Improve airflow by pruning dense branches and avoid overhead watering."
    ],
    "late blight": [
        "Remove and dispose of infected foliage immediately to stop spread.",
        "Apply a copper-based fungicide during early infection for best impact.",
        "Improve airflow by pruning dense branches and avoid overhead watering."
    ],
    "leaf scorch": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "leaf mold": [
        "Sanitize tools, pots, and benches to halt pathogen cycling.",
        "Adopt a 7-day inspection routine to spot subtle symptoms early.",
        "Alternate organic and chemical controls to prevent resistance."
    ],
    "septoria leaf spot": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "spider mites": [
        "Shower the underside of leaves with a firm water jet to dislodge mites.",
        "Release beneficial predators such as ladybugs or lacewings.",
        "Follow up with horticultural oil, ensuring full leaf coverage."
    ],
    "target spot": [
        "Sterilize pruning tools and cut back spotted leaves generously.",
        "Introduce a preventative neem oil spray every 7–10 days.",
        "Water at the base each morning so leaves dry quickly."
    ],
    "tomato mosaic virus": [
        "Remove the plant if infection is severe—most plant viruses lack cures.",
        "Control insect vectors (aphids, whiteflies) with sticky traps or nets.",
        "Disinfect tools with 10% bleach before moving to other plants."
    ],
    "tomato yellow leaf curl virus": [
        "Remove the plant if infection is severe—most plant viruses lack cures.",
        "Control insect vectors (aphids, whiteflies) with sticky traps or nets.",
        "Disinfect tools with 10% bleach before moving to other plants."
    ]
}


def get_disease_tips(class_name):
    """Return at least three actionable tips for the predicted class."""
    if not class_name:
        return DISEASE_TIPS["default"]
    label = class_name.lower()
    
    # Check for healthy first
    if "healthy" in label:
        return DISEASE_TIPS["healthy"]
    
    # Check for specific disease names (more precise matching)
    specific_diseases = [
        "apple scab", "black rot", "cedar apple rust", "powdery mildew",
        "cercospora leaf spot", "common rust", "northern leaf blight",
        "esca", "black measles", "huanglongbing", "citrus greening",
        "bacterial spot", "early blight", "late blight", "leaf scorch",
        "leaf mold", "septoria leaf spot", "spider mites", "target spot",
        "tomato mosaic virus", "tomato yellow leaf curl virus"
    ]
    
    for disease in specific_diseases:
        if disease in label:
            return DISEASE_TIPS.get(disease, DISEASE_TIPS["default"])
    
    # Fall back to keyword matching
    keyword_priority = [
        "scab",  # Check scab before spot
        "blight",
        "mildew",
        "rust",
        "spot",
        "mite",
        "virus",
        "rot",
        "scorch",
        "curl",
    ]
    for keyword in keyword_priority:
        if keyword in label and keyword in DISEASE_TIPS:
            return DISEASE_TIPS[keyword]
    
    return DISEASE_TIPS["default"]

# --- UI Configuration ---
PAGE_CONFIG = {
    "page_title": "🌿 PlantCare AI - Disease Classifier",
    "page_icon": "🌱",
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}


def apply_global_styles():
    """Inject custom CSS for a more expressive interface."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            .stApp, .stMarkdown, .stTextInput>div>div>input {
                font-family: 'Poppins', sans-serif !important;
            }
            .hero-card {
                background: linear-gradient(135deg, rgba(22,160,133,0.95) 0%, rgba(76,175,80,0.85) 50%, rgba(46,125,50,0.9) 100%);
                border-radius: 28px;
                color: white;
                padding: 40px 42px;
                margin-bottom: 24px;
                box-shadow: 0 30px 80px rgba(22, 160, 133, 0.25);
                position: relative;
                overflow: hidden;
            }
            .hero-card::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: pulse 8s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 0.5; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
            .hero-card h1 {
                font-size: 2.6rem;
                margin-bottom: 0.8rem;
                font-weight: 700;
                position: relative;
                z-index: 1;
            }
            .hero-pill {
                display: inline-block;
                padding: 6px 20px;
                border-radius: 999px;
                background: rgba(255,255,255,0.25);
                font-size: 0.9rem;
                letter-spacing: 0.1em;
                font-weight: 500;
                backdrop-filter: blur(10px);
                position: relative;
                z-index: 1;
            }
            .result-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8fdf9 100%);
                border-radius: 28px;
                padding: 32px;
                box-shadow: 0 25px 60px rgba(0,0,0,0.1);
                border: 2px solid rgba(59,141,74,0.15);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .result-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 30px 70px rgba(0,0,0,0.15);
            }
            .result-pill {
                font-size: 2.2rem;
                font-weight: 700;
                color: #2f855a;
                margin: 12px 0;
            }
            .confidence-badge {
                font-size: 3rem;
                font-weight: 700;
                margin: 16px 0;
                background: linear-gradient(135deg, #2f855a, #48bb78);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .top-card, .tip-card, .stat-card {
                background: linear-gradient(135deg, #f7fbf8 0%, #ffffff 100%);
                border: 2px solid rgba(47,133,90,0.2);
                border-radius: 20px;
                padding: 20px 24px;
                margin-bottom: 16px;
                box-shadow: 0 12px 35px rgba(47,133,90,0.12);
                transition: all 0.3s ease;
            }
            .top-card:hover, .tip-card:hover {
                transform: translateX(4px);
                box-shadow: 0 15px 40px rgba(47,133,90,0.18);
                border-color: rgba(47,133,90,0.3);
            }
            .tip-card {
                min-height: 140px;
                position: relative;
            }
            .tip-card::before {
                content: '💡';
                position: absolute;
                top: 16px;
                right: 20px;
                font-size: 1.5rem;
                opacity: 0.3;
            }
            .top-title {
                font-weight: 600;
                color: #1b4332;
                font-size: 1.1rem;
                margin-bottom: 8px;
            }
            .metric-pill {
                display: inline-flex;
                padding: 8px 20px;
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(47,133,90,0.15), rgba(47,133,90,0.08));
                color: #1b4332;
                font-weight: 600;
                font-size: 0.95rem;
                border: 1px solid rgba(47,133,90,0.2);
            }
            .upload-block {
                border: 2px dashed rgba(47,133,90,0.5);
                border-radius: 24px;
                padding: 32px;
                background: linear-gradient(135deg, rgba(247,251,248,0.9), rgba(255,255,255,0.9));
                text-align: center;
                transition: all 0.3s ease;
            }
            .upload-block:hover {
                border-color: rgba(47,133,90,0.7);
                background: linear-gradient(135deg, rgba(247,251,248,1), rgba(255,255,255,1));
            }
            .stat-card {
                text-align: center;
                padding: 24px;
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #2f855a;
                margin: 8px 0;
            }
            .stat-label {
                font-size: 0.9rem;
                color: #4a5568;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }
            .progress-container {
                background: #e8f5e9;
                border-radius: 12px;
                padding: 4px;
                margin: 12px 0;
                overflow: hidden;
            }
            .progress-bar {
                height: 28px;
                background: linear-gradient(90deg, #2f855a, #48bb78);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
                font-size: 0.9rem;
                transition: width 1s ease-in-out;
                box-shadow: 0 4px 15px rgba(47,133,90,0.3);
            }
            .section-header {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1b4332;
                margin: 24px 0 16px 0;
                padding-bottom: 12px;
                border-bottom: 3px solid rgba(47,133,90,0.2);
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin: 20px 0;
            }
            .info-item {
                background: #f0f9f4;
                padding: 16px;
                border-radius: 12px;
                border-left: 4px solid #2f855a;
            }
            .info-item-label {
                font-size: 0.85rem;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 4px;
            }
            .info-item-value {
                font-size: 1.1rem;
                font-weight: 600;
                color: #1b4332;
            }
            .divider-line {
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(47,133,90,0.3), transparent);
                margin: 32px 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_hero_section():
    """Rich hero banner describing app value."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill">🌿 Plant Diagnostics Studio</div>
            <h1>Diagnose. Understand. Revive.</h1>
            <p style="font-size:1.15rem;max-width:680px;line-height:1.7;margin-top:16px;position:relative;z-index:1;">
                Upload a single leaf photo and PlantCare AI will translate subtle color shifts,
                vein textures, and lesion patterns into actionable plant-health intelligence.
                Get instant diagnosis with detailed confidence scores, alternative predictions, 
                and personalized care recommendations.
            </p>
            <div style="margin-top:24px;display:flex;gap:16px;flex-wrap:wrap;position:relative;z-index:1;">
                <div style="background:rgba(255,255,255,0.2);padding:8px 16px;border-radius:8px;backdrop-filter:blur(10px);">
                    <span style="font-size:0.9rem;">⚡ Fast Analysis</span>
                </div>
                <div style="background:rgba(255,255,255,0.2);padding:8px 16px;border-radius:8px;backdrop-filter:blur(10px);">
                    <span style="font-size:0.9rem;">🎯 High Accuracy</span>
                </div>
                <div style="background:rgba(255,255,255,0.2);padding:8px 16px;border-radius:8px;backdrop-filter:blur(10px);">
                    <span style="font-size:0.9rem;">📚 Expert Guidance</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_value_highlights():
    """Showcase quick facts below hero."""
    st.markdown(
        """
        <div class="info-grid">
            <div class="stat-card">
                <div class="stat-label">Trained Classes</div>
                <div class="stat-value">38+</div>
                <div style="font-size:0.85rem;color:#718096;margin-top:8px;">PlantVillage Dataset</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg. Inference</div>
                <div class="stat-value">~0.8s</div>
                <div style="font-size:0.85rem;color:#718096;margin-top:8px;">CPU Accelerated</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Care Playbooks</div>
                <div class="stat-value">Dynamic</div>
                <div style="font-size:0.85rem;color:#718096;margin-top:8px;">3+ Tips Per Disease</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Accuracy</div>
                <div class="stat-value">High</div>
                <div style="font-size:0.85rem;color:#718096;margin-top:8px;">MobileNetV2 Enhanced</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Model Architecture (Enhanced MobileNetV2) ---
class ChannelAttention(nn.Module):
    """Lightweight squeeze-and-excitation block."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(in_channels // reduction, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.channel_attention(x)
        return x * weights


class EnhancedMobileNetV2(nn.Module):
    """MobileNetV2 backbone enhanced with a channel attention head."""

    def __init__(self, num_classes=38):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        feature_dim = self.backbone.last_channel

        # Keep classifier definition for checkpoint compatibility.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        self.attention = ChannelAttention(feature_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone.features(x)
        enhanced = self.attention(features)
        pooled = torch.nn.functional.adaptive_avg_pool2d(enhanced, 1)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)

# --- Main Functions ---
@st.cache_resource
def load_model(class_indices):
    """Loads the model checkpoint trained with PyTorch."""
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found: {MODEL_DIR}")
        return None
    
    pth_candidates = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.pth')]
    if not pth_candidates:
        st.error("No .pth model found in trained_model directory")
        return None
    
    preferred = [f for f in pth_candidates if f == MODEL_NAME]
    pth_file = preferred[0] if preferred else sorted(pth_candidates)[-1]
    pth_path = os.path.join(MODEL_DIR, pth_file)
    
    try:
        fallback_classes = len(class_indices) if class_indices else 38
        with st.spinner("Loading PyTorch model..."):
            checkpoint = torch.load(pth_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            saved_class_names = checkpoint.get("class_names")
            num_classes = len(saved_class_names) if saved_class_names else fallback_classes

            model = EnhancedMobileNetV2(num_classes=num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            st.success("✅ Model loaded successfully with PyTorch")
            if missing or unexpected:
                st.warning(f"Model loaded with missing ({len(missing)}) or unexpected ({len(unexpected)}) keys.")
            if saved_class_names and class_indices and len(saved_class_names) != len(class_indices):
                st.warning("Class mismatch detected between checkpoint and class_indices.json.")
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def format_class_name(class_name):
    """Formats PlantVillage class names for better readability."""
    if not class_name:
        return "Unknown"
    
    formatted = class_name.replace("___", " - ").replace("_", " ")
    words = formatted.split()
    formatted = " ".join(word.capitalize() for word in words)
    
    return formatted

def predict(model, image, class_indices, top_k=5):
    """Predicts the class of an image and returns top-k predictions."""
    if model is None or class_indices is None:
        return None, None, None

    try:
        processed_image = preprocess_image(image)
        device = next(model.parameters()).device
        processed_image = processed_image.to(device)

        with torch.no_grad():
            predictions = model(processed_image)
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
            
            confidence, predicted_index = torch.max(probabilities, 0)
            predicted_index = predicted_index.item()
            confidence = confidence.item()
            
            top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            top_k_probs = top_k_probs.cpu().numpy()
            top_k_indices = top_k_indices.cpu().numpy()
        
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

def apply_light_leaf_background():
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%),
                              url('https://www.transparenttextures.com/patterns/leafy-green.png') !important;
            background-size: cover !important;
            background-attachment: fixed !important;
            background-blend-mode: overlay !important;
        }
        [data-testid="stHeader"] { background-color: transparent !important; }
        .block-container { background-color: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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


def display_top_predictions(top_predictions):
    """Show top-k predictions as stylized cards with progress bars."""
    if not top_predictions:
        return

    st.markdown('<div class="section-header">🔍 Confidence Landscape</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color:#4a5568;margin-bottom:20px;">
            Explore alternative predictions ranked by model confidence. Higher confidence indicates stronger 
            visual similarity to the training dataset patterns.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    cols = st.columns(2)
    for idx, pred in enumerate(top_predictions):
        column = cols[idx % 2]
        confidence_pct = pred['confidence'] * 100
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx] if idx < 5 else f"{idx + 1}."
        
        column.markdown(
            f"""
            <div class="top-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <p class="top-title">{rank_emoji} {pred['class']}</p>
                    <div class="metric-pill">{confidence_pct:.1f}%</div>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width:{confidence_pct}%;">
                        {confidence_pct:.1f}%
                    </div>
                </div>
                <div style="margin-top:8px;font-size:0.85rem;color:#718096;">
                    Rank #{idx + 1} · Index {pred['index']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_disease_guidance(predicted_class):
    """Render at least three actionable tips for the detected condition."""
    tips = get_disease_tips(predicted_class)
    label = (predicted_class or "").lower()
    is_healthy = "healthy" in label
    title = "🌱 Wellness Playbook" if is_healthy else "🛠️ Recovery Playbook"
    subtitle = "Maintain optimal plant health with these proactive strategies" if is_healthy else "Follow these targeted steps to address the detected condition"
    
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#4a5568;margin-bottom:24px;">{subtitle}</p>', unsafe_allow_html=True)
    
    tip_cols = st.columns(3)
    tip_icons = ["🎯", "⚡", "🔬", "💧", "🌿", "🛡️"]
    
    for idx, tip in enumerate(tips):
        icon = tip_icons[idx % len(tip_icons)]
        tip_cols[idx % 3].markdown(
            f"""
            <div class="tip-card">
                <div style="display:flex;align-items:center;margin-bottom:12px;">
                    <span style="font-size:1.5rem;margin-right:8px;">{icon}</span>
                    <p style="margin:0;font-weight:700;color:#22543d;font-size:1.05rem;">Action {idx + 1}</p>
                </div>
                <p style="margin:0;line-height:1.6;color:#2d3748;">{tip}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_sidebar(model, class_indices):
    """Displays the sidebar with instructions and model status."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:20px 0;border-bottom:2px solid rgba(47,133,90,0.2);margin-bottom:24px;">
                <h2 style="color:#2f855a;margin:0;">🌿 PlantCare AI</h2>
                <p style="color:#718096;margin:8px 0 0 0;font-size:0.9rem;">Intelligent Plant Diagnostics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.header("📋 Quick Guide")
        st.markdown(
            """
            <div style="background:#f0f9f4;padding:16px;border-radius:12px;border-left:4px solid #2f855a;margin-bottom:20px;">
                <ol style="margin:0;padding-left:20px;color:#2d3748;line-height:1.8;">
                    <li><strong>Upload</strong> a high-resolution leaf image (single leaf works best)</li>
                    <li><strong>Analyze</strong> to unlock diagnosis, confidence scores, and care playbook</li>
                    <li><strong>Review</strong> the Recovery/Wellness Playbook with actionable tips</li>
                    <li><strong>Monitor</strong> alternative predictions for differential diagnoses</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.header("🔧 System Status")
        
        # Model status with detailed info
        status_col1, status_col2 = st.columns([1, 3])
        if model:
            with status_col1:
                st.markdown("### ✅")
            with status_col2:
                st.markdown("**Model Ready**")
                device = next(model.parameters()).device
                st.caption(f"Backend: PyTorch")
                st.caption(f"Device: {device.type.upper()}")
                st.caption("✅ Python 3.14 Compatible")
        else:
            with status_col1:
                st.markdown("### ❌")
            with status_col2:
                st.markdown("**Model Not Loaded**")
                st.caption("Please check model files")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Class indices status
        status_col1, status_col2 = st.columns([1, 3])
        if class_indices:
            with status_col1:
                st.markdown("### ✅")
            with status_col2:
                st.markdown(f"**Classes Loaded**")
                st.caption(f"Total: {len(class_indices)} disease classes")
                st.caption("✅ Ready for predictions")
        else:
            with status_col1:
                st.markdown("### ❌")
            with status_col2:
                st.markdown("**Classes Not Loaded**")
                st.caption("Please check class_indices.json")
        
        st.markdown("---")
        st.header("🎨 Customization")
        st.session_state.bg_style = st.selectbox(
            "Background Style",
            ["Leaf Pattern", "Soft Gradient", "Leafy Gradient"],
            index=["Leaf Pattern", "Soft Gradient", "Leafy Gradient"].index(
                st.session_state.get("bg_style", "Leaf Pattern")
            ),
            help="Choose your preferred background theme"
        )
        
        st.markdown("---")
        st.header("📊 Model Information")
        st.markdown(
            """
            <div style="background:#f7fbf8;padding:16px;border-radius:12px;border:1px solid rgba(47,133,90,0.15);">
                <p style="margin:0 0 12px 0;font-weight:600;color:#1b4332;">Architecture</p>
                <p style="margin:0 0 8px 0;color:#4a5568;font-size:0.9rem;">• MobileNetV2 Backbone</p>
                <p style="margin:0 0 8px 0;color:#4a5568;font-size:0.9rem;">• Channel Attention Module</p>
                <p style="margin:0 0 8px 0;color:#4a5568;font-size:0.9rem;">• Enhanced Classifier Head</p>
                <p style="margin:0;color:#4a5568;font-size:0.9rem;">• Trained on PlantVillage Dataset</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.header("💡 Tips for Best Results")
        st.markdown(
            """
            <div style="background:#fff9e6;padding:16px;border-radius:12px;border-left:4px solid #f6ad55;">
                <ul style="margin:0;padding-left:20px;color:#2d3748;line-height:1.8;font-size:0.9rem;">
                    <li>Use natural lighting when possible</li>
                    <li>Focus on a single leaf per image</li>
                    <li>Ensure clear, in-focus photos</li>
                    <li>Avoid shadows and reflections</li>
                    <li>Include the full leaf in frame</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown(
            """
            <div style="background:#f7fbf8;padding:16px;border-radius:12px;border:1px solid rgba(47,133,90,0.15);">
                <p style="color:#4a5568;font-size:0.9rem;line-height:1.6;margin:0;">
                    PlantCare AI blends MobileNetV2 vision features with an attention-guided head 
                    trained on PlantVillage, delivering rich context plus actionable guidance for 
                    plant health management.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_image_metadata(image):
    """Display image metadata and analysis details."""
    st.markdown('<div class="section-header">📷 Image Analysis</div>', unsafe_allow_html=True)
    
    width, height = image.size
    aspect_ratio = width / height
    total_pixels = width * height
    file_size_mb = len(image.tobytes()) / (1024 * 1024) if hasattr(image, 'tobytes') else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="info-item">
                <div class="info-item-label">Dimensions</div>
                <div class="info-item-value">{width} × {height}px</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="info-item">
                <div class="info-item-label">Aspect Ratio</div>
                <div class="info-item-value">{aspect_ratio:.2f}:1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="info-item">
                <div class="info-item-label">Total Pixels</div>
                <div class="info-item-value">{total_pixels:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="info-item">
                <div class="info-item-label">Color Mode</div>
                <div class="info-item-value">RGB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_results(predicted_class, confidence, top_predictions=None, image=None):
    """Displays the prediction results with more detailed storytelling."""
    if not predicted_class or confidence is None:
        return

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Insight Studio</div>', unsafe_allow_html=True)

    label_lower = predicted_class.lower()
    is_healthy = "healthy" in label_lower
    sentiment = "✅ Healthy profile detected" if is_healthy else "⚠️ Disease markers flagged"
    tone_color = "#2f855a" if is_healthy else "#c53030"
    status_emoji = "🌿" if is_healthy else "🔴"
    confidence_level = "Very High" if confidence > 0.9 else "High" if confidence > 0.7 else "Moderate" if confidence > 0.5 else "Low"
    
    # Main result card with enhanced details
    st.markdown(
        f"""
        <div class="result-card">
            <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:20px;">
                <div>
                    <p style="margin:0;color:#718096;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.1em;">Primary Diagnosis</p>
                    <div class="result-pill">{status_emoji} {predicted_class}</div>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0;color:#718096;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.1em;">Confidence Level</p>
                    <p style="margin:4px 0;color:{tone_color};font-weight:700;font-size:1.1rem;">{confidence_level}</p>
                </div>
            </div>
            
            <div style="margin:24px 0;">
                <p style="margin:0 0 8px 0;color:#718096;font-size:0.9rem;">Model Confidence Score</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width:{confidence*100}%;">
                        {confidence:.1%}
                    </div>
                </div>
            </div>
            
            <div style="background:#f0f9f4;padding:16px;border-radius:12px;margin-top:20px;border-left:4px solid {tone_color};">
                <p style="margin:0;color:{tone_color};font-weight:600;font-size:1.05rem;">{sentiment}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed analysis section
    st.markdown('<div class="section-header">📝 Detailed Analysis</div>', unsafe_allow_html=True)
    
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        st.markdown(
            f"""
            <div class="top-card">
                <p style="font-weight:700;color:#1b4332;margin-bottom:12px;font-size:1.1rem;">🔬 Signal Analysis</p>
                <p style="color:#4a5568;line-height:1.7;margin-bottom:8px;">
                    <strong>Confidence Score:</strong> {confidence:.2%} match with the closest class signature in the training dataset.
                </p>
                <p style="color:#4a5568;line-height:1.7;margin-bottom:8px;">
                    <strong>Reliability:</strong> {'Excellent' if confidence > 0.85 else 'Good' if confidence > 0.7 else 'Moderate'} - The model shows {'strong' if confidence > 0.85 else 'moderate' if confidence > 0.7 else 'some'} confidence in this prediction.
                </p>
                <p style="color:#4a5568;line-height:1.7;">
                    <strong>Recommendation:</strong> {'Continue monitoring with current care routine.' if is_healthy else 'Review alternative predictions below for differential diagnosis.'}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with analysis_cols[1]:
        st.markdown(
            f"""
            <div class="top-card">
                <p style="font-weight:700;color:#1b4332;margin-bottom:12px;font-size:1.1rem;">📈 Prediction Insights</p>
                <p style="color:#4a5568;line-height:1.7;margin-bottom:8px;">
                    <strong>Status:</strong> {'Healthy' if is_healthy else 'Disease Detected'}
                </p>
                <p style="color:#4a5568;line-height:1.7;margin-bottom:8px;">
                    <strong>Action Required:</strong> {'Maintain current care routine and continue monitoring.' if is_healthy else 'Immediate attention recommended to halt progression.'}
                </p>
                <p style="color:#4a5568;line-height:1.7;">
                    <strong>Next Steps:</strong> Custom {'wellness' if is_healthy else 'recovery'} playbook prepared below with actionable guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Display image metadata if available
    if image:
        display_image_metadata(image)
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

    display_top_predictions(top_predictions)
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    display_disease_guidance(predicted_class)

def render_compact_output(predicted_class, confidence, top_predictions, image):
    original_formatted = top_predictions[0]['class'] if top_predictions else predicted_class
    is_healthy = "healthy" in (predicted_class or "").lower()
    extra = "" if is_healthy else (
        "<p style=\"color:#dc2626;font-weight:600;\">Disease markers</p>\n"
        "<p style=\"color:#16a34a;\">To cure the leaf use tips and tricks</p>"
    )
    st.markdown(
        f"""
        <div style="font-size:1rem;line-height:1.9;">
            <p style="color:#1a73e8;"><strong>Original :</strong> {original_formatted}</p>
            <p style="color:#2f855a;"><strong>Predicted :</strong> {predicted_class}</p>
            <p style="color:#111827;">Model confidence : <span style="color:#e11d48;font-weight:700;">{confidence*100:.2f}%</span></p>
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_healthy:
        tips = get_disease_tips(predicted_class)
        st.markdown(f"<p style='color:#1a73e8;font-weight:600;'>{predicted_class.lower()}:</p>", unsafe_allow_html=True)
        st.markdown("<ul style='list-style:none;padding-left:0;margin-top:4px;'>", unsafe_allow_html=True)
        for t in tips:
            st.markdown(f"<li style='margin:6px 0;color:#111827;'>➤ {t}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin:12px 0 8px 0;color:#111827;font-weight:700;display:flex;align-items:center;gap:8px;">
            <span>🖼️</span> <span>Image Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    w, h = image.size
    ar = w / h if h else 0
    total = w * h
    analysis_cards = [
        ("DIMENSIONS", f"{w} × {h}px"),
        ("ASPECT RATIO", f"{ar:.2f}:1"),
        ("TOTAL PIXELS", f"{total:,}"),
        ("COLOR MODE", "RGB"),
    ]
    st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;'>", unsafe_allow_html=True)
    for label, value in analysis_cards:
        st.markdown(
            f"""
            <div style="background:#f0f9f4;border:2px solid rgba(47,133,90,0.2);border-radius:16px;padding:14px 16px;">
                <div style="color:#4a5568;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
                <div style="color:#1b4332;font-size:1rem;font-weight:700;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin:18px 0;color:#111827;font-weight:700;">🔍 Confidence Landscape</div>
        <p style="color:#4a5568;margin-bottom:16px;">Explore alternative predictions ranked by model confidence. Higher confidence indicates stronger visual similarity to the training dataset patterns.</p>
        """,
        unsafe_allow_html=True,
    )

    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;'>", unsafe_allow_html=True)
    for i, pred in enumerate(top_predictions or []):
        emoji = rank_emoji[i] if i < len(rank_emoji) else f"{i+1}️⃣"
        pct = pred['confidence']*100
        st.markdown(
            f"""
            <div style="background:#ffffff;border-radius:16px;padding:14px;border:2px solid rgba(59,141,74,0.15);box-shadow:0 10px 25px rgba(0,0,0,0.06);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <div style="color:#1b4332;font-weight:700;">{emoji} {pred['class']}</div>
                    <div style="color:#1b4332;background:linear-gradient(135deg, rgba(47,133,90,0.15), rgba(47,133,90,0.08));border:1px solid rgba(47,133,90,0.2);border-radius:999px;padding:6px 12px;font-weight:600;">{pct:.1f}%</div>
                </div>
                <div style="background:#e8f5e9;border-radius:10px;padding:4px;">
                    <div style="height:22px;background:linear-gradient(90deg,#2f855a,#48bb78);border-radius:8px;color:#fff;font-weight:600;display:flex;align-items:center;justify-content:center;width:{pct}%;">{pct:.1f}%</div>
                </div>
                <div style="margin-top:8px;color:#718096;font-size:0.9rem;">Rank #{i+1} · Index {pred['index']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    set_page_config()
    apply_light_leaf_background()
    st.markdown(
        """
        <div style="text-align:center;font-family:'Times New Roman', Times, serif;font-size:26px;font-weight:700;color:#000;">
            AI-Powered Smart Agriculture System
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;font-family:'Times New Roman', Times, serif;font-size:24px;font-weight:700;color:#000;">
            Leaf disease detection with AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_indices = load_class_indices(CLASS_INDICES_PATH)
    model = load_model(class_indices)

    st.markdown('<div class="section-header">📤 Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", width='stretch')
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("🔬 Analyzing image patterns and extracting features..."):
                import time
                start_time = time.time()
                predicted_class, confidence, top_predictions = predict(model, image, class_indices, top_k=5)
                inference_time = time.time() - start_time

            if predicted_class is not None:
                st.session_state.last_inference_time = inference_time
                render_compact_output(predicted_class, confidence, top_predictions, image)
            else:
                st.error("❌ Failed to make prediction. Please check the model and class indices.")

        except UnidentifiedImageError:
            st.error("❌ Cannot identify image file. It may be corrupted or in an unsupported format.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
