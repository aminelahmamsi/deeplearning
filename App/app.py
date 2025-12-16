import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import time

# Ajoute le dossier courant au path pour que Python trouve le dossier 'library'
# Cela suppose que le dossier 'library' est au même niveau ou copié au même endroit par Docker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import de VOTRE architecture de modèle (le vrai DNN)
try:
    from library.model import BrainTumorCNN
except ImportError:
    # Fallback pour éviter que l'app crash si le dossier library est mal placé en local
    st.error("Critical Error: 'library' folder not found. Ensure you are running this from the project root.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# --- 2. Utility Functions ---

# Liste des classes (Ordre alphabétique standard ImageFolder)
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_trained_model():
    """Loads the model weights and architecture."""
    device = torch.device('cpu') # Utilisation du CPU pour compatibilité maximale
    
    # Instanciation de l'architecture
    model = BrainTumorCNN(num_classes=4, dropout=0.5)
    
    # Chemin du fichier de poids
    model_path = "model.pth"
    
    if not os.path.exists(model_path):
        st.error("🔴 Error: 'model.pth' file not found. Please verify your Docker build context.")
        return None
    
    try:
        # Chargement des poids réels
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Critical error loading the model: {e}")
        return None

def preprocess_image(image):
    """Real preprocessing matching your training data."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Ajout dimension batch (1, 3, 224, 224)

# --- 3. Silent Model Loading ---
# Le modèle est chargé en arrière-plan sans afficher de message "Loaded"
model = load_trained_model()

# --- 4. Main Interface ---

st.title("🏥 Brain Imaging Analysis Platform test 1")
st.markdown("### AI-Assisted Diagnosis")

# --- DISCLAIMER ZONE (MANDATORY) ---
with st.container(border=True):
    st.markdown("### ⚠️ Legal & Ethical Warning")
    st.warning(
        """
        **This is a research prototype.** The results provided by this algorithm 
        are based on statistical probabilities and do not constitute a certified 
        medical diagnosis.
        
        Any medical decision must be made by a qualified healthcare professional 
        after a complete analysis of the patient's file.
        """
    )
    # Checkbox bloquante
    agreement = st.checkbox("I acknowledge that this tool is an indicative aid and does not replace a doctor.")

# Arrêt si pas coché
if not agreement:
    st.info(" Please accept the conditions above to unlock the analysis interface.")
    st.stop()

# --- ANALYSIS ZONE ---

st.divider()

col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("1. Scan Import")
    uploaded_file = st.file_uploader("DICOM File / Image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Affichage propre sans warning
        st.image(image, caption="Patient Scan", use_container_width=True)

with col_result:
    st.subheader("2. AI Analysis")
    
    # On vérifie que le fichier est là et que le modèle est chargé
    if uploaded_file and model:
        if st.button("Start Diagnosis", type="primary", use_container_width=True):
            
            # Barre de progression pour l'UX
            progress_text = "Image preprocessing..."
            my_bar = st.progress(0, text=progress_text)
            
            # Simulation visuelle rapide
            for percent_complete in range(0, 30):
                time.sleep(0.01)
                my_bar.progress(percent_complete, text="Tensor normalization...")
            
            # --- VRAIE INFÉRENCE ---
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)
            
            # Fin de la barre
            for percent_complete in range(30, 100):
                time.sleep(0.01)
                my_bar.progress(percent_complete, text="Finalizing analysis...")
            
            my_bar.empty()
            
            # Récupération résultats
            pred_class_name = CLASSES[predicted_idx.item()]
            conf_score = confidence.item()
            
            # Logique d'affichage (Oui/Non)
            if pred_class_name == "No Tumor":
                diagnosis_title = "No Tumor Detected"
                is_positive = False
            else:
                diagnosis_title = "Tumor Detected" # On cache le type précis (Glioma, etc.)
                is_positive = True

            # Carte de résultat
            with st.container(border=True):
                st.markdown("#### Analysis Result")
                
                if not is_positive:
                    # CAS NÉGATIF (Pas de tumeur)
                    st.success(f"**Diagnosis:** {diagnosis_title}")
                    st.balloons() # Les ballons sont là !
                    icon = "✅"
                else:
                    # CAS POSITIF (Tumeur)
                    st.error(f"**Diagnosis:** {diagnosis_title}")
                    icon = "⚠️"
                
                # Métriques
                c1, c2 = st.columns(2)
                c1.metric("Status", diagnosis_title)
                c2.metric("Confidence Score", f"{conf_score:.2%}", delta_color="off")
                
                st.write(f"### {icon}")
                
                if conf_score < 0.70:
                    st.warning("⚠️ Low confidence score. A thorough manual check is required.")

    elif not uploaded_file:
        st.info("⬅️ Please upload an image to start.")