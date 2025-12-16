import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import time
import pandas as pd  
import folium
from streamlit_folium import st_folium


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from library.model import BrainTumorCNN
except ImportError:
    st.error("Critical Error: 'library' folder not found. Ensure you are running this from the project root.")
    st.stop()


st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)



CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_trained_model():
    """Loads the model weights and architecture."""
    device = torch.device('cpu') 
    
    model = BrainTumorCNN(num_classes=4, dropout=0.5)
    model_path = "model.pth"
    
    if not os.path.exists(model_path):
        st.error("🔴 Error: 'model.pth' file not found. Please verify your Docker build context.")
        return None
    
    try:
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
    return transform(image).unsqueeze(0)


model = load_trained_model()



st.title("🏥 Brain Imaging Analysis Platform")
st.markdown("### AI-Assisted Diagnosis")


with st.container(border=True):
    st.markdown("### ⚠️ Legal & Ethical Warning")
    st.warning(
        """
        **This is a research prototype.** The results provided by this Deep Learningalgorithm 
        are based on statistical probabilities and do not constitute a certified 
        medical diagnosis.
        """
    )
    agreement = st.checkbox("I acknowledge that this tool is an indicative aid and does not replace a doctor.")

if not agreement:
    st.info("Please accept the conditions above to unlock the analysis interface.")
    st.stop()



st.divider()

col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("1. Scan Import")
    uploaded_file = st.file_uploader("MRI Image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Patient Scan", use_container_width=True)

with col_result:
    st.subheader("2. AI Analysis")
    
    if uploaded_file and model:
        if st.button("Start Diagnosis", type="primary", use_container_width=True):
            
            progress_text = "Image preprocessing..."
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(0, 30):
                time.sleep(0.01)
                my_bar.progress(percent_complete, text="Tensor normalization...")
            
            # Inference
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)
            
            for percent_complete in range(30, 100):
                time.sleep(0.01)
                my_bar.progress(percent_complete, text="Finalizing analysis...")
            
            my_bar.empty()
            
            pred_class_name = CLASSES[predicted_idx.item()]
            conf_score = confidence.item()
            
            if pred_class_name == "No Tumor":
                diagnosis_title = "No Tumor Detected"
                is_positive = False
            else:
                diagnosis_title = "Tumor Detected"
                is_positive = True

            with st.container(border=True):
                st.markdown("#### Analysis Result")
                
                if not is_positive:
                    st.success(f"**Diagnosis:** {diagnosis_title}")
                    st.balloons()
                    icon = "✅"
                else:
                    st.error(f"**Diagnosis:** {diagnosis_title}")
                    icon = "⚠️"
                
                c1, c2 = st.columns(2)
                c1.metric("Status", diagnosis_title)
                c2.metric("Confidence Score", f"{conf_score:.2%}", delta_color="off")
                
                st.write(f"### {icon}")
                
                if conf_score < 0.70:
                    st.warning("⚠️ Low confidence score. A thorough manual check is required.")

    elif not uploaded_file:
        st.info("⬅️ Please upload an image to start.")



st.divider()
st.subheader("3. Nearby Oncologists and hospitals")
st.markdown("Click on the **red markers** to see hospital details.")


data = [
    {
        "name": "La Paz University Hospital",
        "lat": 40.480420,
        "lon": -3.686057,
        "address": " P.º de la Castellana, 261, Fuencarral-El Pardo, 28046 Madrid"
    },
    {
        "name": "12 de Octubre University Hospital",
        "lat": 40.375397,
        "lon": -3.698072,
        "address": "Gta. Málaga, 11, Usera, 28041 Madrid"
    },
    {
        "name": "Gregorio Marañón General University Hospital",
        "lat": 40.419179,
        "lon": -3.669692,
        "address": "C. del Dr. Esquerdo, 46, Retiro, 28007 Madrid"
    },
    {
        "name": "Ramón y Cajal Hospital",
        "lat": 40.486575,
        "lon": -3.694975,
        "address": "M-607, Km. 9, 100, Fuencarral-El Pardo, 28034 Madrid"
    }
]


m = folium.Map(location=[40.416775, -3.703790], zoom_start=11)


for hospital in data:
    
    popup_content = f"""
    <div style='font-family: sans-serif; font-size: 14px;'>
        <b>{hospital['name']}</b><br>
        <i>{hospital['address']}</i>
    </div>
    """
    
    folium.Marker(
        location=[hospital["lat"], hospital["lon"]],
        popup=popup_content,           
        tooltip=hospital["name"],      
        icon=folium.Icon(color="red", icon="user-md", prefix="fa") 
    ).add_to(m)


st_folium(m, width=None, height=500, use_container_width=True)
