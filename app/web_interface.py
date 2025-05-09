import streamlit as st
from PIL import Image
import numpy as np
from inference import PopcornLungPredictor
import os

# Set up the app
st.set_page_config(page_title="Popcorn Lung X-ray Classifier", layout="wide")

# Title and description
st.title("Popcorn Lung Detection from Chest X-rays")
st.write("""
This application uses a deep learning model to detect signs of popcorn lung (bronchiolitis obliterans) 
from chest X-ray images. Upload a chest X-ray image to get a prediction.
""")

# Initialize predictor
@st.cache_resource
def load_model():
    return PopcornLungPredictor()

predictor = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing the X-ray...'):
        result = predictor.predict(temp_path)
    
    # Display results
    st.subheader("Prediction Results")
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Probability", f"{result['probability']:.4f}")
            st.metric("Diagnosis", result['diagnosis'])
        
        with col2:
            # Visualize confidence
            confidence = result['confidence']
            st.progress(int(confidence * 100))
            st.write(f"Confidence: {confidence:.2%}")
            
            # Interpretation
            if result['probability'] > 0.5:
                st.warning("This X-ray shows characteristics consistent with popcorn lung. Please consult a pulmonologist for further evaluation.")
            else:
                st.success("No significant signs of popcorn lung detected in this X-ray.")
    
    # Remove temporary file
    os.remove(temp_path)

# Add some info about popcorn lung
st.sidebar.title("About Popcorn Lung")
st.sidebar.info("""
**Popcorn lung** (bronchiolitis obliterans) is a rare condition that damages the small airways in the lungs. 
It can be caused by exposure to certain chemicals, including diacetyl used in flavorings.

**Common symptoms:**
- Cough
- Shortness of breath
- Wheezing
- Fatigue

This tool is for preliminary screening only and should not replace professional medical diagnosis.
""")
