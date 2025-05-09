import streamlit as st
from app.inference import predict_xray
import os

st.title('Popcorn Lung Detector')
st.write('Upload a chest X-ray to detect popcorn lung (bronchiolitis obliterans).')

uploaded_file = st.file_uploader('Choose an X-ray image', type=['png', 'jpg'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = 'temp.png'
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Predict
    model_path = 'outputs/models/best_model.h5'
    label, prob = predict_xray(model_path, temp_path)

    # Display results
    st.image(temp_path, caption='Uploaded X-ray', use_column_width=True)
    st.write(f'**Prediction**: {label}')
    st.write(f'**Probability**: {prob:.2f}')

    # Clean up
    os.remove(temp_path)
