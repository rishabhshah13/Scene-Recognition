import streamlit as st
import os
import PIL as pil
import torch
# from ..models.Predict import predict
# from SceneRecognition.models.Predict import predict

import sys
from pathlib import Path

# Get the absolute path of the current file
current_dir = Path(__file__).resolve().parent.parent
# Append the parent directory to the Python path
sys.path.append(str(current_dir))

# Now you can import from the models package
from models.Predict import predict





upload_model_btn = st.button("Upload Model File")

if "upload_model_btn_state" not in st.session_state:
    st.session_state.upload_model_btn_state = None

if upload_model_btn or st.session_state.upload_model_btn_state:
    st.session_state.upload_model_btn_state = True

# if upload_model_button:
    model_file = st.file_uploader("Choose a model file", type=['h5', 'pth','pt'])
    # model_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])


    if model_file is not None:
        st.success('Model file uploaded successfully!')
        # Save the uploaded file temporarily
        temp_model_path = os.path.join('/tmp', model_file.name)
        with open(temp_model_path, 'wb') as f:
            f.write(model_file.getvalue())

        model = torch.load(temp_model_path).to('mps')
    else:
        st.warning('No model file was uploaded.')




uploadbtn = st.button("Upload Image")

if "uploadbtn_state" not in st.session_state:
    st.session_state.uploadbtn_state = False

if uploadbtn or st.session_state.uploadbtn_state:
    st.session_state.uploadbtn_state = True

    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])

    if image_file is not None:
        org_image = pil.Image.open(image_file, mode='r')
        st.text("Uploaded image")
        st.image(org_image, caption='Image for Prediction')
        pred_button = st.button("Perform Prediction")
        if pred_button:
            st.image(org_image, caption='Predicted Image')
            device = 'mps'

            predicted_class, predicted_class_name = predict(model,org_image,device)
            st.write(f"The class is : {predicted_class_name}")
