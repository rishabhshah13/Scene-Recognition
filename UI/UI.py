import streamlit as st
import os
import PIL as pil
import torch
from ..Predict import predict


# load_model_button = st.button("Load Model")

# if "model_path" not in st.session_state:
#     st.session_state.model_path = ""

# model_path_input = st.text_input("Enter the path to the model file:", value=st.session_state.model_path)

# if load_model_button:
#     st.session_state.model_path = model_path_input
#     if os.path.exists(st.session_state.model_path):
#         st.success('Model loaded successfully!')
#         # Assuming the model is a Keras model saved with the '.h5' extension
#         # model = load_model(st.session_state.model_path)
#         model = torch.load(st.session_state.model_path).to('mps')
#         model.eval()
#     else:
#         st.error('Model file not found at the specified path.')


upload_model_button = st.button("Upload Model File")

if "model_file" not in st.session_state:
    st.session_state.model_file = None

if upload_model_button:
    st.session_state.model_file = st.file_uploader("Choose a model file", type=['h5', 'pth','pt'])
    if st.session_state.model_file is not None:
        st.success('Model file uploaded successfully!')
        # Save the uploaded file temporarily
        temp_model_path = os.path.join('/tmp', st.session_state.model_file.name)
        with open(temp_model_path, 'wb') as f:
            f.write(st.session_state.model_file.getvalue())
        # Load the model from the temporary path
        # model = torch.load(st.session_state.model_path).to('mps')  
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
