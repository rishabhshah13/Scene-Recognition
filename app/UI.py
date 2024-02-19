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

from app import load_models
from app import get_torch_cam

st.title('Scene Recognition App')
# st.header('Upload an Image and Predict the Scene')
st.markdown("<h3 style='text-align: left; color: White;'>Upload an Image and Predict the Scene</h2>", unsafe_allow_html=True)


model_loaded_list = {'densenet':'', \
                    'enet_s':'', \
                    'resnet18':'', \
                    'vgg':''}

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()


best_model_btn = st.button("Load Best Models")

if "best_model_btn_btn_state" not in st.session_state:
    st.session_state.best_model_btn_btn_state = None

if best_model_btn or st.session_state.best_model_btn_btn_state:
    st.session_state.best_model_btn_btn_state = True

    densenet_model, enet_model, resnet_model, vgg_model = load_models()
    model_loaded_list['densenet'] = densenet_model
    model_loaded_list['enet_s'] = enet_model
    model_loaded_list['resnet18'] = resnet_model
    model_loaded_list['vgg'] = vgg_model
    st.write("Models Loaded")


uploadbtn = st.button("Upload Image")

if "uploadbtn_state" not in st.session_state:
    st.session_state.uploadbtn_state = False

if uploadbtn or st.session_state.uploadbtn_state:
    st.session_state.uploadbtn_state = True

    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])

    if image_file is not None:
        org_image = pil.Image.open(image_file, mode='r')

        pred_button = st.button("Perform Prediction")
        if pred_button:
            st.image(org_image, caption='Uploaded Image')
            
            st.markdown("<h3 style='text-align: left; color: White;'>Image Heatmap Visualization</h2>", unsafe_allow_html=True)
            
            # st.header('Heatmap Visualisation')

            # Create a  2x2 grid layout
            col1, col2, col3, col4 = st.columns(4)
            image_col_list = [col1, col2, col3, col4]


            i = 0
            for model_name, model in model_loaded_list.items():
                predicted_class, predicted_class_name = predict(model, org_image, device)
                # st.write(f"The class predict by {model_name} is  : {predicted_class_name}")
                result = get_torch_cam(model, model_name, org_image)
                # st.image(result, caption=f'{model_name }Predicted Heat Map Image')
                image_col_list[i].image(result, caption=f'{model_name } Predicted: {predicted_class_name}')
                i += 1
