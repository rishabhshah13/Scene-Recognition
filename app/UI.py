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

model_loaded_list = {'densenet':'', \
                    'enet_s':'', \
                    'resnet18':'', \
                    'vgg':''}

# upload_model_btn = st.button("Upload Model File")

# if "upload_model_btn_state" not in st.session_state:
#     st.session_state.upload_model_btn_state = None

# if upload_model_btn or st.session_state.upload_model_btn_state:
#     st.session_state.upload_model_btn_state = True

# # if upload_model_button:
#     model_file = st.file_uploader("Choose a model file", type=['h5', 'pth','pt'])
#     # model_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])


#     if model_file is not None:
#         st.success('Model file uploaded successfully!')
#         # Save the uploaded file temporarily
#         temp_model_path = os.path.join('/tmp', model_file.name)
#         with open(temp_model_path, 'wb') as f:
#             f.write(model_file.getvalue())

#         model = torch.load(temp_model_path).to('mps')
#     else:
#         st.warning('No model file was uploaded.')


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
    print("Models Loaded")


    # densenet_model, enet_model, resnet_model, vgg_model = load_models()
    # model_loaded_list['DenseNet'] = densenet_model
    # model_loaded_list['EfficientNet'] = enet_model
    # model_loaded_list['ResidualNet'] = resnet_model
    # model_loaded_list['VGG'] = vgg_model
    # st.write("Models Loaded")
    
# # if upload_model_button:
#     model_file = st.file_uploader("Choose a model file", type=['h5', 'pth','pt'])
#     # model_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])


#     if model_file is not None:
#         st.success('Model file uploaded successfully!')
#         # Save the uploaded file temporarily
#         temp_model_path = os.path.join('/tmp', model_file.name)
#         with open(temp_model_path, 'wb') as f:
#             f.write(model_file.getvalue())

#         model = torch.load(temp_model_path).to('mps')
#     else:
#         st.warning('No model file was uploaded.')




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

            # for model in model_loaded_list:
            for model_name, model in model_loaded_list.items():

                predicted_class, predicted_class_name = predict(model,org_image,device)
                st.write(f"The class predict by {model_name} is  : {predicted_class_name}")
