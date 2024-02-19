import pandas
import torch

import sys
from pathlib import Path
import os
# current_dir = Path(__file__).resolve().parent.parent
# # Append the parent directory to the Python path
# sys.path.append(str(current_dir))

# # Now you can import from the models package
# from models.Predict import predict


#Load Best models from dataframe
import torchvision.transforms as transforms


dir_path = 'models/saved_models/best_models/'

best_model_files = [entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry)) and entry.endswith('.pt')]

print(best_model_files)


device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()


best_models_path = {'densenet':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/best_models/densenet_best_val_loss_33.pt', \
                    'enet_s':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/best_models/enet_s_best_val_loss_20.pt', \
                    'resnet18':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/best_models/resnet18_best_val_loss_3.pt', \
                    'vgg':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/best_models/vgg_best_val_loss_42.pt'}


def load_models():

    densenet_model = torch.load(best_models_path['densenet'],map_location=device )
    enet_model = torch.load(best_models_path['enet_s'],map_location=device)
    resnet_model = torch.load(best_models_path['resnet18'],map_location=device)
    vgg_model = torch.load(best_models_path['vgg'],map_location=device)

    return densenet_model, enet_model, resnet_model, vgg_model


from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp,ScoreCAM,SSCAM,XGradCAM,LayerCAM

import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask


def get_torch_cam(model,model_name,img):
    
    for param in model.parameters():
        param.requires_grad = True

    if 'densenet' in model_name:
        model_name = 'densenet'
        layer_name = 'features'
    if 'resnet18' in model_name:
        model_name = 'resnet'
        layer_name = 'layer4'
    if 'vgg' in model_name:
        model_name = 'vgg'
        layer_name = 'avgpool'
        # layer_name = model.features[30]
    if 'enet_s' in model_name:
        model_name = 'enet'
        layer_name = model.features[5]
        layer_name = model.features[6][13]

    print(model_name)
    
    preprocess = transforms.Compose([
        transforms.Resize([224,224]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,  0.456,  0.406], std=[0.229,  0.224,  0.225]),
    ])

    # Load and preprocess the image
    # org_image = pil.Image.open('/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/output/test/cathedral/gsun_1a8814d41a9e6565b1b947d8b79c4c39.jpg')
    input_tensor = preprocess(img)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model


    # img = read_image('/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/data/raw/test/campsite/gsun_1b62d632c3a49e04c7410080aad77c50.jpg')

    # Preprocess it for your chosen model
    # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # print(model)

    # with SmoothGradCAMpp(model,layer_name=layer_name) as cam_extractor:
    # with SSCAM(model,target_layer=layer_name) as cam_extractor:
    with LayerCAM(model,target_layer=layer_name) as cam_extractor:
        out = model(input_tensor.to(device).unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(input_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    return result