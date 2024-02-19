import torch
import os

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp,ScoreCAM,SSCAM,XGradCAM,LayerCAM

from torchcam.utils import overlay_mask

import torchvision.transforms as transforms


dir_path = 'models/saved_models/best_models/'

best_model_files = [entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry)) and entry.endswith('.pt')]

print(best_model_files)


device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()


best_models_path = {'densenet':'models/saved_models/best_models/densenet_best.pt', \
                    'enet_s':'models/saved_models/best_models/enet_s_best.pt', \
                    'resnet18':'models/saved_models/best_models/resnet18_best.pt', \
                    'vgg':'models/saved_models/best_models/vgg_best.pt'}


def load_models():
    try:
        densenet_model = torch.load(best_models_path['densenet'], map_location=device)
    except Exception as e:
        print(f"Error loading DenseNet model: {e}")
        return False

    try:
        enet_model = torch.load(best_models_path['enet_s'], map_location=device)
    except Exception as e:
        print(f"Error loading ENet model: {e}")
        return False

    try:
        resnet_model = torch.load(best_models_path['resnet18'], map_location=device)
    except Exception as e:
        print(f"Error loading ResNet model: {e}")
        return False

    try:
        vgg_model = torch.load(best_models_path['vgg'], map_location=device)
    except Exception as e:
        print(f"Error loading VGG model: {e}")
        return False

    return densenet_model, enet_model, resnet_model, vgg_model



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
        transforms.ToTensor()
    ])

    # Load and preprocess the image
    input_tensor = preprocess(img)
 

    with LayerCAM(model,target_layer=layer_name) as cam_extractor:
        out = model(input_tensor.to(device).unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(input_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    return result