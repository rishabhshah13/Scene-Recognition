import pandas
import torch

import sys
from pathlib import Path

# current_dir = Path(__file__).resolve().parent.parent
# # Append the parent directory to the Python path
# sys.path.append(str(current_dir))

# # Now you can import from the models package
# from models.Predict import predict


#Load Best models from dataframe


best_models_path = {'densenet':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/densenet/run_2024-02-12_001137/densenet_best_val_loss_4.pt', \
                    'enet_s':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/enet_s/run_2024-02-11_223300/enet_s_best_val_loss_4.pt', \
                    'resnet18':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/resnet18/run_2024-02-11_223046/resnet18_best_val_loss_3.pt', \
                    'vgg':'/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/models/saved_models/vgg/run_2024-02-11_232047/vgg_best_val_loss_1.pt'}


def load_models():

    densenet_model = torch.load(best_models_path['densenet'])
    enet_model = torch.load(best_models_path['enet_s'])
    resnet_model = torch.load(best_models_path['resnet18'])
    vgg_model = torch.load(best_models_path['vgg'])

    return densenet_model, enet_model, resnet_model, vgg_model