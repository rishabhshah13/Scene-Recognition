import os

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataloader import get_data_loader, get_data_loader_split

from models.resnet import resnet18
from models.efficientnet import effnet_s
from models.VGG import VGG

# Add your models here
models = {'resnet18': resnet18,
         'enet_s':effnet_s,
         'vgg':VGG
         }

def create_cm(model, save_path, dataloader): 
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)

    preds = []
    labels = []
    with torch.no_grad():
        for _data, _label in dataloader: 
            data = _data.to(device)
            label = _label.to(device)
            preds += model(data).argmax(dim=1).tolist()
            labels += label.tolist()
    try: 
        class_to_idx = dataloader.dataset.class_to_idx
    except: 
        class_to_idx = dataloader.dataset.dataset.class_to_idx
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels = class_to_idx,
                                 )
    disp.plot()
    plt.savefig(save_path+'_CM.png')

def get_tests(model, save_path, dataloader):
    model.eval()
    create_cm(model, save_path, dataloader)
