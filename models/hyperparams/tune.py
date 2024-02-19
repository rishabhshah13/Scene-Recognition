import os 
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn

#import your model here
from models.log import create_logger
from dataloader import get_data_loader, get_data_loader_split
from models.resnet import resnet18
from models.efficientnet import effnet_s
from models.VGG import VGG
from datetime import datetime

def test(model,test_dataloader,criterion):

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)

    test_loss =  []
    test_metric = [ ]
    batch_metric = []
    batch_loss = []
    total_imgs = 0

    with torch.no_grad():
        model.eval()
        batch_loss = []
        for i, (_data, _target) in (enumerate(test_dataloader)): 
            data = _data.to(device)
            target = _target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            batch_loss.append(loss.item())
            batch_metric.append(sum(torch.argmax(pred, dim=1)==target).item()/len(target))
            total_imgs += len(target)
        test_loss.append(sum(np.array(batch_loss)/len(test_dataloader)))
        # log(f'\tval loss: {val_loss[-1]}')
        # test_metric.append(np.mean(batch_metric)) #TODO: add metrics
        print(total_imgs)
        test_metric.append(sum(batch_metric) / total_imgs)

        print(f"Test Metric --- Test Accuracy: {sum(batch_metric)/total_imgs} ---- Val Loss: {sum(np.array(batch_loss)/len(test_dataloader))}")

    return test_loss, test_metric

