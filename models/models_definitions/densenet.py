import torch.nn as nn
from torchvision import models

def densenet121(): 
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 5),
                                     nn.LogSoftmax(dim=1))
    return model