import torch.nn as nn
from torchvision import models

def effnet_s(): 
    # model = models.efficientnet_v2_s(pretrained=True)
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(nn.Dropout(0.2), 
                                     nn.Linear(1280, 512),
                                     nn.Linear(512, 5),
                                     nn.Sigmoid(),
                                    )
    # model.classifier = nn.Sequential(nn.Flatten(),
    #     nn.Linear(1280, 128),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(128, 5),
    #     nn.Sigmoid(),
    #     )
    return model