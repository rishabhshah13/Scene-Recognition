import torch
import torch.nn as nn
from torchvision import models


def VGG():
    
    model = models.vgg16(pretrained=True)  # Choose appropriate model (vgg16, vgg19, etc.)
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze convolutional layers
    num_ftrs = model.classifier[6].in_features  # Get number of features from pre-trained model
    # model.classifier[6] = nn.Linear(num_ftrs, 5)  # Replace with your classifier head
    model.classifier[6] = nn.Sequential(nn.Dropout(0.2), 
                                     nn.Linear(num_ftrs, 5),
                                     nn.Sigmoid(),
                                    ) 

    # optimizer = torch.optim.Adam(model.classifier[6].parameters())  # Optimize only new classifier layer
    # loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
    
    return model
