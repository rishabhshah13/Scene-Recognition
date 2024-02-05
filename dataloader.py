import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def get_data_loader(data_dir, batch_size=256, shuffle=True, train_split=0.8):
    """
    Define the way we compose the batch dataset including the augmentation for increasing the number of data
    and return the augmented batch-dataset
    :param data_dir: root directory where the dataset is
    :param batch_size: size of the batch
    :param train: true if current phase is training, else false
    :param train_split: percentage of data to be used for training
    :return: augmented batch dataset
    """

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize([224, 224]),  # Resizing the image as the VGG only takes 224 x 224 as input size
        transforms.RandomHorizontalFlip(),  # Flip the data horizontally
        # TODO: Add random crop if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # Calculate sizes of train and test sets
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


    # Create data loader
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                       batch_size=batch_size, 
                                                       shuffle=shuffle, 
                                                    #    num_workers=4,
                                                    #    pin_memory=True,
                                                       )
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                      batch_size=batch_size, 
                                                      shuffle=shuffle, 
                                                    #   num_workers=4,
                                                    #   pin_memory=True,
                                                      )

    return train_dataset_loader, test_dataset_loader