## Includes albumentations

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class Transform():
  def __init__(self,transform):
    self.transform=transform
  def __call__(self,image):
    # print(self.transform(image=image)["image"].shape)
    return self.transform(image=image)["image"]

def open_img(img_path):
    # print(img_path)
    img=Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    return np.array(img)


def get_data_loader(data_dir, batch_size=256, shuffle=True, train_split=0.70):
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
    # print(len(full_dataset))
    # Calculate sizes of train and test sets
    train_size = int(train_split * len(full_dataset))
    test_size = int((len(full_dataset) - train_size)/2)
    val_size = len(full_dataset) - train_size - test_size

    # Split dataset into train and test sets
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size])


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
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                      batch_size=batch_size, 
                                                      shuffle=shuffle, 
                                                    #   num_workers=4,
                                                    #   pin_memory=True,
                                                      )

    return train_dataset_loader, test_dataset_loader, val_dataset_loader



def get_data_loader_split(data_dir, batch_size=256, shuffle=True,use_albumentations=True):
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


    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),  # Resizing the image as the VGG only takes 224 x 224 as input size
        transforms.ToTensor()
    ])


    albumentations_transform = A.Compose(
      [
          A.SmallestMaxSize(max_size=160),
          A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
          A.RandomCrop(height=128, width=128),
          A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
          A.RandomBrightnessContrast(p=0.5),
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(),
      ]
    )
    
    # Load dataset
    if use_albumentations==True:
      train_dataset = torchvision.datasets.ImageFolder(root= data_dir + '/train/', transform=Transform(albumentations_transform),loader=open_img)
      test_dataset = torchvision.datasets.ImageFolder(root=data_dir + '/test/',transform=val_transform)
      val_dataset = torchvision.datasets.ImageFolder(root=data_dir + '/val/',transform=val_transform)
    else:
      train_dataset = torchvision.datasets.ImageFolder(root= data_dir + '/train/', transform=transform)
      test_dataset = torchvision.datasets.ImageFolder(root=data_dir + '/test/', transform=val_transform)
      val_dataset = torchvision.datasets.ImageFolder(root=data_dir + '/val/', transform=val_transform)


    
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
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                      batch_size=batch_size, 
                                                      shuffle=shuffle, 
                                                    #   num_workers=4,
                                                    #   pin_memory=True,
                                                      )

    return train_dataset_loader, test_dataset_loader, val_dataset_loader