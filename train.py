import os 
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn

#import your model here
from log import create_logger
from dataloader import get_data_loader, get_data_loader_split
from models.resnet import resnet18
from models.efficientnet import effnet_s
from models.VGG import VGG
from datetime import datetime




def train(model, train_dataloader, val_dataloader, num_epochs, save_checkpoints,run_dir,optimizer,criterion,model_base):
    

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)


    train_loss = []
    val_loss = []
    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        batch_loss = []
        batch_metric = []
        total_imgs =  0
        for i, (_data, _target) in enumerate(train_dataloader):
            data = _data.to(device)
            target = _target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_metric.append((pred.argmax(dim=1) == target).sum().item())
            total_imgs += len(target)
        train_loss.append(sum(batch_loss) / len(train_dataloader))
        train_metrics.append(sum(batch_metric) / total_imgs)
        print(f"Training Metric --- Train Accuracy: {sum(batch_metric)/total_imgs} ---- Train Loss: {sum(np.array(batch_loss)/len(train_dataloader))}")

        del data 
        del target
        del pred
        del loss
        
        # Validation phase
        model.eval()
        batch_metric = []
        batch_loss = []
        total_imgs =  0
        with torch.no_grad():
            for i, (_data, _target) in enumerate(val_dataloader):
                data = _data.to(device)
                target = _target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                batch_loss.append(loss.item())
                batch_metric.append((pred.argmax(dim=1) == target).sum().item())
                total_imgs += len(target)
        val_loss.append(sum(batch_loss) / len(val_dataloader))
        val_metrics.append(sum(batch_metric) / total_imgs)
        print(f"Validation Metric --- Val Accuracy: {sum(batch_metric)/total_imgs} ---- Train Loss: {sum(np.array(batch_loss)/len(val_dataloader))}")

        # Save checkpoint if required
        if epoch in save_checkpoints:
            print(f'Saving {run_dir}/{epoch}.chkpt')
            
            # Save checkpoint
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': sum(np.array(batch_loss)/len(train_dataloader)),
                    'val_loss': sum(np.array(batch_loss)/len(val_dataloader)),
                    'model': model
                    }, f'{run_dir}/{epoch}.chkpt')
            
            # Save Whole Model
            torch.save(model,f'{run_dir}/{model_base}_full_model_{epoch}.pt')
            # torch.save(model.state_dict(), f'{run_dir}/{epoch}.chkpt')
        
        if device == 'mps':
            torch.mps.empty_cache()

    return train_loss, val_loss, train_metrics, val_metrics

