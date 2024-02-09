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




def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, save_checkpoints):
    

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


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

        # Save checkpoint if required
        if epoch in save_checkpoints:
            torch.save(model.state_dict(), f'{epoch}.chkpt')

    return train_loss, val_loss, train_metrics, val_metrics




# define model 
model = models[model_base]()
model.to(device)

# define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# training loop
train_loss = []
val_loss = []
test_loss = []
train_metrics = []
val_metrics = []
for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    log(f'epoch {epoch}')
    #training
    model.train()
    batch_loss = []
    batch_metric = []
    total_imgs = 0
    for i, (_data, _target) in tqdm(enumerate(train_dataloader)): 
        data = _data.to(device)
        target = _target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_metric.append(sum(torch.argmax(pred, dim=1)==target).item())
        total_imgs += len(target)
    train_loss.append(sum(np.array(batch_loss)/len(train_dataloader)))
    log(f'\ttrain loss: {train_loss[-1]}')
    train_metrics.append(sum(batch_metric)/total_imgs) #TODO: add metrics
    print(f"Training Metric --- Train Accuracy: {sum(batch_metric)/total_imgs} ---- Train Loss: {sum(np.array(batch_loss)/len(train_dataloader))}")
    del data 
    del target
    del pred
    del loss

    # validation
    total_imgs = 0
    batch_metric = []
    with torch.no_grad():
        model.eval()
        batch_loss = []
        for i, (_data, _target) in tqdm(enumerate(val_dataloader)): 
            data = _data.to(device)
            target = _target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            batch_loss.append(loss.item())
            batch_metric.append(sum(torch.argmax(pred, dim=1)==target).item())
            total_imgs += len(target)
        val_loss.append(sum(np.array(batch_loss)/len(val_dataloader)))
        log(f'\tval loss: {val_loss[-1]}')
        val_metrics.append(sum(batch_metric)/total_imgs) #TODO: add metrics
        print(f"Validation Metric --- Val Accuracy: {sum(batch_metric)/total_imgs} ---- Val Loss: {sum(np.array(batch_loss)/len(val_dataloader))}")


    if epoch in save_chks: 
        torch.save(model.state_dict(), os.path.join(run_dir, f'{epoch}.chkpt'))

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss'))
    plt.close()
    plt.plot(train_metrics, label='train accuracy')
    plt.plot(val_metrics, label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'accu'))
    plt.close()
    del data 
    del target
    del pred
    del loss

    if device == 'mps':
        torch.mps.empty_cache()


# testing
with torch.no_grad():
    model.eval()