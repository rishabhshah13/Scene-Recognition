# import packages 
import os 
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn

#import your model here
from log import create_logger
from dataloader import get_data_loader
from models.resnet import resnet18
from models.efficientnet import effnet_s

# Add your models here
models = {'resnet18': resnet18,
         'enet_s':effnet_s,
         }

# RUN DETAILS
run_name = "jly_0207_resenet18_lr1e-2_bs=128_sgdwm08_tvt"
model_base = 'resnet18'
num_epochs = 20
bs = 128
lr = 1e-2
random_seed = 42
save_chks = [19] # iterable of epochs for which to save the model

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()

# set up run dir 
run_dir = os.path.join('/Users/JuliaYang/OneDrive - Duke University/Spring24/SceneRec/saved_models', run_name)
os.makedirs(run_dir, exist_ok = True)
log, logclose = create_logger(log_filename=os.path.join(run_dir, 'train.log'), display = False)
log(f'using device: {device}')
log(f'saving models to: {run_dir}')
log(f'using base model: {model_base}')
log(f'using batch size: {bs}')
log(f'learning rate: {lr}')
log(f'random seed: {random_seed}')

# seed randoms and make deterministic
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
# random.seed(random_seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# dataloader
train_dataloader, test_dataloader, val_dataloader  = get_data_loader(data_dir="/Users/JuliaYang/Documents/Data/",  batch_size=bs, shuffle=True)

# define model 
model = models[model_base]()
model.to(device)

# define optimizer and criterion
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# training loop
train_loss = []
val_loss = []
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
    del data 
    del target
    del pred
    del loss

    # validation
    with torch.no_grad():
        model.eval()
        total_imgs = 0
        batch_metric = []
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
    total_imgs = 0
    batch_metric = []
    for i, (_data, _target) in tqdm(enumerate(test_dataloader)): 
        data = _data.to(device)
        target = _target.to(device)
        pred = model(data)
        batch_metric.append(sum(torch.argmax(pred, dim=1)==target).item())
        total_imgs += len(target)
    log(f'\ttest accuracy: {sum(batch_metric)/total_imgs}')
