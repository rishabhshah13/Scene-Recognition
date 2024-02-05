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

# Add your models here
models = {'resnet18': resnet18,}

# RUN DETAILS
run_name = "jly_0131_resnet_test"
model_base = 'resnet18'
num_epochs = 10
lr = 1e-3
random_seed = 42
save_chks = range(num_epochs) # iterable of epochs for which to save the model

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()

# set up run dir 
run_dir = os.path.join('saved_models', run_name)
os.makedirs(run_dir, exist_ok = True)
log, logclose = create_logger(log_filename=os.path.join(run_dir, 'train.log'), display = False)
log(f'using device: {device}')
log(f'saving models to: {run_dir}')
log(f'using base model: {model_base}')
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
train_dataloader, val_dataloader = get_data_loader(data_dir="Data/", shuffle=True)

# define model 
model = models['resnet18']()
model.to(device)

# define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# training loop
train_loss = []
val_loss = []
# train_metrics = []
# val_metrics = []
for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    log(f'epoch {epoch}')
    #training
    model.train()
    batch_loss = []
    for i, (_data, _target) in tqdm(enumerate(train_dataloader)): 
        data = _data.to(device)
        target = _target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        batch_loss.append(loss.item())
        optimizer.step()
    train_loss.append(sum(np.array(batch_loss)/len(train_dataloader)))
    log(f'\ttrain loss: {train_loss[-1]}')
    # train_metrics.append() #TODO: add metrics
    del data 
    del target
    del pred
    del loss

    # validation
    with torch.no_grad():
        model.eval()
        batch_loss = []
        for i, (_data, _target) in tqdm(enumerate(val_dataloader)): 
            data = _data.to(device)
            target = _target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            batch_loss.append(loss.item())
        val_loss.append(sum(np.array(batch_loss)/len(val_dataloader)))
        log(f'\tval loss: {val_loss[-1]}')
        # val_metrics.append() #TODO: add metrics

    if epoch in save_chks: 
        torch.save(model.state_dict(), os.path.join(run_dir, f'{epoch}.chkpt'))

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(run_dir, 'loss'))
    del data 
    del target
    del pred
    del loss

    if device == 'mps':
        torch.mps.empty_cache()


# testing
with torch.no_grad():
    model.eval()