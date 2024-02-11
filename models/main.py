import os 
# from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn

#import your model here
from log import create_logger
from dataloader.dataloader import get_data_loader, get_data_loader_split
from models_definitions.resnet import resnet18
from models_definitions.efficientnet import effnet_s
from models_definitions.VGG import VGG
from datetime import datetime
import argparse
from train import train
from test import test

# seed randoms and make deterministic
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H%M%S")
run_name = f"run_{timestamp}"

parser = argparse.ArgumentParser(description='Training script for scene recognition.')

# Add the arguments
parser.add_argument('--model_base', type=str, default='enet_s', help='Base model to use (default: enet_s).')
parser.add_argument('--num_epochs', type=int, default=250, help='Number of training epochs (default:  250).')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading (default:  64).')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for the optimizer (default:  1e-6).')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility (default:  42).')
parser.add_argument('--use_split', action='store_true', help='Use split dataset (default: False).')
parser.add_argument('--save_checkpoints', type=lambda s: [int(item) for item in s.split(',')], default=[], help='Epochs at which to save checkpoints (comma-separated values).')

args = parser.parse_args()
# Set the variables based on the arguments
model_base = args.model_base
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
random_seed = args.random_seed
use_split = args.use_split
save_checkpoints = args.save_checkpoints

# Add your models here
models = {'resnet18': resnet18,
         'enet_s':effnet_s,
         'vgg':VGG
         }

save_chks = range(num_epochs) # iterable of epochs for which to save the model
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()

# set up run dir 
run_dir = os.path.join('models/saved_models', run_name)
os.makedirs(run_dir, exist_ok = True)
log, logclose = create_logger(log_filename=os.path.join(run_dir, 'train.log'), display = False)
log(f'using device: {device}')
log(f'saving models to: {run_dir}')
log(f'using base model: {model_base}')
log(f'using batch size: {batch_size}')
log(f'learning rate: {learning_rate}')
log(f'random seed: {random_seed}')


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


# dataloader
if use_split==True:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader_split(data_dir="data/raw/",  batch_size=batch_size, shuffle=True)
else:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader(data_dir="Data/",  batch_size=batch_size, shuffle=True)

# define model 
model = models[model_base]()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


train_loss, val_loss, train_metrics, val_metrics = train(model, train_dataloader, val_dataloader, num_epochs, save_checkpoints,run_dir,optimizer,criterion,model_base)


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


print("NOW WE WILL TEST!")

test_loss, test_metric = test(model,test_dataloader,criterion)

# plt.plot(test_loss, label='test')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.savefig(os.path.join(run_dir, 'loss'))
# plt.close()
# plt.plot(test_metric, label='train accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.savefig(os.path.join(run_dir, 'test accu'))
# plt.close()
