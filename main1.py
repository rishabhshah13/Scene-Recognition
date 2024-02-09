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
import argparse

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
run_dir = os.path.join('saved_models', run_name)
os.makedirs(run_dir, exist_ok = True)
log, logclose = create_logger(log_filename=os.path.join(run_dir, 'train.log'), display = False)
log(f'using device: {device}')
log(f'saving models to: {run_dir}')
log(f'using base model: {model_base}')
log(f'using batch size: {bs}')
log(f'learning rate: {lr}')
log(f'random seed: {random_seed}')


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


# dataloader
if use_split==True:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader_split(data_dir="output/",  batch_size=bs, shuffle=True)
else:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader(data_dir="Data/",  batch_size=bs, shuffle=True)

# define model 
model = models[model_base]()
model.to(device)




