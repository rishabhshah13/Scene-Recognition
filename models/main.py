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
from models_definitions.densenet import densenet121
from metric_logging import get_df, add_to_database
from datetime import datetime
import argparse
from train import train
from test import test, create_cm

# seed randoms and make deterministic
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
import shutil

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

    
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
parser.add_argument('--use_albumentations', default=False, type=boolean_string, help='Use albumentations for data transformations')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer: sgd or adam')


args = parser.parse_args()
# Set the variables based on the arguments
model_base = args.model_base
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
random_seed = args.random_seed
use_split = args.use_split
save_checkpoints = args.save_checkpoints
use_albumentations = args.use_albumentations
opt = args.opt

# Add your models here
models = {'resnet18': resnet18,
         'enet_s':effnet_s,
         'vgg':VGG,
         'densenet':densenet121,
         }

save_chks = range(num_epochs) # iterable of epochs for which to save the model
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()

# set up run dir 
run_dir = os.path.join('models/saved_models', model_base ,run_name)
os.makedirs(run_dir, exist_ok = True)
log, logclose = create_logger(log_filename=os.path.join(run_dir, 'train.log'), display = False)
log(f'using device: {device}')
log(f'saving models to: {run_dir}')
log(f'using base model: {model_base}')
log(f'using batch size: {batch_size}')
log(f'learning rate: {learning_rate}')
log(f'random seed: {random_seed}')
log(f'use_albumentations: {use_albumentations}')




torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


# dataloader
if use_split==True:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader_split(data_dir="data/raw/",  batch_size=batch_size, shuffle=True,use_albumentations=use_albumentations)
else:
    train_dataloader, test_dataloader, val_dataloader = get_data_loader(data_dir="Data/",  batch_size=batch_size, shuffle=True)

# define model 
model = models[model_base]()
model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
elif opt == 'adam': 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()


train_loss, val_loss, train_metrics, val_metrics, best_val_loss, best_val_loss_path, best_val_accuracy, best_val_accuracy_path, top1_train_accuracy, top1_val_accuracy  = train(model, train_dataloader, val_dataloader, num_epochs, save_checkpoints,run_dir,optimizer,criterion,model_base)


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

print('-'*100)
print(f'Using {best_val_loss_path}')
# best_model_path = os.path.join(run_dir, best_val_loss_path)
print("best_val_loss_path Metrics")
if os.path.exists(best_val_loss_path):
    model = torch.load(best_val_loss_path)
else:
    print("Best model weights not found.")
    # return

test_loss, test_metric,top1_test_accuracy = test(model, test_dataloader, device, criterion)
create_cm(model, best_val_loss_path, test_dataloader)
print('-'*100)



print('-'*100)
print(f'Using {best_val_accuracy_path}')
# best_model_path = os.path.join(run_dir, 'best_model.pth')
if os.path.exists(best_val_accuracy_path):
    model = torch.load(best_val_accuracy_path)
else:
    print("Best model weights not found.")
    # return

new_best_val_accuracy_path = os.path.join('models/saved_models/best_models/', f'{model_base}_best.pt')

# Check if the original path exists and create a new copy at the new path
if os.path.exists(best_val_accuracy_path):
    shutil.copy(best_val_accuracy_path, new_best_val_accuracy_path)
    print(f"Created a new copy of the best model at {new_best_val_accuracy_path}")
else:
    print("Original best model weights not found.")


test_loss, test_metric,top1_test_accuracy  = test(model, test_dataloader, device, criterion)
create_cm(model, best_val_accuracy_path, test_dataloader)
print('-'*100)

df = get_df(run_dir)
add_to_database(df, model_base, run_name, train_metrics, train_loss, val_metrics, val_loss, \
                best_val_loss, \
                best_val_loss_path, \
                best_val_accuracy, \
                best_val_accuracy_path,test_loss,test_metric,top1_train_accuracy, top1_val_accuracy,top1_test_accuracy)



# get_tests(model, os.path.join(run_dir, 'test'), test_dataloader)



# best_model_path = os.path.join(run_dir, 'best_model.pth')
# if os.path.exists(best_model_path):
#     model.load_state_dict(torch.load(best_model_path))
# else:
#     print("Best model weights not found.")
#     # return

# test_loss, test_metric = test(model,test_dataloader,criterion)




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
