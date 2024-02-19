from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def test(model, test_dataloader, device, criterion):
    model.eval()
    batch_metric = []
    batch_loss = []
    total_imgs =  0
    
    fortopk_test_preds = []
    fortopk_test_targets = []


    with torch.no_grad():
        for i, (_data, _target) in tqdm(enumerate(test_dataloader)):
            data = _data.to(device)
            target = _target.to(device)
            pred = model(data)

            fortopk_test_preds.append(pred)
            fortopk_test_targets.append(target)


            loss = criterion(pred, target)
            batch_loss.append(loss.item())
            batch_metric.append((pred.argmax(dim=1) == target).sum().item())
            total_imgs += len(target)

    test_accuracy = sum(batch_metric) / total_imgs
    test_loss = sum(np.array(batch_loss) / len(test_dataloader))

    total_correct = sum([(pred.argmax(dim=1) == target).float().sum().item() for pred, target in zip(fortopk_test_preds, fortopk_test_targets)])
    total_samples = sum([len(target) for target in fortopk_test_targets])
    top1_test_accuracy = total_correct / total_samples
    print(f"Final Top-1 Test Accuracy: {top1_test_accuracy}")
    

    print('-'*100)
    print(f"Test Metric --- Test Accuracy: {test_accuracy} ---- Test Loss: {test_loss}")
    print('-'*100)


    return test_accuracy, test_loss, top1_test_accuracy



def create_cm(model, save_path, dataloader): 
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)

    preds = []
    labels = []
    with torch.no_grad():
        for _data, _label in dataloader: 
            data = _data.to(device)
            label = _label.to(device)
            preds += model(data).argmax(dim=1).tolist()
            labels += label.tolist()
    try: 
        class_to_idx = dataloader.dataset.class_to_idx
    except: 
        class_to_idx = dataloader.dataset.dataset.class_to_idx
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels = class_to_idx,
                                 )
    
    disp.plot()
    plt.savefig(save_path+'_CM.png')
