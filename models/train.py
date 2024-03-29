# import os 
# from tqdm import tqdm

# import matplotlib.pyplot as plt
import numpy as np

import torch 


from tqdm import tqdm



def train(model, train_dataloader, val_dataloader, num_epochs, save_checkpoints,run_dir,optimizer,criterion,model_base):
    

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        torch.mps.empty_cache()
    model.to(device)


    train_loss = []
    val_loss_list = []
    train_metrics = []
    val_metrics = []
    best_val_accuracy =  0.0  # Initialize the best validation accuracy
    best_val_loss = float('inf')  # Initialize the best validation loss

    fortopk_train_preds = []
    fortopk_train_targets = []
    fortopk_val_preds = []
    fortopk_val_targets = []

    best_val_loss_path = ''
    best_val_accuracy_path = ''

    for epoch in range(num_epochs):
        print('-'*100)
        print(f'\t\t\t\t\tEpoch: {epoch}\t\t')
        print('-'*100)
        # Training phase
        model.train()
        batch_loss = []
        batch_metric = []
        total_imgs =  0
        for i, (_data, _target) in tqdm(enumerate(train_dataloader)):
            data = _data.to(device)
            target = _target.to(device)
            optimizer.zero_grad()
            pred = model(data)

            fortopk_train_preds.append(pred)
            fortopk_train_targets.append(target)


            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_metric.append((pred.argmax(dim=1) == target).sum().item())
            total_imgs += len(target)
        train_loss.append(sum(batch_loss) / len(train_dataloader))
        train_metrics.append(sum(batch_metric) / total_imgs)
        print(f"Training Metric --- Train Accuracy: {sum(batch_metric)/total_imgs} ---- Train Loss: {sum(np.array(batch_loss)/len(train_dataloader))}")
        print('-'*100)
        del data 
        del target
        del pred
        del loss
        
        # Validation phase
        model.eval()
        batch_metric = []
        batch_loss = []
        total_imgs =  0
        # Calculate validation metrics
        
        with torch.no_grad():
            for i, (_data, _target) in enumerate(val_dataloader):
                data = _data.to(device)
                target = _target.to(device)
                pred = model(data)

                fortopk_val_preds.append(pred)
                fortopk_val_targets.append(target)


                loss = criterion(pred, target)
                batch_loss.append(loss.item())
                batch_metric.append((pred.argmax(dim=1) == target).sum().item())
                total_imgs += len(target)
        
        val_accuracy = sum(batch_metric) / total_imgs
        val_loss = sum(np.array(batch_loss)/len(train_dataloader))

        val_loss_list.append(val_loss)
        val_metrics.append(val_accuracy)

        print(f"Validation Metric --- Val Accuracy: {val_accuracy} ---- Val Loss: {val_loss}")
        print('-'*100)


        # Check if the current validation accuracy is better than the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model because the validation accuracy has improved
            # torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'train_loss': sum(np.array(batch_loss)/len(train_dataloader)),
            #         'val_loss': sum(np.array(batch_loss)/len(val_dataloader)),
            #         'model': model
            #         }, f'{run_dir}/{epoch}.chkpt')
            print("Saving Model, got better val accuracy")
            torch.save(model, f'{run_dir}/{model_base}_best_val_acc_{epoch}.pt')

        # Check if the current validation loss is lower than the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model because the validation loss has decreased
            print("Saving Model, got better val loss")
            best_val_loss_path = f'{run_dir}/{model_base}_best_val_loss_{epoch}.pt'
            torch.save(model, f'{run_dir}/{model_base}_best_val_loss_{epoch}.pt')


        # Save checkpoint if required
        if epoch in save_checkpoints:
            print(f'Saving {run_dir}/{epoch}.chkpt')
            
            # Save checkpoint
            # torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'train_loss': sum(np.array(batch_loss)/len(train_dataloader)),
            #         'val_loss': sum(np.array(batch_loss)/len(val_dataloader)),
            #         'model': model
            #         }, f'{run_dir}/{epoch}.chkpt')
            
            # Save Whole Model
            best_val_accuracy_path = f'{run_dir}/{model_base}_full_model_{epoch}.pt'
            torch.save(model,f'{run_dir}/{model_base}_full_model_{epoch}.pt')
            # torch.save(model.state_dict(), f'{run_dir}/{epoch}.chkpt')
        


        total_correct = sum([(pred.argmax(dim=1) == target).float().sum().item() for pred, target in zip(fortopk_train_preds, fortopk_train_targets)])
        total_samples = sum([len(target) for target in fortopk_train_targets])
        top1_train_accuracy = total_correct / total_samples
        print(f"Final Top-1 Train Accuracy: {top1_train_accuracy}")

        total_correct = sum([(pred.argmax(dim=1) == target).float().sum().item() for pred, target in zip(fortopk_val_preds, fortopk_val_targets)])
        total_samples = sum([len(target) for target in fortopk_val_targets])
        top1_val_accuracy = total_correct / total_samples
        print(f"Final Top-1 Val Accuracy: {top1_val_accuracy}")



        if device == 'mps':
            torch.mps.empty_cache()

        print('\n\n')


    return train_loss, val_loss, train_metrics, val_metrics, best_val_loss, best_val_loss_path, best_val_accuracy, best_val_accuracy_path, top1_train_accuracy, top1_val_accuracy

