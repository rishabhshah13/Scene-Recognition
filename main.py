# import packages 
import torch 

run_name = "jly_0131_resnet_default"
num_epochs = 100
save_chks = range(num_epochs) # iterable of epochs for which to save the model

# make dataset + dataloader
train_dataset = 
val_dataset = 
train_dataloder = 
val_dataloder = 

# define model 
model = 

# define optimizer and criterion
optimizer = 
criterion = 

# training loop
train_loss = []
val_loss = []
train_metrics = []
val_metrics = []
for epoch in range(num_epochs):
    #training
    model.train()
    batch_loss = []
    for i, (data, target) in enumerate(train_dataloader): 
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        batch_loss.append(loss)
        optimizer.step()
    train_loss.append(sum(batch_loss/len(train_dataloader)))
    train_metrics.append() #TODO: add metrics

    # validation
    with torch.no_grad():
        model.eval()
        batch_loss = []
        for i, (data, target) in enumerate(val_dataloader): 
            pred = model(data)
            loss = criterion(pred, target)
            batch_loss.append(loss)
        val_loss.append(sum(batch_loss/len(val_dataloader)))
        train_metrics.append() #TODO: add metrics

    if epoch in save_chks: 
        torch.save(model.state_dict(), run_name)


# testing
with torch.no_grad():
    model.eval()