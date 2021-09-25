import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt



def train(model, iterator, optimizer, criterion, device):
    total = 0
    correct = 0
    epoch_loss = 0
    epoch_acc = 0
    predicted_list = []
    model.train()
    
    for batch, labels in iterator:
        
        #Move tensors to the configured device
        batch  = batch.to(device)
        labels = labels.to(device)
       
        
        #Forward pass
        outputs = model(batch.float())
        outputs = outputs.to(device)
        
        loss    = criterion(outputs, labels).to(device)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        #check accuracy
        predictions  = model(batch.float())
        _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
        total        += labels.size(0)  #keep track of total
        correct      += (predicted == labels).sum().item()  #.item() give the raw number
        acc          = 100 * (correct / total)
                
        epoch_loss   += loss.item()
        epoch_acc    = acc
        predicted_list.append(predicted)
        
    return epoch_loss / len(iterator), epoch_acc, predicted_list

#======================================================================
def evaluate(model, iterator, criterion , device):
    
    total        = 0
    correct      = 0
    epoch_loss   = 0
    epoch_acc    = 0
    predicted_list = []
    labels_list    = []
        
    model.eval()
    
    with torch.no_grad():
    
        for batch, labels in iterator:
            
            #Move tensors to the configured device
            batch  = batch.to(device)
            labels = labels.to(device)
            
            #print(labels)
            

            predictions = model(batch.float())
            loss        = criterion(predictions, labels)
            
            
            _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
            #print(predicted)
            
            total   += labels.size(0)  #keep track of total
            correct += (predicted == labels).sum().item()  #.item() give the raw number
            acc     = 100 * (correct / total)
            
            epoch_loss += loss.item()
            epoch_acc  += acc
            
            labels_list.append(labels)
            predicted_list.append(predicted)
           

    return epoch_loss / len(iterator), epoch_acc / len(iterator) ,predicted_list, labels_list


#======================================================================

# define a time function useful for calculating time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def do_plot(train_losses, valid_losses):
    plt.figure(figsize=(25,5))
#     clear_output(wait=True)
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='valid_loss')
    plt.title('Classification based Encoder loss')
    plt.legend()
    plt.show()

def save_loss_graph(_train_losses, _valid_losses, _path, _file):
    import matplotlib as mpl
    mpl.use('agg')
    _fig = plt.figure()
    plt.figure(figsize=(25,5))
    _ax = plt.plot(_train_losses, label='train loss')
    plt.plot(_valid_losses, label='valid_loss')
    plt.title('Classification based Encoder loss')
    plt.legend()
    plt.savefig(f'{_path}{_file}-loss.png')
    plt.close(_fig)

def save_acc_graph(_train_acc, _valid_acc, _path, _file):
    import matplotlib as mpl
    mpl.use('agg')
    _fig  = plt.figure()
    plt.figure(figsize=(25,5))
    plt.plot(_train_acc, 'r' , label='train acc')
    plt.plot(_valid_acc, 'g' , label='valid acc')
    plt.title('Accuracy graph')
    plt.legend()
    plt.savefig(f'{_path}{_file}-acc.png')
    plt.close(_fig)
