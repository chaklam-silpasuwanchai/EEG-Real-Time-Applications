
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch.optim as optim
import numpy as np
import time, copy
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from .EEGModels.EEGNet import EEGNet
from .utilities import get_freer_gpu, squeeze_tensor_to_list


class Model_Trainning():
    def __init__( self , model_obj , learning_rate=0.001, criterion=nn.CrossEntropyLoss()  ):

        self.device         = torch.device( get_freer_gpu() if torch.cuda.is_available() else 'cpu')
        print( "Model train device:", self.device )
        self.MODEL          = model_obj.to(self.device)
        self.criterion      = criterion.to(self.device)
        self.optimizer      = torch.optim.Adam( list( self.MODEL.parameters()) , lr=learning_rate)
        # self.scheduler    = ReduceLROnPlateau(self.optimizer, 'min', patience = 10)
        self.scheduler       = None   # set it in the do_trainning()
        
        self.best_model      = None
        self.best_train_acc  = None
        self.best_val_acc    = None
        self.best_test_acc   = None
        self.best_epoch      = None 
        print(f'The model {type(self.MODEL).__name__} has {self.count_parameters():,} trainable parameters')# Train the model
        # print(self.MODEL)
        
        self.best_val_loss = float('inf')
        self.train_losses    = []
        self.train_accuracies= []
        self.val_losses      = []
        self.val_accuracies  = []

        self.is_debug        = False
        self.is_plot_graph   = False
        self.curr_lr         = None
        
    def count_parameters(self):
        return sum(p.numel() for p in self.MODEL .parameters() if p.requires_grad)

    def train_epoch(self, iterator ):
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_acc = 0
        predicted_list = []
        self.MODEL.train()
    
        for batch, labels in iterator:
            
            #Move tensors to the configured device
            batch  = batch.to(self.device)
            labels = labels.to(self.device)
        
            #Forward pass
            outputs = self.MODEL( batch.float() )
            outputs = outputs.to(self.device)
            loss    = self.criterion(outputs, labels.long()).to(self.device)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                    
            #check accuracy
            predictions  = self.MODEL(batch.float())
            _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
            total        += labels.size(0)  #keep track of total
            correct      += (predicted == labels).sum().item()  #.item() give the raw number
            acc          = 100 * (correct / total)
                    
            epoch_loss   += loss.item()
            epoch_acc    = acc
            predicted_list.append(predicted)
            
            #print("labels", labels)
            #print("predicted", predicted)
            
            self.curr_lr = self.optimizer.param_groups[0]['lr']
            # self.scheduler.step(epoch_loss/len(iterator))
            if self.scheduler != None:
                self.scheduler.step()

        return epoch_loss / len(iterator), epoch_acc, predicted_list

    def evaluate_epoch(self ,  iterator, test_mode=False):
        total        = 0
        correct      = 0
        epoch_loss   = 0
        epoch_acc    = 0
        predicted_list = []
        labels_list    = []
        
        if test_mode:
            self.MODEL =  copy.deepcopy( self.best_model )
            print("...Load best model weight to the model for test")
        
        self.MODEL.eval()
        with torch.no_grad():
            for batch, labels in iterator:
                #Move tensors to the configured device
                batch  = batch.to(self.device)
                labels = labels.to(self.device)
                
                #print(labels)
            
                predictions = self.MODEL(batch.float())
                loss        = self.criterion(predictions, labels.long())
                 
                _, predicted = torch.max(predictions.data, 1)  # returns max value, indices
                #print(predicted)
                
                total   += labels.size(0)                      # keep track of total
                correct += (predicted == labels).sum().item()  #.item() give the raw number
                acc     = 100 * (correct / total)
                
                epoch_loss += loss.item()
                epoch_acc  = acc
                
                labels_list.append(labels)
                predicted_list.append(predicted)

        return epoch_loss / len(iterator), epoch_acc , predicted_list[-1].tolist()  , labels_list[-1].tolist()


    def do_training( self, train_iterator_in, val_iterator_in,  n_epochs = 10, scheduler=None  ):
        
        print(f"...Do trainning for : {n_epochs} epochs")
        
        if scheduler == "MultiStempLR":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[ n_epochs//8 , n_epochs//6 ,  n_epochs//4 ,  n_epochs//2 ], gamma=0.1)
            
        best_valid_loss = float('inf')
        best_epoch      = 0
           
        for epoch in range(n_epochs):
            train_loss, train_acc, train_predicted       = self.train_epoch(     train_iterator_in    )
            val_loss, val_acc, valid_predicted, actual_y = self.evaluate_epoch(  val_iterator_in    )

            self.train_losses.append(       train_loss  )
            self.train_accuracies.append(   train_acc   )
            self.val_losses.append(         val_loss    )
            self.val_accuracies.append(     val_acc     )
            
            
            ## display progress              
            if self.is_debug : display_every = 2 
            else:  display_every = 20
            
            
            if (epoch+1) % display_every == 0:
                
                if self.is_plot_graph    :
                    clear_output(wait=True)
                    data_plot = { "train_loss" : self.train_losses, "validate_loss": self.val_losses }
                    self.do_plot(data_plot, title= "Clasification base loss")

                print(f'Epoch: {epoch+1:02}/{n_epochs} |',end='')
                print(f'\tTrain Loss: {train_loss:.5f}   | Train Acc: {train_acc:.2f}%   |', end='')
                print(f'\t Val. Loss: {val_loss:.5f}  | Val. Acc: {val_acc:.2f}%   |', end='')
                print(f'\t LR: {self.curr_lr} |', end='')
                print(f"\tBest epoch : {self.best_epoch}")
                print(f"Actual  \t: { [ int(x) for x in actual_y ] }")
                print(f"Predicted \t: {valid_predicted}") 

            if val_loss < self.best_val_loss:
                self.best_val_loss   = val_loss
                self.best_train_acc  = train_acc
                self.best_val_acc    = val_acc
                self.best_epoch      = epoch 
                #print("Keep best weight of model")
                self.best_model     = copy.deepcopy( self.MODEL )

    def save_state_dict(self, path):
        if self.best_model != None:
            print("Model:{} saved.".format(type(self.best_model).__name__))
            torch.save(self.best_model.state_dict(), f'{path}_{type(self.best_model).__name__}_dict.pt.tar')
        else:
            print("There is no model to save")


    
    def do_testing(self, test_iterator_in):
 
        test_loss, test_acc , y_hat_test, y_test = self.evaluate_epoch( test_iterator_in , test_mode=True )
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        print(f"The number of test {len(y_test)}")
        out = zip(y_test, y_hat_test)
        print("(Actual y , Predicted y)")
        print(list(out))
        self.best_test_acc   = test_acc

    def get_result(self):
        result = [ type(self.best_model).__name__ ,  self.best_train_acc, self.best_val_acc, self.best_test_acc , self.best_epoch ]
        return result

    def do_plot(self,  data_dic, title="" ):
        plt.figure(figsize=(15,3))
        for _key, _val in data_dic.items():
            plt.plot( _val , label=_key)
        plt.title(title)
        plt.legend()
        plt.show()

    def save_plot(self, save_path, data_dic, title="" ):
        import matplotlib as mpl
        mpl.use('agg')
        fig = plt.figure(figsize=(15,3))
        for _key, _val in data_dic.items():
            plt.plot( _val , label=_key)
        plt.title(title)
        plt.legend()
        plt.savefig(f'{save_path}-{title}.png')
        plt.close(fig)