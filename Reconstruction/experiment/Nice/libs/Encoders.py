
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt


#======================================================================
class Cnn1dEncoder(nn.Module):
    '''
    Expected Input Shape: (batch, channels, height , width)
                          (128   , 15      , 1      , 58   )
    '''
    def __init__(self):
        super(Cnn1dEncoder, self).__init__()
        
        # self.activation = nn.Tanh()
        # self.activation     = nn.ELU()
        self.activation = nn.ReLU()
        #self.activation = nn.LeakyReLU()
        self.is_debug       = False   
        self.channel_out_c1 = 32
        self.kernel_size    = 9
        self.stride         = 1
        self.drop_out       = 0.5
        
    def init_network(self, _width):
        ## calculate the fc1 input channel size
        ## since cnn size is I-(K-1) , but we have 2 CNN layers
        ## So, I just run it two time then, multiply with the CNN layter's two out put size
        self.fc1_ch_in    = _width-(self.kernel_size-1)
        self.fc1_ch_in    = (self.fc1_ch_in-(self.kernel_size-1))*(self.channel_out_c1*2)
        self.conv1 = nn.Sequential(    nn.Conv1d(15, self.channel_out_c1   , kernel_size=(1,self.kernel_size),   padding=(0,0), stride=(1,self.stride))  ,  self.activation )
        self.conv2 = nn.Sequential(    nn.Conv1d(32, self.channel_out_c1*2 , kernel_size=(1,self.kernel_size),   padding=(0,0), stride=(1,self.stride))  ,  self.activation )
        self.fc1   = nn.Sequential(    nn.Linear(self.fc1_ch_in,1024),  self.activation ,nn.Dropout( self.drop_out )   )
        self.fc2   = nn.Sequential(    nn.Linear(1024,512),     self.activation ,nn.Dropout( self.drop_out )   )
        self.fc3   = nn.Sequential(    nn.Linear(512,256),      self.activation ,nn.Dropout( self.drop_out )   )
        self.fc4   = nn.Sequential(    nn.Linear(256,128),      self.activation ,nn.Dropout( self.drop_out )    )
        self.fc5   = nn.Sequential(    nn.Linear(128,64),       self.activation ,nn.Dropout( self.drop_out )    )
        self.fc6   = nn.Sequential(    nn.Linear(64,32),        self.activation ,nn.Dropout( self.drop_out )    )
        self.fc7   = nn.Sequential(    nn.Linear(32,10)     )

    def encode(self, X):
        if self.is_debug  : print('--------Convolute--------'); print(X.shape) 

        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
       
        X = self.conv2(X)
        if self.is_debug  : print(X.shape) 
       
        X = X.flatten(start_dim = 1)
        if self.is_debug  : print('--------Flatten--------') ; print(X.shape) 

        X = self.fc1(X)
        if self.is_debug : print('----------FC----------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc5(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc6(X)
        if self.is_debug  : print(X.shape) 
            
        X = self.fc7(X)
        if self.is_debug  : 
            print(X.shape)
            time.sleep(20)
    
       
        return X
        
    def forward(self,X):
        X = self.encode(X)
        return X
    
    def get_latent( self, X):
        if self.is_debug  : print('--------Convolute--------'); print(X.shape) 
            
        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
            
        X = self.conv2(X)
        if self.is_debug  : print(X.shape) 
            
        X = X.flatten(start_dim = 1)
        if self.is_debug  : print('--------Flatten--------') ; print(X.shape) 
 
        X = self.fc1(X)
        if self.is_debug : print('--------FC--------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
    
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc5(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc6(X)
        if self.is_debug  : print(X.shape) 
        
        return X
    
    def classifier(self, latent):
        return self.fc5(latent)


#======================================================================

class Cnn2dEncoder(nn.Module):
    '''
    Expected Input Shape: (batch, channels, width , height)
                          (64   , 15      , x      , xx   )
    '''
    def __init__(self, ):
        super().__init__()
        
#         self.activation = nn.Tanh()
        self.activation     = nn.ELU()
        # self.activation     = nn.ReLU()
        self.channel_out_c1 = 32
        self.kernel_size    = 3 
        self.stride         = 1
        self.drop_out       = 0.5
        self.is_debug       = False   

    def init_network(self, _high, _width):
        # if _width > _high:
        #     self.kernel_size   = _high-2
        # else:
        #     self.kernel_size   = _width -2

        print(self.kernel_size)

        ## calculate the fc1 input channel size
        ## since cnn size is I-(K-1) , but we have 2 CNN layers
        ## So, I just run it two time then, multiply with the CNN layter's two out put size
        

        self.width_conv    = _width-(self.kernel_size-1)
        self.width_conv    = self.width_conv-(self.kernel_size-1)
        
        self.high_conv     = _high-(self.kernel_size-1)
        self.high_conv     = self.high_conv-(self.kernel_size-1)
        
        self.fc1_ch_in     = self.width_conv * self.high_conv * self.channel_out_c1*2
        
        self.conv1 = nn.Sequential(    nn.Conv2d(15, self.channel_out_c1   , kernel_size=(self.kernel_size,self.kernel_size),   padding=(0,0), stride=(self.stride,self.stride))  ,  self.activation )
        self.conv2 = nn.Sequential(    nn.Conv2d(32, self.channel_out_c1*2 , kernel_size=(self.kernel_size,self.kernel_size),   padding=(0,0), stride=(self.stride,self.stride))  ,  self.activation )
        self.fc1   = nn.Sequential(    nn.Linear(self.fc1_ch_in,512),  self.activation ,nn.Dropout( self.drop_out )    )
        self.fc2   = nn.Sequential(    nn.Linear(512,128),  self.activation ,nn.Dropout( self.drop_out )    )
        self.fc3   = nn.Sequential(    nn.Linear(128,64),  self.activation  ,nn.Dropout( self.drop_out )    )
        self.fc4   = nn.Sequential(    nn.Linear(64,32),  self.activation   ,nn.Dropout( self.drop_out )    )
        self.fc5   = nn.Sequential(    nn.Linear(32,10)   )
    def encode(self, X):
        
    
        if self.is_debug  : 
            print('--------Convolute--------')
            print(X.shape) 
            print(f'Kernel size : {self.kernel_size}')
            
        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
       
        X = self.conv2(X)
        if self.is_debug  : 
            print(X.shape) 
            print(self.width_conv)
            print(self.high_conv)
            print(self.fc1_ch_in)
            
       
        X = X.flatten(start_dim = 1)
        if self.is_debug  : print('--------Flatten--------') ; print(X.shape) 

        X = self.fc1(X)
        if self.is_debug : print('----------FC----------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc5(X)
        if self.is_debug  : 
            print(X.shape)
            time.sleep(200)
        
       
        return X
        
    def forward(self,X):
        X = self.encode(X)
        return X
    
    def get_latent( self, X):
        if self.is_debug  : 
            print('--------Convolute--------'); print(X.shape) 
            
        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
            
        X = self.conv2(X)
        if self.is_debug  : print(X.shape) 
            
        X = X.flatten(start_dim = 1)
        if self.is_debug  : print('--------Flatten--------') ; print(X.shape) 
 
        X = self.fc1(X)
        if self.is_debug : print('--------FC--------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 
        
        return X
    
    def classifier(self, latent):
        return self.fc5(latent)
  