
import matplotlib.pyplot as plt
import numpy as np
import random, time , pandas as pd
import inspect
import mne
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from libs.EEGModels.EEGNet import *
from libs.utilities import get_freer_gpu
from libs.train_utilities import do_plot
from libs.EEG_preProcessing import df_to_raw, eog_remove, mne_to_numpy, minmax_norm, get_epochs, zscore

class EEG_Signal():
    def __init__( self, eeg_file , channels, event_id ):
        # super(self, Participant).__init__()
        self.is_debug   = False
        self.event_id   = event_id
        self.channels   = channels
        
        self.data_frame = self.load_csv( eeg_file ) 
        
        self.mne_eeg_signal = None
        self.X_eeg      = None
        self.y_label    = None
        
    def load_csv (self, eeg_file) :
        _df = pd.read_csv( eeg_file ,low_memory=False)
        _df = _df.drop(["timestamps"], axis=1)
         #channels named according to how we plug our eeg device
        # _df.columns = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3','Pz', 'P4','O1', 'O2','unuse', 'Marker']  

        _df.columns     = self.channels

        # remove the 16th channel since we recored only 15 channel
        _df = _df.drop(["unuse"], axis=1)
        _df.head()
        return _df
        
    def fill_iti_time(self , iti_time_in ):
        start_time = time.time()
        #use numpy as another view of the pandas columns for faster operation
        _df_eeg_raw = self.data_frame.copy()
        _marker_np  = _df_eeg_raw['Marker'].values

        for idx, marker in enumerate( _marker_np ):
            if "," in str(marker):
                if ( marker.split(",")[-1] == iti_time_in):
                    _marker_np[idx] = int(self.event_id[marker.split(",")[-4]])
                elif (marker.split(",")[-2] == iti_time_in     ):
                    _marker_np[idx] = int(self.event_id[marker.split(",")[-4]])
                else:
                    _marker_np[idx] = int(0)
            else:
                _marker_np[idx] = int(0)
        
        # #check whether _df['Marker'] changed according to np
        _df_eeg_raw['Marker'] = _marker_np
        print(_df_eeg_raw.groupby('Marker').agg(['count']))

        self.mne_eeg_signal = df_to_raw( _df_eeg_raw )
        print("smapling rate: ", self.mne_eeg_signal.info['sfreq'])
        print("len of mne_eeg_signal", self.mne_eeg_signal.__len__() )
        if self.is_debug :
            print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

    def banpass_filter(self, fmin, fmax):
        # notch filter 
        start_time   = time.time()
        freqs =(50,)
        _raw_format = self.mne_eeg_signal.copy().notch_filter(freqs=freqs, phase='zero',filter_length='auto', verbose=self.is_debug)
        if self.is_debug:
            _raw_format.plot_psd()

        # band pass filter
        _raw_format.filter(fmin, fmax, method='iir', verbose=self.is_debug)
        if self.is_debug:
            _raw_format.plot_psd()

        self.mne_eeg_signal = _raw_format
        
        del _raw_format

        if self.is_debug:
            self.print_debug( time.time()-start_time )

    def eog_removal(self, electrod, ratio, remove_method):
        start_time   = time.time()
        # eog removal
        # eog_remove(_mne_raw_eeg, _ch_name, _threshold, _measure )
        self.mne_eeg_signal = eog_remove(self.mne_eeg_signal, electrod, ratio, remove_method)
        if self.is_debug:
            self.print_debug( time.time()-start_time )

        if self.is_debug:
            self.print_debug( time.time()-start_time )
    
    def csp( self, n_components=15 ) :
        start_time   = time.time()
        from mne.decoding import CSP    
            
        csp = CSP(n_components=n_components, transform_into='csp_space', reg=None, log=None)
        self.X_eeg = csp.fit_transform(self.X_eeg, self.y_label)

#         evoked = epochs.average()
#         evoked.data = csp.patterns_.T
#         evoked.times = np.arange(evoked.data.shape[0])

#         layout = read_layout('EEG1020')
#         evoked.plot_topomap(times=[0, 1, 2, 3, 4, 5], ch_type='eeg', layout=layout,
#                             scale_time=1, time_format='%i', scale=1,
#                             unit='Patterns (AU)', size=1.5)
        
        
            
        if self.is_debug:
            self.print_debug( time.time()-start_time )  
        
    def mne_to_np(self, event_id , stim_time):
        start_time   = time.time()
        # MNE raw to numpy
        #this one requires expertise to specify the right tmin, tmax
        #_tmin = 0.04 #0
        _tmin = 0.0 #0
        _tmax = float(stim_time) #0.5 seconds

        # _ , self.X_eeg, self.y_label = mne_to_numpy(self.mne_eeg_signal, event_id, _tmin, _tmax)
        # self.mne_eeg_signal = None


        # if self.is_debug:
        #     self.print_debug( time.time()-start_time )

        _picks       = mne.pick_types(self.mne_eeg_signal.info, eeg=True)
        _epochs  = get_epochs(self.mne_eeg_signal, event_id, _tmin, _tmax, _picks)
        _y       = _epochs.events[:, -1]
        self.y_label  = _y - 1
        self.X_eeg    = _epochs.get_data()
    

        if self.is_debug:
            print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

        # return _epochs, _X_eeg, _y


    def minmax_norm(self):
        
        self.X_eeg      = minmax_norm(self.X_eeg)
        if self.is_debug:
            self.print_debug( time.time()-start_time )

            
    def zscore(self):
        # z-score each stimulus in each channel
        self.X_eeg  = zscore(self.X_eeg )
  
        
    def get_XY(self):
        return self.X_eeg, self.y_label

    def convert_to_4d(self, d0, d1, d2, d3):
        self.X_eeg = self.X_eeg.reshape(d0,d1,d2,d3)
        print("Converted X_eeg to ", self.X_eeg.shape )
        pass

    def get_EEG_shape(self):
        return self.X_eeg.shape

    def update_XY(self, X_in, y_in):
        
#         if not torch.is_tensor( X_in ) :
#             X_in =  torch.tensor(X_in)
            
#         if not torch.is_tensor( y_in ) :
#             y_in =  torch.tensor(y_in)
        
        
        self.X_eeg = X_in
        self.y_label = y_in

    def iterator_split_3( self, ratio_in =[0.1, 0.22], batch_size=256, shuffle=True):
        
        #ratio_in=[0.1, 0.22]  ( 70, 20, 10)
        X_train_val, X_test, y_train_val, y_test    = train_test_split( self.X_eeg, self.y_label,test_size=ratio_in[0],shuffle=True, stratify =self.y_label)
        X_train,     X_val,  y_train,     y_val     = train_test_split( X_train_val, y_train_val, test_size = ratio_in[1], shuffle=True, stratify =y_train_val)
        # from collections import Counter
        # print(Counter(np.sort(y_test)))
        
        torch_X_train = torch.from_numpy(X_train)
        torch_y_train = torch.from_numpy(y_train)
        print(f'torch_X_train size : {torch_X_train.size()}' )

        torch_X_val  = torch.from_numpy(X_val)
        torch_y_val  = torch.from_numpy(y_val)
        print(f'torch_X_val size : {torch_X_val.size()}' )

        torch_X_test = torch.from_numpy(X_test)
        torch_y_test = torch.from_numpy(y_test)
        print(f'torch_X_test size : {torch_X_test.size()}' )

        # Define dataset
        train_set  = TensorDataset(torch_X_train, torch_y_train)
        valid_set  = TensorDataset(torch_X_val, torch_y_val)
        test_set   = TensorDataset(torch_X_test, torch_y_test)

        BATCH_SIZE = batch_size #keeping it binary so it fits GPU
        #Train set loader
        train_iterator = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE , shuffle=shuffle)
    
        #Validation set loader
        valid_iterator = torch.utils.data.DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=shuffle)

        #Test set loader
        test_iterator  = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=shuffle)

        return train_iterator, valid_iterator, test_iterator

    def iterator_split( self, ratio_in =[0.1, 0.22], batch_size=256, shuffle=True):
        
        #ratio_in=[0.1, 0.22]  ( 70, 20, 10)
        X_train, X_val, y_train, y_val    = train_test_split( self.X_eeg, self.y_label,test_size=ratio_in[1],shuffle=True, stratify =self.y_label)
       
        # from collections import Counter
        # print(Counter(np.sort(y_test)))
        
        torch_X_train = torch.from_numpy(X_train).float()
        torch_y_train = torch.from_numpy(y_train).float()
        print(f'torch_X_train size : {torch_X_train.size()}' )

        torch_X_val = torch.from_numpy(X_val).float()
        torch_y_val = torch.from_numpy(y_val).float()
        print(f'torch_X_test size : {torch_X_val.size()}' )

        # Define dataset
        train_set  = TensorDataset(torch_X_train, torch_y_train)
        val_set   = TensorDataset(torch_X_val, torch_y_val)

        BATCH_SIZE = batch_size #keeping it binary so it fits GPU
        #Train set loader
        train_iterator = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE , shuffle=shuffle)
    
        #Validation set loader
        val_iterator = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=shuffle)


        return train_iterator, val_iterator
    
    def get_num_class(self):
        return len(np.unique(self.y_label))
    
    def sort_by_y(self):
        tmp = list(zip(self.X_eeg,self.y_label))
        res = sorted(tmp, key = lambda x: x[1])
        X, y = zip(*res)
        
        self.X_eeg, self.y_label = np.array(X), np.array(y)
        

    def get_sampling_rate( self ) :
        return self.mne_eeg_signal.info['sfreq']
    
    def print_debug(self, time_duration):
        print(f'time_trace of {inspect.stack()[1][3]} = {time_duration:.4f}')
        print("="*30)