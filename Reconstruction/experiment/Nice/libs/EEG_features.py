
from mne.time_frequency import psd_welch
import time
import mne
from mne import create_info
from mne.io import RawArray
import matplotlib.pyplot as plt
import numpy as np
import inspect




is_debug = True

#====================================================================
def making_psd(_epochs, _fmin, _fmax, _n_fft):
    start_time   = time.time()


    epochs_copy = _epochs.copy() # shape = (400, 15, 58)
    kwargs      = dict( fmin=_fmin, fmax=_fmax, n_jobs=1, n_fft = _n_fft )
    # kwargs = dict(fmin=1, fmax=40, n_jobs=1, n_fft = 10, n_overlap = 3, n_per_seg = 20)

    psds_welch, freqs = psd_welch(epochs_copy, average='mean', **kwargs)

    # psds_welch = np.log10(psds_welch) # take log

    all_mean =  psds_welch.mean(0) # get the mean of each FFT
    all_std = psds_welch.std(0) # get std of each FFT

    # print("PSD SHAPE : ",psds_welch.shape)
    # print("FREQ : ", freqs)
    # print("ALL MEAN ", all_mean.shape)
    # print("ALL STD ", all_std.shape)

    if is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

    return psds_welch


#====================================================================


def making_spectrogram(_np_eeg, _NFFT, _Fs, _overlap):
    _start_time  = time.time()
    _cmap        = plt.get_cmap('magma') 

    _fig = plt.figure()
    _test_spec = plt.specgram(_np_eeg[0,0,], NFFT=_NFFT, Fs=_Fs, noverlap=_overlap ,cmap=_cmap )[0]
    # plt.savefig('test.png')
    plt.close(_fig)

    
    _X_spec = np.zeros(  (_np_eeg.shape[0], _np_eeg.shape[1], _test_spec.shape[0], _test_spec.shape[1])  )
    for _sample in range(_np_eeg.shape[0]):
        for _channel in range(_np_eeg.shape[1]) :
            _fig = plt.figure()
            _X_spec[ _sample, _channel,:,:  ] = plt.specgram(_np_eeg[_sample,_channel,], NFFT=_NFFT, Fs=_Fs, noverlap=_overlap ,cmap=_cmap)[ 0]
            plt.close(_fig)

    if is_debug:
        print(f'EEG shape \t {_np_eeg.shape}' ) 
        print(f'Spectrogram shape\t{_X_spec.shape}' ) 
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-_start_time}')

    return _X_spec  # retrun 4d eeg


#====================================================================

import pywt
def wavelet_1d(_signal,_components:list):
    '''
    _signal input shape is 1d array
    since we had only 125hz, so we only need 4 level
    62.5
    cA1 : 1-31.25    cD1: 31.25-62.5
    cA2 : 1-15.625   cD2: 15.625-31.25 
    cA3 : 1-7.8125   cD3: 7.8125-15.625
    cA4 : 1-3.90625  cD4: 3.90625-7.8125
    ''' 
    dcom_level  = pywt.dwt_max_level(_signal.shape[0], 'bior3.9')
    coeffs      = pywt.wavedec(_signal, wavelet="bior3.9", level= dcom_level)

    if dcom_level == 1:
        cA1, cD1 = coeffs
    elif dcom_level ==2:
        cA2, cD2, cD1 = coeffs
    elif dcom_level ==3:
        cA3, cD3, cD2, cD1 = coeffs
    else:
        print("Check decomposition level")

    _selected_com = []
    for _component in _components :
        if _component == 'cA1':
            _selected_com.append(cA1.ravel())
        if _component == 'cA2':
            _selected_com.append(cA2.ravel())
        if _component == 'cA3':
            _selected_com.append(cA3.ravel())
        if _component == 'cD3':
            _selected_com.append(cD3.ravel())
        if _component == 'cD2':
            _selected_com.append(cD2.ravel())
        if _component == 'cD1':
            _selected_com.append(cD1.ravel())
    _selected_com = np.concatenate( _selected_com, axis=0 )

    return _selected_com

def making_wavelet4d(_signal, _components:list):
    # _signal is 3d eeg data [sample, channel, time_features]
    
    _start_time     = time.time()
    print(f'coefficians selected {_components}')

    _d4_lenth   = wavelet_1d(_signal[0,0,], _components).shape[0]
    print(_d4_lenth)

    _X_wavelet  = np.zeros(  ( _signal.shape[0], _signal.shape[1], 1, _d4_lenth )  )
    print(_X_wavelet.shape)
                    
    for _sample in range(_signal.shape[0]):
        for _channel in range(_signal.shape[1]) :
            _X_wavelet[_sample, _channel,0,:] = wavelet_1d( _signal[_sample, _channel,],  _components )

    del _d4_lenth
    if is_debug:
        print(f'selected wavelet coe : {_components}')
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-_start_time}')

    return _X_wavelet

#====================================================================

from sklearn.manifold import TSNE
import seaborn as sns

def tsne2d( data_in ):
    '''
    data_in  :n dimension of array
    label_in :label of data
    '''
    time_start     = time.time()
    _data_dim_size = len( data_in.shape)

    dim2 = 1
    for ii in range(_data_dim_size-1):
        dim2 = dim2 * data_in.shape[ii+1]
    _data =  data_in.reshape( data_in.shape[0] ,dim2 )
    tsne         = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=34)
    #tsne         = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform( _data )
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    return tsne_results

#====================================================================
from sklearn.manifold import Isomap
def isomap2d( data_in ):
    '''
    data_in  :n dimension of array
    label_in :label of data
    '''
    time_start     = time.time()
    _data_dim_size = len( data_in.shape)

    dim2 = 1
    for ii in range(_data_dim_size-1):
        dim2 = dim2 * data_in.shape[ii+1]
   
    _data        =  data_in.reshape( data_in.shape[0] ,dim2 )
    embedding    = Isomap(n_components=2)
    X_transformed = embedding.fit_transform( _data  )
    
    #print('isomap done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return X_transformed

#====================================================================
def scater_plot( data_in, label_list, label_uniuqe, title="", figsize=(10,5) ):
    plt.figure(figsize=figsize )
    plt.title(title)
    plot = plt.scatter(data_in[:,0], data_in[:, 1] , c = label_list , alpha=0.3 , cmap='tab10' ) 
    plt.legend( handles=plot.legend_elements()[ 0 ] , title="Class", labels=label_uniuqe)
