import mne
from mne import create_info
from mne.io import RawArray
from mne import Epochs, find_events

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
import os
import csv
import time
import inspect
import ast
import pickle
import json
from itertools import combinations


_is_debug = False



def is_equa_lenth(_var1, _var2):
    start_time   = time.time()
    _result = False
    if len(_var1) == len(_var2):
        _result = True
    if _is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

    return _result
#====================================================================
def df_to_raw(df): # get raw
    start_time   = time.time()
    sfreq = 125
    ch_names = list(df.columns)
    ch_types = ['eeg'] * (len(df.columns) - 1) + ['stim']
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    df = df.T  #mne looks at the tranpose() format
    df[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, verbose=False)

    raw = mne.io.RawArray(df, info)
    raw.set_montage(ten_twenty_montage)

    # Just plotting the raw data of its power spectral density
    # raw.plot_psd()

    if _is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

    return raw

#====================================================================

def eog_remove(_mne_raw_eeg, _ch_name, _threshold, _measure, is_debug=False ): 
    ## print ICA componant
    from mne.preprocessing import ICA

    #make copy of raw for later signal reconstruction
    filt_raw = _mne_raw_eeg.copy()

    # set up and fit the ICA
    # print(mne.sys_info())

    ica = ICA(n_components=15, random_state=32, max_iter ="auto", verbose=is_debug)

    
    ica.fit(filt_raw, verbose=True)
   
    # ica.plot_sources(filt_raw)
    # ica.plot_components()

    
    ica.exclude = []
    # find which ICs match the EOG pattern

    eog_indices, eog_scores = ica.find_bads_eog(filt_raw, ch_name=_ch_name , threshold = _threshold , measure= _measure,  verbose=is_debug)

    ica.exclude = eog_indices

    # barplot of ICA component "EOG match" scores
    # ica.plot_scores(eog_scores)

    # plot diagnostics
    # ica.plot_properties(_mne_raw_eeg, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    # ica.plot_sources(filt_raw, show_scrollbars=False)

    # ica.apply() changes the Raw object in-place, so let's make a copy first for comparison:
    orig_raw = _mne_raw_eeg.copy()  #we apply ica to raw

    ica.apply(_mne_raw_eeg, verbose=is_debug)



    regexp = r'(F)|(AF)'
    artifact_picks = mne.pick_channels_regexp(_mne_raw_eeg.ch_names, regexp=regexp)

    # orig_raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
    # _mne_raw_eeg.plot(order=artifact_picks, n_channels=len(artifact_picks))

    return _mne_raw_eeg

#====================================================================

def get_epochs(_raw, _event_id, _tmin, _tmax, _picks):
    _start_time   = time.time()
    #epoching
    _events = find_events(_raw)


    
    #reject_criteria = dict(mag=4000e-15,     # 4000 fT
    #                       grad=4000e-13,    # 4000 fT/cm
    #                       eeg=100e-6,       # 150 μV
    #                       eog=250e-6)       # 250 μV

    _reject_criteria = dict(eeg=100e-6)  #most voltage in this range is not brain components

    _epochs = Epochs(_raw, events=_events, event_id=_event_id, tmin=_tmin, tmax=_tmax, 
                    baseline=None, preload=True,verbose=True , picks=_picks)  # 15 channels
    print('sample drop %: ', (1 - len(_epochs.events)/len(_events)) * 100)

    if _is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-_start_time}')

    return _epochs
#====================================================================

def mne_to_numpy(_mne_raw_eeg, _event_id, _tmin, _tmax):
    print("******************************")
    print(type(_mne_raw_eeg))
    start_time   = time.time()
    _picks       = mne.pick_types(_mne_raw_eeg.info, eeg=True)


    _epochs  = get_epochs(_mne_raw_eeg, _event_id, _tmin, _tmax, _picks)
    _y       = _epochs.events[:, -1]
    _y       = _y - 1
    _X_eeg   = _epochs.get_data()

    if _is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')


    return _epochs, _X_eeg, _y



#====================================================================

def minmax_norm( _eeg ):
    # input eeg [bath, channel, data point]
    # norm each channel in each stimulus
    start_time   = time.time()
    print(_eeg.shape)
    _min = np.min(_eeg,axis=2).reshape( _eeg.shape[0],_eeg.shape[1],1 )
    _max = np.max(_eeg,axis=2).reshape( _eeg.shape[0],_eeg.shape[1],1 )
    _X_norm = (_eeg-_min) / ( _max-_min )
    
    if _is_debug:
        print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')
    
    return _X_norm

#====================================================================

def zscore( _eeg ):
    # input eeg [bath, channel, data point]
    # norm each channel in each stimulus
    mean = np.mean( _eeg , axis=2 )
    mean = mean[:,:, np.newaxis]
    sd   = np.std( _eeg, axis=2 )
    sd = sd[:,:, np.newaxis]

    _eeg = (_eeg-mean)/sd
    print(_eeg[0])
    
    return _eeg

