#!/usr/bin/env python
## code by Alexandre Barachant
import numpy as np
import pandas as pd
from time import time, sleep
from pylsl import StreamInlet, resolve_byprop
import sys
sys.path.insert(1, '/home/chanapa/BCI_fork/BCI/P300/utils')

import helper as helper #custom
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from mne.decoding import Vectorizer  

################# INITIALIZE EEG RECORDING ##################  
def eeg_record(workQueue , target = True):
    
    # model path
    pkl_filename = '../../3-analysis/sit_Xdawn + RegLDA_cb_model.pkl'

    # dejitter timestamps
    dejitter = False

    print("looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise(RuntimeError, "Cant find EEG stream")
    print("Start aquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    
    sleep(10)
    
    print("looking for a Markers stream...")
    marker_streams = resolve_byprop('type', 'Markers', timeout=2)
    if marker_streams:
        inlet_marker = StreamInlet(marker_streams[0])
        marker_time_correction = inlet_marker.time_correction()
        print("Found Markers stream !")
    else:
        inlet_marker = False
        print("Cant find Markers stream")

    info = inlet.info()
    description = info.desc()

    freq = info.nominal_srate()
    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    ######################### DEFINE SKLEARN MODEL ###########################
    lda = LDA(shrinkage='auto', solver='eigen') #Regularized LDA
    n_components = 3    
    model = make_pipeline(XdawnCovariances(n_components, 
                                estimator='oas'), Vectorizer(), lda)    
    # Load from file
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        print('Successfully loaded the model !')
    t_init = time()
    print('Start recording at time t=%.3f' % t_init)

    ######################### GET EEG DATA & PREDICT ###########################
    count_letter = 0
    while count_letter < 6 :
        res = []
        timestamps = []
        markers = []
        m = False
        while True:
            data, timestamp = inlet.pull_chunk(timeout=1.0,
                                            max_samples=12)
            if timestamp:
                res.append(data)
                timestamps.extend(timestamp)

            # marker
            if inlet_marker:
                marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                if marker != None:
                    print(timestamp, marker)
                    # if 88 in marker:
                    #     print("Found Start flash")
                    if 99 in marker:
                        m = True    
            if timestamp:
                markers.append([marker, timestamp])

            res_np = np.concatenate(res, axis=0) # shape = (len(timestamps), num_channels)
            timestamps_np = np.array(timestamps) # eeg timestamps | shape = (len(timestamps), )

            ch_names = ['Fz', 'F3','C4', 'Cz','Pz', 'P3', 'O2', 'O1']
            res_for_df = np.c_[timestamps_np, res_np] # shape = (len(timestamps), num_channel+1)
            data = pd.DataFrame(data=res_for_df, columns=['timestamps'] + ch_names)
            
            data['Marker'] = 0
            # process markers:
            count_m = 0
            ix_ = []
            for marker in markers[:-1]:
                if marker == 99:
                    break
                # find index of markers
                ix = np.argmin(np.abs(marker[1] - timestamps_np))
                ix_.append(ix)
                data.loc[ix, 'Marker'] = marker[0][0]        
                count_m += 1

            if m == True:
                break
            
        # print("count_m ", count_m) # should get 121
        assert count_m == 121
        
        data['Marker'][ix_[0]] = 0
        df = data.iloc[ix_[0]:ix_[-1]+1000]
        df = df.drop(["timestamps"], axis=1)

        # print("="*20)
        # print(df['Marker'].value_counts()) # 1:110, 2:10
        # print("="*20)

        raw = helper.df_to_raw(df)
        raw.notch_filter(np.arange(50, 125, 50), filter_length='auto', phase='zero') #250/2 based on Nyquist Theorem
        raw.filter(1, 20, method='iir')

        if target :
            event_id = {'Non-Target': 1, 'Target' : 2}
        else :
            event_id = {'Non-Target': 1}
            
        tmin, tmax = 0.2, 0.5
        picks = 'eeg'
        epochs, drop_idx = helper.getEpochs(raw, event_id, tmin, tmax, picks)

        X = epochs.get_data()
        ix_ = ix_[1:]
        final_idx = [] # eeg index
        final_idx_x = [] # for x
        for i in range(len(ix_)):
            if i not in drop_idx:
                final_idx.append(ix_[i])
                final_idx_x.append(i)
                
        pred = model.predict(X)
        final_idx_x = np.array(final_idx_x)
        pred_out = final_idx_x[np.argwhere(pred==2)]
        # print(pred_out)  
        
        workQueue.put(pred_out)
        print("******** Put pred_out to Q successfully *********")
        # print("eeg_record QUEUE", workQueue)
        
        count_letter += 1