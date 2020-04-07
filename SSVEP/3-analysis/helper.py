#Transform df into raw mne object

import mne
from mne import create_info
from mne.io import RawArray

def df_to_raw(df):
    sfreq = 250
    ch_names = list(df.columns)
    ch_types = ['eeg'] * (len(df.columns) - 1) + ['stim']
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    df = df.T  #mne looks at the tranpose() format
    df[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    raw = mne.io.RawArray(df, info)
    raw.set_montage(ten_twenty_montage)

    #try plotting the raw data of its power spectral density
    raw.plot_psd()

    return raw


'''
We will chunk (epoch) the data into segments representing the data "tmin" to ""tmax" after each stimulus. No baseline correction is needed (signal is filtered) and we will reject every epoch where the amplitude of the signal exceeded 100 uV, which should be mostly eye blinks in case our ICA did not detect them (it should, theoretically...right?).

**Sample drop % is an important metric representing how noisy our data set was**. If this is greater than 20%, consider ensuring that signal variances is very low in the raw EEG viewer and collecting more data
'''
from mne import Epochs, find_events

def getEpochs(raw, event_id, tmin, tmax):

    #epoching
    events = find_events(raw)
    
    #reject_criteria = dict(mag=4000e-15,     # 4000 fT
    #                       grad=4000e-13,    # 4000 fT/cm
    #                       eeg=100e-6,       # 150 μV
    #                       eog=250e-6)       # 250 μV

    reject_criteria = dict(eeg=100e-6)  #most frequency in this range is not brain components

    epochs = Epochs(raw, events=events, event_id=event_id, 
                    tmin=tmin, tmax=tmax, baseline=None, preload=True, 
                    reject=reject_criteria,verbose=False, picks=[0,1,2])  #8 channels
    print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

    return epochs