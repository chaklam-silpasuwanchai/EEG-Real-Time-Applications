import matplotlib.pyplot as plt

from mne.datasets import sample
from mne.io import  RawArray

from mne_realtime import LSLClient, MockLSLStream, RtEpochs, MockRtClient
import numpy as np
import pandas as pd
import mne as mne
from pylsl import StreamInlet, resolve_byprop

# Load a file to stream raw data
df = pd.read_csv('data/ssvep-20trials-3s-chaky-bigsquare.csv')
df.rename(columns={'Unnamed: 1':'O1',
                          'Unnamed: 2':'Oz',
                          'Unnamed: 3':'O2'
                      }, 
                 inplace=True)
df = df.drop(["timestamps", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], axis=1)
print(df)

# convert dataframe to raw array
data = df.to_numpy()
data = data.transpose()
print(data)

sfreq = 250
#Channel name
ch_names = ['O1', 'Oz', 'O2','Marker']
ch_types = ['eeg','eeg','eeg','misc']
# Create the info structure needed by MNE
info = mne.create_info(ch_names, sfreq,ch_types=ch_types)

# Finally, create the Raw object
raw = mne.io.RawArray(data, info)
# print (raw.times)

host = 'mne_stream'
# stream = MockLSLStream(host, raw, 'eeg')
stream = resolve_byprop('type', 'EEG', timeout=2)

epoch = ''
# Let's observe it
# plt.ion()  # make plot interactive
# _, ax = plt.subplots(1)
    
with LSLClient(info=raw.info, host=host, wait_max=3) as client:
    stream.start()
    client_info = client.get_measurement_info()
    print(client_info)
    sfreq = int(client_info['sfreq'])
# #     print(client_info)


#     # let's observe ten seconds of data
    for ii in range(10):
        # plt.cla()
        epoch = client.get_data_as_epoch(n_samples=sfreq)
        print(epoch.getData())
        # epoch.average().plot(axes=ax)
        # plt.pause(1)
    
# Let's terminate the mock LSL stream
stream.stop()