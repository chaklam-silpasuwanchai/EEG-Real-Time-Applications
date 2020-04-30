from mne_realtime import LSLClient
import numpy as np
import mne as mne
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE, StreamInlet, resolve_byprop
from time import time, sleep
import threading
from scipy import stats
import sys

#defining the marker
info = StreamInfo(name='ResultMarkerStream', type='ResultMarkers',
                  channel_count=1, channel_format='int8', nominal_srate=IRREGULAR_RATE,
                  source_id='resultmarker_stream', handle=None)

outlet = StreamOutlet(info)  #for sending the predicted classes

#defining the stream data format
sfreq = 250
ch_names = ['P4', 'Pz', 'P3', "PO4", "POz", "PO3", "O2", "O1"]
ch_types = ['eeg'] * 8
info = mne.create_info(ch_names, sfreq, ch_types = ch_types)
host = 'OpenBCItestEEG'

#defining interface parameters
num_rows = 6
num_cols = 6
min_markers = num_rows * num_cols

######HYPERPARAMETERS
epoch_width = 3  #2.6
waittime = 0.2
tmin = int(sfreq/1000 * 200) # start from  sfreq / total milisec   *  milisec
tmax = int(sfreq/1000 * 500) # end at      sfreq / total milisec   *  milisec
#########

markers = []
epoch_chunks = int(np.round(sfreq * epoch_width))  #freq * 2.6seconds

arrayEEGData =[]  #for keeping the incoming EEG data
arrayMarkerData = []  #incoming marker data
epochArray = []  #combining eegs and corresponding marker that has already been windowed
scores = []  #

def start(count):
    x= threading.Thread(target = startEEG, args=())
    y= threading.Thread(target = startMarker, args=())
    x.start()
    y.start()
    x.join()
    y.join()
    mapMarkerToEEG(count)
    epochArray = makeEpochs(count)
    classify(epochArray)

def startEEG():
    eeg, eeg_time = eeg_input()
    arrayEEGData.append([eeg,eeg_time])
   
def startMarker():
    marker, timestamp = marker_input()
    arrayMarkerData.append([marker, timestamp])

def mapMarkerToEEG(count):
   
    # 0 -> EEG Data
    # 1 -> Timestamp
    # 2 -> markerData
    #create marker data container
    emptyList = [0]* len(arrayEEGData[count][0])
    arrayEEGData[count].append(emptyList)  #add the third dimension [2] as marker

    timestamps = np.array(arrayEEGData[count][1])

    for i in range(len(arrayMarkerData[count][0])):
        
        # market time at i
        markerTime = arrayMarkerData[count][1][i] 
        markerData  = arrayMarkerData[count][0][i][0]   #marker is a list, so require [0]

        # find index of marker by finding the smallest differences of one marker against all EEG timestamps
        ix = np.argmin(np.abs(markerTime - timestamps))
        
        #assign marker data to the closest time
        arrayEEGData[count][2][ix] = markerData

def makeEpochs(count):

    score = np.zeros(36)

    for i in range(len(arrayEEGData[count][2])-tmax):   #-tmax so we make sure this data can be windowed
        markerData = arrayEEGData[count][2][i]

        #any eeg data containing marker !=0 from tmin to  tmax is extracted
        if(markerData>0):
            eegData = arrayEEGData[count][0][i+tmin:i+tmax]
            epochArray.append([eegData,markerData])

def classify(epochArray):

    #epochArray = [ [many set of 8 channel eeg data], marker]
    #first extract the mean of variances
    for i in range(len(epochArray)):
        markerPos = epochArray[i][1]-1  #make it 0 by -1
        var = np.var(epochArray[i][0],axis=0)  #get variance along columns
        mean = np.mean(var) #get total mean 

        if(score[markerPos]==0):
            score[markerPos] = mean
        else:
            score[markerPos] = (score[markerPos]+mean)/2

    scores.append(score)

    #wait until we got at least 10 scores
    if(len(scores)>10):
        res = np.mean(epochArrayTestAll[-10:],axis=0)  #get the last 10 results
        candidate = np.argmax(res)
        print("Candidate Letter: ", pos_to_char(np.argmax(res)))  #find the maximum scores and convert to char
        #if it pass certain threshold, will 
            #outlet.push_sample([candidate])

def marker_input():
    marker, timestamp = inlet_marker.pull_chunk(timeout=1, max_samples=100)
    return marker, timestamp

def eeg_input():
    with LSLClient(info=info, host=host, wait_max=2) as client:
        eeg, timestamp =  client.client.pull_chunk(timeout=1, max_samples=epoch_chunks)
        return eeg, timestamp

def pos_to_char(pos):
    return chr(pos+ 97)


#start code
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise(RuntimeError, "Cant find EEG stream")

print("Start aquiring data")
inlet = StreamInlet(streams[0], max_chunklen=epoch_chunks)
eeg_time_correction = inlet.time_correction()

print("looking for a Markers stream...")
marker_streams = resolve_byprop('name', 'LetterMarkerStream')
if marker_streams:
    inlet_marker = StreamInlet(marker_streams[0])
    marker_time_correction = inlet_marker.time_correction()
    print("Found Markers stream")

count = 0
while(True):
    sleep(waittime)
    start(count)
    count+=1