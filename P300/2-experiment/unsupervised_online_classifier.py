from mne_realtime import LSLClient
import numpy as np
import mne as mne
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE, StreamInlet, resolve_byprop
from time import time, sleep
import threading
from scipy import stats
import sys
from itertools import chain
import math

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
epoch_width = 2  #2.6
waittime = 0.2
tmin = int(sfreq/1000 * 200) # start from  sfreq / total milisec   *  milisec
tmax = int(sfreq/1000 * 500) # end at      sfreq / total milisec   *  milisec
#########

markers = []
epoch_chunks = int(np.round(sfreq * epoch_width))  #freq * 2.6seconds

arrayEEGData =[]  #for keeping the incoming EEG data
arrayMarkerData = []  #incoming marker data
# testing
newEEGData = []
newEEGTime = []
countIndex = []
#----
scorelist = []

def start(count):
    time_start = time()
    # x= threading.Thread(target = startEEG, args=(time_start,))
    y= threading.Thread(target = startMarker, args=(time_start,))
    # x.start()
    y.start()
    # x.join()  #wait until x and y finishes
    # y.join()
    # mapMarkerToEEG(count)
    # classify(makeEpochs(count))

def startEEG(time_start,):
    # print(count,": Start EEG",time_start-time())
    eeg, eeg_time = eeg_input()
    #eeg = [ [eeg sample1], [eeg sample 2], .....250]
    #eeg_time [ [eeg sample time], [eeg sample time 2], .....250]
    # print("EEG received length: ", len(eeg))
    # print("EEG time received length: ", len(eeg_time))
    arrayEEGData.append([eeg,np.add(eeg_time,eeg_time_correction)])
    

    print(count,": End EEG",time_start-time())
    #testing
    # global newEEGData 
    # global newEEGTime
    # newEEGData += eeg
    # newEEGTime +=eeg_time

def startEEGTest():

    while True:
        eeg, eeg_time = eeg_input()
        global newEEGData 
        global newEEGTime
        # newEEGData += eeg
        if eeg_time:
            newEEGTime +=eeg_time
            print("newEEGTime",len(newEEGTime))
    # print(count,": Start EEG",time_start-time())
   
    # #eeg = [ [eeg sample1], [eeg sample 2], .....250]
    # #eeg_time [ [eeg sample time], [eeg sample time 2], .....250]
    # print("EEG received length: ", len(eeg))
    # print("EEG time received length: ", len(eeg_time))
    # arrayEEGData.append([eeg,np.add(eeg_time,eeg_time_correction)])
    

    # print(count,": End EEG",time_start-time())
    #testing
   

def startMarker(time_start,):
    # print("eeg_time_correction: ",eeg_time_correction)
    # sleep(math.abs(eeg_time_correction ))
    # print(count,": Start marker",time_start - time())
    marker, timestamp = marker_input()
    # print("Marker received: ", marker)
    # print("eeg_time_correction: ",eeg_time_correction)
    # print("marker_time_correction: ",marker_time_correction)
    arrayMarkerData.append([marker,timestamp])
    # print(count,": Stop marker",time_start - time())
    # print(count,": Stop Marker",time_start - time())

def mapMarkerToEEG(count):


    print("Mapping marker to EEG....")
   
    #arrayEEGData
    # 0 -> EEG Data
    # 1 -> Timestamp
    # 2 -> markerData
    #arrayMarkerData
    # 0 -> markerdata
    # 1 -> timestamp
    #create marker data container
    emptyList = [0]* len(arrayEEGData[count][0])
    arrayEEGData[count].append(emptyList)  #add the third dimension [2] as marker
    
    timestamps = np.array(arrayEEGData[count][1])

    print("First 5 EEG timestamps: ", timestamps[0:5])
    print("Last 5 EEG timestamps: ", timestamps[-5:])
    countIndex.append([0,0])

    if(len(arrayMarkerData[count][1]) > 10):
        print("Start 5 marker timestamps: ", arrayMarkerData[count][1][0:5])
        print("Start offset: ", timestamps[0:5] - arrayMarkerData[count][1][0:5])
        print("End 5 marker timestamps: ", arrayMarkerData[count][1][-5:])
        print("End offset: ", timestamps[-5:] - arrayMarkerData[count][1][-5:])

        
    if(len(arrayMarkerData[count][1])>0):
        for i in range(len(arrayMarkerData[count][0])):
            #print("i: ", i)
            
            # market time at i
            markerTime = arrayMarkerData[count][1][i] 
            markerData  = arrayMarkerData[count][0][i][0]   #marker is a list, so require [0]

            # find index of marker by finding the smallest differences of one marker against all EEG timestamps
            ix = np.argmin(np.abs(markerTime - timestamps))
            
            print("Maker: ", markerData, " got mapped to index: ", ix)

            #assign marker data to the closest time
            arrayEEGData[count][2][ix] = markerData

            # print("EEG data mapped to marker: ", arrayEEGData[count][0][ix])
            # print("EEG timestamp mapped to marker: ", arrayEEGData[count][1][ix])
            
            #testing 
            # print("newEEGTime----------",newEEGTime)
            ix = np.argmin(np.abs(markerTime - np.array(newEEGTime)))
           
            # get start index
            if i==1:
                countIndex[count][0]=ix

            if i == len(arrayMarkerData[count][0])-1:
                countIndex[count][1]=ix

    print("countIndex-",count," : ",countIndex[count])


    # for i in range(len(arrayMarkerData[count][0])):

    #     # market time at i
    #     markerTime = arrayMarkerData[count][1][i]
    #     markerData  = arrayMarkerData[count][0][i][0]   #marker is a list, so require [0]

    #     # find index of marker by finding the smallest differences of one marker against all EEG timestamps
    #     ix = np.argmin(np.abs(markerTime - timestamps))

    #     #assign marker data to the closest time
    #     arrayEEGData[count][2][ix] = markerData

def makeEpochs(count):

    epochArray = []  #combining eegs and corresponding marker that has already been windowed

    print("Making epochs....")

    for i in range(len(arrayEEGData[count][2])-tmax):   #-tmax so we make sure this data can be epoched
        markerData = arrayEEGData[count][2][i]
        #any eeg data containing marker !=0 from tmin to  tmax is extracted
        if(markerData>0):
            print("Created epoch for marker# : ", markerData)
            eegData = arrayEEGData[count][0][i+tmin:i+tmax]
            epochArray.append([eegData,markerData])

    return epochArray

#epochArray = [ [eeg, marker], [eeg, marker] ] where eeg = [[8 data], [8 data]]
def classify(epochArray):

    print("Classifying....")

    print("Epoch array length (i.e., how many event markers): ", len(epochArray))

    mean_var = []

    epochnp = np.array(epochArray)

    #loop through 36 markers start from 1 to 36
    if(epochnp.size > 0):
        for i in np.arange(1,37):
            cond = epochnp[:, 1] == i
            selected_epochs_eeg =epochnp[cond][:, 0]
            #flat out the lists of list, and take mean
            flat_list = list(chain(*selected_epochs_eeg))
            mean = np.mean(flat_list, axis=0)  #mean across all samples  
            var = np.var(flat_list, axis=0)    #var across all samples        
            mean_var.append([mean, var])
            
        scores = []
        for i, (mean, var) in enumerate(mean_var):
            m = np.mean(mean)  #mean across all channels
            v = np.mean(var)
            if not(np.isnan(v)):
                scores.append(v)  #simply use variances as scores (we may want to use more fancy disrimant analysis later)
            else:
                scores.append(0)

        print("Epoch score: ", scores)

        scorelist.append(scores)

        # #wait until we got at least 10 scores
        if(len(scorelist) > 9):
            res = np.mean(scorelist[-10:], axis=0)
            print("Last 10 epoch scores mean: ", res)
            candidate = np.argmax(res)
            print("Candidate index (argmax): ", candidate)
            print("Candidate Letter: ", pos_to_char(np.argmax(res)))  #find the maximum scores and convert to char
        #     #if it pass certain threshold, will 
        #         #outlet.push_sample([candidate])
    else:
        print("Waiting for more markers....")
    #epochArray = [ [many set of 8 channel eeg data], marker]
    #first extract the mean of variances
    # for i in range(len(epochArray)):
    #     markerPos = epochArray[i][1]-1  #make it 0 by -1
    #     var = np.var(epochArray[i][0],axis=0)  #get variance along columns
    #     mean = np.mean(var) #get total mean

    #     if(score[markerPos]==0):
    #         score[markerPos] = mean
    #     else:
    #         score[markerPos] = (score[markerPos]+mean)/2

    # scores.append(score)

    # #wait until we got at least 10 scores
    # if(len(scores)>10):
    #     res = np.mean(epochArrayTestAll[-10:],axis=0)  #get the last 10 results
    #     candidate = np.argmax(res)
    #     print("Candidate Letter: ", pos_to_char(np.argmax(res)))  #find the maximum scores and convert to char
    #     #if it pass certain threshold, will
    #         #outlet.push_sample([candidate])

def marker_input():
    marker, timestamp = inlet_marker.pull_chunk(timeout=2.5, max_samples=10000)
    return marker, timestamp

def eeg_input():

    # with LSLClient(info=info, host=host, wait_max=10) as client:
    eeg, timestamp = inlet.pull_chunk(timeout=0.0)
    # print("Time stamp----------------------------------- ",timestamp)
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
marker_time_correction =0
print("looking for a Markers stream...")
marker_streams = resolve_byprop('name', 'LetterMarkerStream')
if marker_streams:
    inlet_marker = StreamInlet(marker_streams[0])
    marker_time_correction = inlet_marker.time_correction()
    print("Found Markers stream")

count = 0
runStart = threading.Thread(target = startEEGTest,args=())
runStart.start() # start reading the eeg stream 

while True:
      sleep(waittime)
      start(count)
      count+=1
# while(True): 
#     sleep(waittime)
#     start(count)
#     count+=1