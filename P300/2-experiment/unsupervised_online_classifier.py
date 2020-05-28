from mne_realtime import LSLClient
import numpy as np
import mne as mne
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE, StreamInlet, resolve_byprop
from time import time, sleep
import threading
from scipy import stats
import sys
import itertools
from itertools import chain
import math
import warnings

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/chaklam/bci_project/BCI/P300/utils')
from filter import butter_bandpass, butter_bandpass_filter

#defining the marker
info = StreamInfo(name='ResultMarkerStream', type='ResultMarkers',
                  channel_count=1, channel_format='int8', nominal_srate=IRREGULAR_RATE,
                  source_id='resultmarker_stream', handle=None)

outlet = StreamOutlet(info)  #for sending the predicted classes

#defining interface parameters
num_rows = 6
num_cols = 6
min_markers = num_rows * num_cols

######HYPERPARAMETERS
epoch_width = 1  #second
sfreq = 250
waittime = 0.2 #overlapping period
tmin = int(sfreq * 0.2) 
tmax = int(sfreq * 0.5)
#########

markers = []

EEGData = []
EEGTime = []
Marker = []

scorelist = []

 
def acquireMarkersAndEEG():
    while True:
        eeg, eeg_time = eeg_input()
        marker, marker_time = marker_input()
        if eeg_time:
            eeg_time = [x + eeg_time_correction for x in eeg_time]
            EEGData.append(eeg)
            EEGTime.extend(eeg_time)
        if marker_time:
            marker_time += marker_time_correction
            Marker.append([marker,marker_time])            

def mapAndClassify(start_time):
    end_time = start_time + epoch_width
    sleep(epoch_width - waittime + 0.1)  #wait for all data; 0.1 for possible delay hacking

    #1. Getting trial EEG
    eegdata_copy = EEGData.copy()  #to avoid any change of data
    eegtime_copy = EEGTime.copy()
    eeg_time_numpy = np.array(eegtime_copy)
    eeg_numpy = np.concatenate(eegdata_copy, axis=0)
    #print(len(eeg), " should equal to ", len(eeg_timestamps))
    eeg_start_ix = np.argmin(np.abs(start_time - eeg_time_numpy))
    eeg_end_ix = np.argmin(np.abs(end_time - eeg_time_numpy))
    trialEEG = eeg_numpy[eeg_start_ix:eeg_end_ix+1]

    lowcut, highcut = 1, 30
    # print("Before filter: ", trialEEG[0])
    trialEEG = butter_bandpass_filter(trialEEG, lowcut, highcut, sfreq, order=6)
    # print("After filter: ", trialEEG[0])
    trialEEG_time = eeg_time_numpy[eeg_start_ix:eeg_end_ix+1]
    #print("Trial EEG Length: ", len(trialEEG))

    #2. Getting trial markers
    if(len(Marker)):  #if marker has arrived
        marker_copy = Marker.copy()
        marker_numpy = np.array(marker_copy)
        # print("Length Marker Numpy: ", marker_numpy.shape)
        marker_timestamps = marker_numpy[:, 1]
        marker_data = list(itertools.chain(*marker_numpy[:, 0]))
        marker_start_ix = np.argmin(np.abs(start_time - marker_timestamps))
        marker_end_ix = np.argmin(np.abs(end_time - marker_timestamps))
        trialMarkers = marker_data[marker_start_ix:marker_end_ix+1]
        trialMarkers_time = marker_timestamps[marker_start_ix:marker_end_ix+1]
        print("Trial Markers: ", trialMarkers)

        markerMapped = [0] * len(trialEEG)
        marker_ind = []

        #3. Mapping markers to EEG
        for i, (time) in enumerate(trialMarkers_time):
            ix = np.argmin(np.abs(time - trialEEG_time))
            markerMapped[ix] = trialMarkers[i]
            marker_ind.append(ix)

        #print("Marker_index: ", marker_ind)

        #4. Making epochs of size with tmin {} tmax {} of: ".format(tmin,tmax))
        epochArray = []  #combining eegs and corresponding marker that has already been windowed

        for i, (index) in enumerate(marker_ind):
            if(index < (len(trialEEG) - tmax)):
                eeg = trialEEG[index+tmin:index+tmax]
                epochArray.append([eeg, trialMarkers[i]])
                #print("Created epoch from eeg sample {} to {} for marker# {}... ".format(index+tmin, index+tmax, trialMarkers[i]))


        #5. Classifying
        mean_var = []
        epochnp = np.array(epochArray)

        #print("Epoch array: ", epochnp)

        #loop through 36 markers start from 1 to 36
        if(epochnp.size > 0):
            for i in np.arange(1,37):
                cond = epochnp[:, 1] == i
                selected_epochs_eeg =epochnp[cond][:, 0]
                #flat out the lists of list, and take mean
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    flat_list = list(chain(*selected_epochs_eeg))
                    mean = np.mean(np.mean(flat_list, axis=0))  #mean across all samples across all channels
                    var = np.mean(np.var(flat_list, axis=0))    #var across all samples across all channels   
                    mean_var.append([mean, var])
    
            print("Mean_var: ", mean_var)
            scores = fisher_criterion_score(mean_var)

            print("Epoch score: ", scores)



            scorelist.append(scores)  #convert any nan to zero

            # #wait until we got at least 10 scores
            if(len(scorelist) > 4):
                # print("Scorelist: ", scorelist[:10:])
                res = np.nanmean(np.nan_to_num(scorelist[-5:]), axis=0)
                print("Last 10 epoch scores mean: ", res)
                candidate = np.argmax(res)
                score = res[candidate]
                temp = np.delete(res, candidate)
                nextcandidate = np.argmax(temp)
                score_next = temp[nextcandidate]
                ratio = score / score_next


                print("---------------------------------------------------------------------------------")
                print("1st score: ", score, "Index: ", candidate, "; Letter: ", pos_to_char(candidate))
                print("2nd score: ", score_next, "Index: ", nextcandidate, "; Letter: ", pos_to_char(nextcandidate))
                print("Ratio diff: ", ratio)
                if(ratio > 2):    
                    # outlet.push_sample([candidate])
                    print("Pushed ", pos_to_char(candidate), " to LSL")
                # if the ratio is still small send all possible targets to recalculate (more than average)
                else:
                    # get average of the res
                    avg_score = np.mean(res)

                    # create filter average function
                    def filter_average(res_score):
                        return res_score > avg_score

                    # get candidate list
                    candidate_list = list(filter(filter_average, res))
                    # push back the list
                    print("Pushed ", pos_to_char(candidate_list), " to LSL")
        else:
            print("Waiting for more markers....")


'''
TBD:  Currently we are using simple fisher criterion which appears ineffective.
It is probably wise to first try out Verhoeven online method:
 T.   Verhoeven,   D.   H ̈ubner,   M.   Tangermann,   K.-R.   M ̈uller,J.  Dambre,  and  P.-J.  Kindermans,  
 “Improving  zero-trainingbrain-computer interfaces by mixing model estimators,”Journalof Neural Engineering, vol. 14, no. 3, p. 036021, Apr. 2017
'''
def fisher_criterion_score(mean_var):
    fisher_criterion_score = []
    for i, (mean_i, var_i) in enumerate(mean_var):
        mean_var_temp = mean_var.copy()
        mean_var_temp.pop(i)  #pop in place
        each_marker_score = []
        for j, (mean_j, var_j) in enumerate(mean_var_temp):
             if not (np.isnan(mean_i) or np.isnan(mean_j)) and not (var_i == 0  or var_j == 0):
                each_marker_score.append( np.abs(mean_i - mean_j) ** 2 / (var_i + var_j) )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fisher_criterion_score.append(np.mean(each_marker_score))

    return fisher_criterion_score

def marker_input():
    marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
    return marker, timestamp

def eeg_input():
    eeg, timestamp = inlet.pull_chunk(timeout=0.0)
    return eeg, timestamp

def pos_to_char(pos):
    return chr(pos+ 97)

#start code
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise(RuntimeError, "Cant find EEG stream")

print("Start aquiring data")
inlet = StreamInlet(streams[0])
eeg_time_correction = inlet.time_correction()

print("looking for a Markers stream...")
marker_streams = resolve_byprop('name', 'LetterMarkerStream')
if marker_streams:
    inlet_marker = StreamInlet(marker_streams[0])
    marker_time_correction = inlet_marker.time_correction()
    print("Found Markers stream")
    print("You can now press Start on the speller")

acquire = threading.Thread(target = acquireMarkersAndEEG,args=())
acquire.start() # start reading the eeg stream 

count = 0
while True:
    start_time = local_clock()
    sleep(waittime)
    start = threading.Thread(target = mapAndClassify,args=(start_time,))
    start.start() # start reading the eeg stream 