from mne_realtime import LSLClient
import numpy as np
import mne as mne
from pylsl import StreamInfo, StreamOutlet
import time
import threading
from scipy import stats
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/chaklam/bci_project/BCI/SSVEP/utils')
from fbcca import fbcca_realtime

#defining the marker
info = StreamInfo('MarkerStream', 'Markers', 1, 0, 'int32', 'myuidw43536')
outlet = StreamOutlet(info)  #for sending the predicted classes
outlet.push_sample([0])  #initialized the stream so others can find

#defining the stream data format
sfreq = 250
ch_names = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8"]
ch_types = ['eeg'] * 8
info = mne.create_info(ch_names, sfreq, ch_types = ch_types)
host = 'OpenBCItestEEG'

######HYPERPARAMETERS
epoch_width = 3.5  #2.6
waittime = 0.2
threshold = 4  #3 out of 5 epochs
num_of_epochs = 5
#########

epoch_chunks = int(np.round(sfreq * epoch_width))  #freq * 2.6seconds

a = time.time()
arrayData = []  #for keeping the results

def runRT(count):
	with LSLClient(info=info, host=host, wait_max=3) as client:
		#print("data to array:",count, "  Start Time:",a-time.time())

		epoch = client.get_data_as_epoch(n_samples=epoch_chunks)
		#print("data to array:",count, "  End Time:",a-time.time())
		epoch.filter(4, 77, method='iir')
		X = epoch.get_data() *1e-6  #n_epochs * n_channel * n_time_samples
		X=X[:,[0,1,2],:]	#get only the first three channels

		#parameters
		list_freqs=[6, 11, 15]
		fs = 250
		num_harms = 5
		num_fbs = 5

		res = fbcca_realtime(X, list_freqs, fs, num_harms, num_fbs)
		arrayData.append([res])
		maxArray = len(arrayData)
		#print("Number of epochs:     ",maxArray)
		
		if (maxArray > num_of_epochs):  #make sure there is at least n number of epochs
			lastFive = np.array(arrayData[maxArray-num_of_epochs:maxArray][:]).transpose()  #tranpose the data
			print("Last Five: ", lastFive)
			selectedRes, modeCount  = stats.mode(lastFive[0])   #get the most recurrent classes
			print("Result:  ",selectedRes[0])
			print("Count: ", modeCount[0])
			print("Threshold: ", threshold)
			print(modeCount[0] > threshold)

			if modeCount[0] > threshold:
				print("pushed")
				selectedRes = int(selectedRes[0])
				outlet.push_sample([selectedRes])

	    
count = 0
while(True):
	time.sleep(waittime)
	x= threading.Thread(target = runRT, args=(count,))
	x.start()
	count+=1
