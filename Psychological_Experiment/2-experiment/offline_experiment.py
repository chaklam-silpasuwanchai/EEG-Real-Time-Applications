# %%
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import pandas as pd
import time
from psychopy import visual, core, event
from glob import glob
from random import choice, random

# %%
#name, type, channel_count, sampling rate, channel format, source_id
#Note that Markers, 1, and 0.0 cannot be altered
info = StreamInfo('CytonMarkers', 'Markers', 1, 0.0, 'int32', 'CytonMarkerID')

# %%
#make an outlet
outlet = StreamOutlet(info)

# %%
markernames = [1, 2]  #1 for non-Target, 2 for target; 0 is default for non-events

# %%
start = time.time()

# %%
#setup parameters
#this requires expertise and paper reading to know what are the appropriate parameters
n_trials = 100
iti = 1  #inter trial interval, i.e., how long the fixation will stay
soa = 3  #Stimulus-onset asynchrony, i.e., how long the stimulus will stay
jitter = 0.2
record_duration = np.float32(10000)

# %%
#setup appearance probability
position = np.random.binomial(1, 0.5, n_trials)  #randomize between 0 and 1, with 0.5 probability for 1, this is to set up the appearance of images

# %%
#position = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

# %%
trials = pd.DataFrame(dict(position=position, timestamp=np.zeros(n_trials)))

# %%
trials

# %%
def loadImage(filename):
    return visual.ImageStim(win=mywin, image=filename)

# %%
mywin = visual.Window([1920, 1080], fullscr=False)

targets = list(
        map(loadImage, glob('emo_stim/anger*.jpg')))  #map each file to loadImage(here) #sad is target
nontargets = list(
        map(loadImage, glob('emo_stim/amusement*.jpg')))


while True:
    message = visual.TextStim(mywin, text='Start recording and press space to continue')
    message.draw()
    mywin.flip()
    keys = event.getKeys()
    if 'space' in keys:  # If space has been pushed
        message.setText = ''
        message.draw()
        mywin.flip()
        for ii, trial in trials.iterrows():
            
            #fixation cross
            fixation = visual.ShapeStim(mywin, 
                vertices=((0, -0.5), (0, 0.5), (0,0), (-0.5,0), (0.5, 0)),
                lineWidth=5,
                closeShape=False,
                lineColor="white"
            )
            fixation.draw()
            mywin.flip() #refresh

            # inter trial interval
            core.wait(iti + np.random.rand() * jitter)
            mywin.flip() #clear fixation


            # onset
            pos = trials['position'].iloc[ii]  #running each position, using index as iterator (.iloc)
            image = choice(targets if pos == 1 else nontargets)  #if position == 1, randomly select images from targets
            image.draw()
            timestamp = time.time()
            outlet.push_sample([markernames[pos]])  #(x, timestamp)  #remind that markernames[1] is 2
            mywin.flip() #draw

            #offset
            core.wait(soa) #then wait
            mywin.flip()  #then clear frame automatically since nothing is drawn
            
            if 'escape' in event.getKeys():
                core.quit()

            if len(event.getKeys()) > 0 or (time.time() - start) > record_duration:
                break
            event.clearEvents()
        break;

mywin.close()  #do not delete, otherwise, the window will not turn off

# %%
mywin.close()

# %%
