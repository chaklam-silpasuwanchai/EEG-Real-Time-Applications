# %%
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import pandas as pd
import time
from psychopy import visual, core, event
from glob import glob
from random import choice, random
from psychopy.visual import ShapeStim

# %%
#name, type, channel_count, sampling rate, channel format, source_id
#Note that Markers, 1, and 0.0 cannot be altered
info = StreamInfo('CytonMarkers', 'Markers', 1, 0.0, 'int32', 'CytonMarkerID')
#make an outlet
outlet = StreamOutlet(info)
markernames = [1, 2, 3]  #according to hz condition; 0 reserved for non-event

# %%
#define a function to get frame on and off

#Assuming that you use a 60 Hz monitor:

#8.57 Hz corresponds to 60/8.57 = 7 frames on and 7 frames off.
#10 Hz --> 6 frames
#12 Hz --> 5 frames
#15 Hz --> 4 frames

#here assuming that the frequency is frequency of SHIFTS (8.57 shifts per seconds) 
#rather than CYCLES (8.57 on and offs per second) since the latter would be 
#impossible on a 60 Hz monitor, because you should then change the image between frame 3 and 4.

import math
def getFrames(freq):
    framerate = 60 # mywin.getActualFrameRate()
    frame = int(round(framerate / freq))
    frame_on = math.ceil(frame / 2)
    frame_off = math.floor(frame / 2)
    return frame_on, frame_off

# %%
def one_stimuli_blinking(frame_on, frame_off, pattern1, pattern2):
    while trialclock.getTime()<soa:
        pattern1.setAutoDraw(True)

        for frameN in range(frame_on):
            mywin.flip()

        pattern1.setAutoDraw(False)
        pattern2.setAutoDraw(True)

        for frameN in range(frame_off):
            mywin.flip()
        pattern2.setAutoDraw(False)

# %%
def three_stimuli_blinking(frame_on1, frame_off1, frame_on2, frame_off2, frame_on3, frame_off3, shapes, flipCount,count):
    looptime = math.gcd(frame_on1,math.gcd(frame_on2,frame_on3))
    
    #reset clock for next trial
    trialclock.reset()   
    while trialclock.getTime()<soa:
        #if count% freq_len ==0:
            if(flipCount == 0 or (flipCount%frame_on1 ==0 and flipCount%(frame_on1+frame_on1) !=0)):
                shapes[0].setAutoDraw(True)
                shapes[1].setAutoDraw(False)
            if(flipCount%(frame_off1+frame_off1) ==0):
                shapes[1].setAutoDraw(True)
                shapes[0].setAutoDraw(False)

        #if count% freq_len ==1:
            if(flipCount == 0 or(flipCount%frame_on2 ==0 and flipCount%(frame_on2+frame_on2) !=0)):
                shapes[2].setAutoDraw(True)
                shapes[3].setAutoDraw(False)
            if(flipCount%(frame_off2+frame_off2) ==0):
                shapes[3].setAutoDraw(True)
                shapes[2].setAutoDraw(False)
        #if count% freq_len ==2:
            if(flipCount == 0 or(flipCount%frame_on3 ==0 and flipCount%(frame_on3+frame_on3) !=0)):
                shapes[4].setAutoDraw(True)
                shapes[5].setAutoDraw(False)
            if(flipCount%(frame_off3+frame_off3) ==0):
                shapes[5].setAutoDraw(True)
                shapes[4].setAutoDraw(False)

            for frameN in range(looptime):
                mywin.flip()
                flipCount+=1
    shapes[0].setAutoDraw(False)
    shapes[1].setAutoDraw(False)
    shapes[2].setAutoDraw(False)        
    shapes[2].setAutoDraw(False)
    shapes[3].setAutoDraw(False)
    shapes[4].setAutoDraw(False)
    shapes[5].setAutoDraw(False)

# %%
#setting params
mywin = visual.Window([1920, 1080], fullscr=False)

soa = 3  #stimulus onset asynchrony
iti = 1  #inter trial interval

trials_no = 20
test_freq = [6, 10, 16]  #, 15]
stimuli_seq = [0,1,2] * trials_no  #five trials for each freq in test_freq
freq_len = len(test_freq)

frame_on1, frame_off1 = getFrames(test_freq[0])
frame_on2, frame_off2 = getFrames(test_freq[1])
frame_on3, frame_off3 = getFrames(test_freq[2])

#print(getFrames(16))

count = 0
trialclock = core.Clock()

patternup1Pos = [0, 0.65]
patternright1Pos = [0.65, -0.5]
patternleft1Pos =[-0.65, -0.5]

# Arrow position is now y+0.2
arrowUp1Pos = [0, 0.85]
arrowRigh1Pos = [0.65, -0.3]
arrowLeft1Pos=[-0.65, -0.3]

# array to identify the sequence of the stimuli
arrowSequence = [arrowUp1Pos,arrowRigh1Pos,arrowLeft1Pos]

patternup1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern1', autoLog=False, color=[1,1,1], pos=patternup1Pos)
patternup2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern2', autoLog=False, color=[-1,-1,-1], pos=patternup1Pos)

patternright1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern1', autoLog=False, color=[1,1,1], pos=patternright1Pos)
patternright2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern2', autoLog=False, color=[-1,-1,-1], pos=patternright1Pos)

#patterndown1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.3,
#    name='pattern1', autoLog=False, color=[1,1,1], pos=(0, -0.5))
#patterndown2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.3,
#    name='pattern2', autoLog=False, color=[-1,-1,-1], pos=(0, -0.5))

patternleft1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern1', autoLog=False, color=[1,1,1], pos=patternleft1Pos)
patternleft2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
    name='pattern2', autoLog=False, color=[-1,-1,-1], pos=patternleft1Pos)

# prepare the arrow shape
arrowVert = [(0,0),(-0.1,0.15),(-0.05,0.15),(-0.05,0.3),(0.05,0.3),(0.05,0.15),(0.1,0.15)]

shapes = [patternup1, patternup2, patternright1, patternright2, patternleft1, patternleft2]

#fixation cross
fixation = visual.ShapeStim(mywin, 
    vertices=((0, -0.5), (0, 0.5), (0,0), (-0.5,0), (0.5, 0)),
    lineWidth=5,
    closeShape=False,
    lineColor="white"
)


# %%
#running the actual experiment
while True:
    message = visual.TextStim(mywin, text='Start recording and press space to continue')
    message.draw()
    mywin.flip()
    keys = event.getKeys()
    
    if 'space' in keys:  # If space has been pushed
        message.setText = ''
        message.draw()
        mywin.flip()  
        
        fixation.draw()
        mywin.flip() #refresh
        core.wait(iti)
        mywin.flip()
        
        # create arrow shape for the first sequence
        arrow = ShapeStim(mywin, vertices=arrowVert, fillColor='darkred', size=.5, lineColor='red', pos=arrowSequence[0])
        arrow.setAutoDraw(True)
        mywin.flip()
        core.wait(iti)
        mywin.flip()
        
        while count < len(stimuli_seq):
            print("Count: ", count)
            
            #draw the stimuli and update the window
            print("freq: ", test_freq[count%freq_len])
            #print("frameon-off: ", frame_on, frame_off)
            print("markername: ", markernames[count%freq_len])
            print("======")
            
            outlet.push_sample([markernames[count%freq_len]])  #(x, timestamp)
            
            flipCount = 0
            #one_stimuli_blinking(frame_on, frame_off, shapes[count%freq_len*2], shapes[count%freq_len*2+1])
            three_stimuli_blinking(frame_on1, frame_off1, frame_on2, frame_off2, frame_on3, frame_off3, shapes, flipCount,count)
            
            # close the finish arrow
            arrow.setAutoDraw(False)
            
            # draw the next arrow
            arrow = ShapeStim(mywin, vertices=arrowVert, fillColor='darkred', size=.5, lineColor='red', pos=arrowSequence[(count+1)%freq_len])
            arrow.setAutoDraw(True)
            
            #clean black screen off
            mywin.flip()
            #wait certain time for next trial
            core.wait(iti)
            #clear fixation
            mywin.flip() 
            #count number of trials
            count+=1

            
            if 'escape' in event.getKeys():
                core.quit()

        break;
            
mywin.close()  #do not delete, otherwise, the window will not turn off+

# %%


# %%
