# %%
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import pandas as pd
import time 
from psychopy import visual, core, event
from glob import glob
from random import choice, random
from psychopy.visual import ShapeStim
from pylsl import StreamInlet, resolve_byprop
import threading

#%%
print("looking for a Markers stream...")
marker_streams = resolve_byprop('type', 'Markers')
if marker_streams:
    inlet_marker = StreamInlet(marker_streams[0])
    marker_time_correction = inlet_marker.time_correction()
    print("Found Markers stream")

# %%
import math
def getFrames(freq):
    framerate = 60 # mywin.getActualFrameRate()
    frame = int(round(framerate / freq))
    frame_on = math.ceil(frame / 2)
    frame_off = math.floor(frame / 2)
    return frame_on, frame_off

# %%
#define a marker stream to receive the predicted class
def marker_result():
    marker, timestamp = inlet_marker.pull_chunk()
    return marker

# %%

# %%
#Author: Apiporn Simapornchai

def stimuli_blinking_nonstop(frame_on1, frame_off1, frame_on2, frame_off2, frame_on3, frame_off3, shapes, arrow):
            
    looptime = math.gcd(frame_on1,math.gcd(frame_on2,frame_on3))
    
    global flipCount
    
    while True:
        # if flipCount == 59:
        #     flipCount = 0
        if(flipCount == 0 or (flipCount%frame_on1 ==0 and flipCount%(frame_on1*2) !=0)):
            shapes[0].setAutoDraw(True)
            shapes[1].setAutoDraw(False)
        if(flipCount%(frame_off1*2) ==0):
            shapes[1].setAutoDraw(True)
            shapes[0].setAutoDraw(False)

        if(flipCount == 0 or(flipCount%frame_on2 ==0 and flipCount%(frame_on2*2) !=0)):
            shapes[2].setAutoDraw(True)
            shapes[3].setAutoDraw(False)
        if(flipCount%(frame_off2*2) ==0):
            shapes[3].setAutoDraw(True)
            shapes[2].setAutoDraw(False)

        if(flipCount == 0 or(flipCount%frame_on3 ==0 and flipCount%(frame_on3*2) !=0)):
            shapes[4].setAutoDraw(True)
            shapes[5].setAutoDraw(False)
        if(flipCount%(frame_off3*2) ==0):
            shapes[5].setAutoDraw(True)
            shapes[4].setAutoDraw(False)

        # time = trialclock.getTime()

        # for frameN in range(looptime):
        mywin.flip()
        flipCount+=1

        result = marker_result()
        if(result):
            print("Marker received: ", result[0][0])
            arrow.setAutoDraw(False)
            arrow = ShapeStim(mywin, vertices=arrowVert, fillColor='darkred', size=.2, lineColor='red', pos=arrowSequence[result[0][0]])
            arrow.setAutoDraw(True)
            mywin.flip()
            core.wait(2.0)

        if 'escape' in event.getKeys():
                core.quit()


        #arrow.setAutoDraw(False)

# %%
#setting params
mywin = visual.Window([1920, 1080], fullscr=False)

test_freq = [6, 10, 16]  #, 15]
freq_len = len(test_freq)

frame_on1, frame_off1 = getFrames(test_freq[0])
frame_on2, frame_off2 = getFrames(test_freq[1])
frame_on3, frame_off3 = getFrames(test_freq[2])

flipCount = 0
trialclock = core.Clock()

frame_on = 0
frame_off = 0

patternup1Pos = [0, 0.65]
patternright1Pos = [0.65, -0.5]
patternleft1Pos =[-0.65, -0.5]

# Arrow position is now y+0.2
arrowUp1Pos = [0, 0.95]
arrowRigh1Pos = [0.65, -0.2]
arrowLeft1Pos=[-0.65, -0.2]

# array to identify the sequenct of the stimuli
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


# %%
#running the actual experiment
message = visual.TextStim(mywin, text='Start recording and press space to continue')
message.draw()
mywin.flip()
while True:
    keys = event.getKeys()
    if 'space' in keys:  # If space has been pushed    
        message.setText = ''
        message.draw()
        mywin.flip() 
        arrow =ShapeStim(mywin, vertices=arrowVert, fillColor='darkred', size=.5, lineColor='red', pos=arrowSequence[0])
        stimuli_blinking_nonstop(frame_on1, frame_off1, frame_on2, frame_off2, frame_on3, frame_off3, shapes,arrow)

# %%
