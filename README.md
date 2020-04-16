# BCI @ AIT

This simple project was created to demonstrate how BCI can be done from Data acquisition, Running the Experiment, and Analyzing the Data perspective, mainly for my students at Asian Institute of Technology, ICT Department.  It is intended only for research and academic purpose. Particularly, this is an ongoing project and I intend to populate this with at least five examples.

Contributors:
Chaklam Silpasuwanchai, Apiporn Simapornchai, Anyone at AIT wanna join?

## About

Hardware:
- OpenBCI + Cyton/Daisy, 250Hz

Software:
- Python-based

All code will be centered around 3 typical steps
1. Data acquisition (Credit: https://github.com/NeuroTechX)
2. Experiment (stimuli) (Online/Offline)
3. Analysis

Analysis will include:
1. Basic feature extraction and classification
2. Common Spatial Pattern and classification
3. Filter-Bank Canonical Correlation Analysis for SSVEP
4. Long Short-Term Memory Recurrent Neural network
5. Temporal Convolutional Network

Things need to take precaution:
- To get a clean signal, it is important to stay at a location free of electric artifacts.  When you look at your brain signals using lsl-viewer, it should be around low frequency, around -10 or less.  If it is more, try make sure your feet/hand is touching the ground and see whether the volts changes.  Also, if your bluetooth receiver is near the power outlet, it can also increase the frequency significantly.  Try move to different locations that are free of power influences.  Last, even your feet/hand is grounded, make sure no electricity is on the ground!, e.g., leaving some plugs on the ground

## Topics

1. **Psychological Experiment**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab; brain microvolts should be around -10; make sure you are properly   grounded (you can close after checking)
   3. Run <code>python offline_experiment.py</code>  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis
2. **SSVEP + control  (online)**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab to check and then close
   3. Run <code>python offline_experiment.py</code>  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis.  Since we are using unsupervised CCA, this file is mainly to  find the optimal parameters when used with online
   7. run <code> python online_classifier.py </code> (make sure to do this before step (viii), or else, no markers stream will be found)
   8. Now, <code> python online_experiment.py </code>
3. **P300 + speller  (real-time)** (TBD)
4. **MI + control   (real-time)** (TBD)
5. **Existing Dataset (probably start with DEAP, and other famous datasets)** (TBD)
