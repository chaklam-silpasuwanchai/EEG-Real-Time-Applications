# BCI @ AIT

This simple project was created to demonstrate how BCI can be done from Data acquisition, Running the Experiment, and Analyzing the Data perspective, mainly for my students at Asian Institute of Technology, ICT Department.  It is intended only for research and academic purpose. This is an ongoing project and I intend to populate this with at least six topics (see below).

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

Topics include:
1. **Psychological Experiment** - follows typical paradigm of showing stimuli and inferring user's state based on EEG signal, typically power spectrum (e.g., alpha).  Offline analysis based.  Useful for students to understand the basic setup of the EEG system.
2. **SSVEP** - aims to create an unsupervised online classification of three targets using Filter-bank Canonical Correlation Analysis.  Since this is unsupervised, offline analysis is done mainly to identify optimal parameters (e.g., epoch width).  Useful for students to understand the SSVEP paradigms.
3. **P300** - aims to create a 6 x 6 matrix speller using different stimuli variations; both supervised (training first) and unsupervised method (e.g., discriminant analysis) will be demonstrated for online classification.  Useful for students to understand how ERP works.
4. **MI** - aims to create a simple left and right movement control.  Similar to P300, this is a supervised method.  Useful for students to understand Mu rhythms
5. **Real-time Emotion Recognition** - aims to create a real-time (online) emotion classification system.  Not sure how to approach this but I believe it's not too difficult
6. **Existing Dataset** - aims to perform offline analysis on typical datasets for BCI research using contemporary ML and signal processing techqniues.   Useful for students to understand how to apply contemporary ML and signal processing techniques.

Things need to take precaution:
- To get a clean signal, it is important to **stay at a location free of electric artifacts**.  When you look at your brain signals using lsl-viewer, it should be around low frequency, around -10 or less.  If it is more, try make sure your feet/hand is touching the ground and see whether the volts changes.  Also, if your bluetooth receiver is near the power outlet, it can also increase the frequency significantly.  Try move to different locations that are free of power influences.  Last, even your feet/hand is grounded, make sure no electricity is on the ground!, e.g., leaving some plugs on the ground

## How to run

1. **Psychological Experiment**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab; brain microvolts should be around -10; make sure you are properly   grounded (you can close after checking)
   3. Run <code>python offline_experiment.py</code>  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis
2. **SSVEP**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab to check and then close
   3. Run <code>python offline_experiment.py</code>  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis.  Since we are using unsupervised CCA, this file is mainly to  find the optimal parameters when used with online
   7. run <code> python online_classifier.py </code> (make sure to do this before step (viii), or else, no markers stream will be found)
   8. Now, <code> python online_experiment.py </code>
3. **P300** (TBD)
4. **MI** (TBD)
5. **Real-time Emotion Recognition** (TBD)
6. **Existing Dataset** (TBD)
