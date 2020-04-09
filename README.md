# BCI

This simple project was created to demonstrate how BCI can be done from Data acquisition, Running the Experiment, and Analyzing the Data perspective.  Particularly, this is an ongoing project and I intend to populate this with at least five examples:

1. Psychological Experiment
   a. Run <code>python lsl-stream</code> on the background
   b. Run <code>python lsl-viewer</code> on another tab; brain microvolts should be around -10; make sure you are properly   grounded (you can close after checking)
   c. Run 2-experiment/Presents-stimuli.ipynb  (do not click space yet)
   d. Run <code>python lsl-record</code>; it should detect the marker stream from (c)
   e. Press space bar from (c)
   f. Data can be found in 3-analysis/data; Open Analyze-EEG-signal.ipynb for offline analysis
2. SSVEP + control  (real-time)
   a. Run <code>python lsl-stream</code> on the background
   b. Run <code>python lsl-viewer</code> on another tab to check and then close
   c. Run 2-experiment/Training.ipynb  (do not click space yet)
   d. Run <code>python lsl-record</code>; it should detect the marker stream from (c)
   e. Press space bar from (c)
   f. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis.  Notice it will save the trained model 
   g. Now, open the 2-experiment/Online.ipynb (do not click space yet)
   h. run <code> python online-classification.py </code> (make sure you have input the trained model, set hyperparameters)
   i. Spacebar (g)....it should get the predicted class if the threshold exceeds
3. P300 + speller  (real-time) (TBD)
4. MI + control   (real-time) (TBD)
5. Existing Dataset (probably start with DEAP, and other famous datasets) (TBD)

All will be centered around the use of:

Hardware:
- OpenBCI + Cyton/Daisy, 250Hz

Software:
- Python-based

All code will be centered around 3 typical steps
1. Data acquisition
2. Experiment (stimuli)
3. Analysis

Analysis will include:
1. Basic feature extraction and classification
2. Common Spatial Pattern and classification
3. Long Short-Term Memory Recurrent Neural network
4. Temporal Convolutional Network
