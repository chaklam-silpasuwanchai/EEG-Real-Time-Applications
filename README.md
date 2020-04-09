# BCI

This simple project was created to demonstrate how BCI can be done from Data acquisition, Running the Experiment, and Analyzing the Data perspective, mainly for my students at Asian Institute of Technology, ICT Department.  Particularly, this is an ongoing project and I intend to populate this with at least five examples.

Hardware:
- OpenBCI + Cyton/Daisy, 250Hz

Software:
- Python-based

All code will be centered around 3 typical steps
1. Data acquisition
2. Experiment (stimuli) (incl. Online Classification or Training)
3. Analysis

Analysis will include:
1. Basic feature extraction and classification
2. Common Spatial Pattern and classification
3. Long Short-Term Memory Recurrent Neural network
4. Temporal Convolutional Network

Contributors:
Chaklam Silpasuwanchai, Apiporn Simapornchai, Anyone at AIT wanna join?

1. **Psychological Experiment**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab; brain microvolts should be around -10; make sure you are properly   grounded (you can close after checking)
   3. Run 2-experiment/Presents-stimuli.ipynb  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Analyze-EEG-signal.ipynb for offline analysis
2. **SSVEP + control  (online)**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab to check and then close
   3. Run 2-experiment/Training.ipynb  (do not click space yet)
   4. Run <code>python lsl-record</code>; it should detect the marker stream from (iii)
   5. Press space bar from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis.  Notice it will save the trained model 
   7. Now, open the 2-experiment/Online.ipynb (do not click space yet)
   8. run <code> python online-classification.py </code> (make sure you have input the trained model, set hyperparameters)
   9. Spacebar (vii)....it should get the predicted class if the threshold exceeds
3. **P300 + speller  (real-time)** (TBD)
4. **MI + control   (real-time)** (TBD)
5. **Existing Dataset (probably start with DEAP, and other famous datasets)** (TBD)
