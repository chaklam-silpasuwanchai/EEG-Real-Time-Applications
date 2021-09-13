# Thai P300 BCI Speller

## About

Hardware:
- OpenBCI + Cyton, Sampling rate of 250Hz (8 Channels : Fz, F3, C4, Cz, Pz, P3, O2, O1)

Software:
- Python

### Offline P300 to collectm sample data and train the model
   1. Run <code>python3 1-acquisition/lsl-stream.py</code> on the background
   2. Run <code>python3 1-acquisition/lsl-viewer.py</code> on another tab to check the EEG and then close
   3. Run <code>python3 2-experiment/offline/th_offline_cb.py</code>  (do not click "start" yet)
   4. Run <code>python3 2-experiment/offline/lsl-record-offline.py</code> on another tab; it should detect the marker stream from (iii)
   5. Click "start" on (iii)
   6. Data can be found in th_data. Use <code>3-analysis/offline_analysis.ipynb</code> to analyze the data. You should see a spike for targets, and vice versa for non-targets.
   7. Use <code>3-analysis/combine_par_data.ipynb</code> to combine the participant's data.
   8. Use <code>3-analysis/train_model.ipynb</code> to train and save the classification model.


### Online P300 (with target)
   1. Run <code>python3 2-experiment/online/main_target.py</code>
   2. Click "start" on the pop-up window

   
### Online P300 (without target)
   1. Run <code>python3 2-experiment/online/main_notarget.py</code>
   2. Click "start" on the pop-up window
