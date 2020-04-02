###############################################
####Spectral Features -> Classification########
###############################################

#This is rather a common pipeline for understanding anything related to spectrum

#1. **Feature Extraction**: Power spectral features will be used as main features
#2. **Classification**

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mne.time_frequency import psd_welch
from sklearn.metrics import accuracy_score
from mne.viz import tight_layout
import numpy as np
import helper as helper

"""
This function takes an ``mne.Epochs`` object and creates EEG features based
on relative power in specific frequency bands that are compatible with
scikit-learn.
"""
def eeg_power_band(epochs):
    
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 2],
                  "theta": [4, 8],
                  "alpha": [8, 12],
                  "beta": [12, 20],
                  "lowgamma": [20, 30],
                  "midgamma": [30, 50]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=50)
    
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


"""
Scikit-learn pipeline composes an estimator as a sequence of transforms and a final estimator, while the FunctionTransformer converts a python function in an estimator compatible object. In this manner we can create scikit-learn estimator that takes mne.Epochs thanks to eeg_power_band function we just created.
"""
def decode(raw, event_id, tmin, tmax):
    epochs = helper.getEpochs(raw, event_id, tmin, tmax)

    y = epochs.events[:, -1]

    # define cross validation 
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25, 
                            random_state=42)

    pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False), RandomForestClassifier(n_estimators=100, random_state=42))

    # do cross-validation
    preds = np.empty(len(y))
    for train, test in cv.split(epochs, y):
        pipe.fit(epochs[train], y[train])
        preds[test] = pipe.predict(epochs[test])
        acc = accuracy_score(y[test], preds[test])
        print("Accuracy score: {}".format(acc))

    # classification report
    print(classification_report(y[test], preds[test], target_names=event_id.keys()))

    # confusion matrix
    print(confusion_matrix(y[test], preds[test]))