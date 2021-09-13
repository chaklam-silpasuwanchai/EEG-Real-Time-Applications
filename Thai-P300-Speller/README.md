# BCI @ AIT

This simple project was created to demonstrate how BCI can be done from data acquisition, running the experiment, and analyzing the data perspective, mainly for my students at Asian Institute of Technology, ICT Department.  It is intended to quickly boostrap my lab students for starting their research. This is an ongoing project and I intend to populate this with at least six topics (see below), all of which are ongoing research topics in my lab.

## About

Hardware:
- OpenBCI + Cyton, Sampling rate of 250Hz

Software:
- Python

Some basic terminology defined for new students:
1. **Online vs. Offline analysis**:  Online refers to real-time I/O of EEG signals while offline analysis refers to recording and aftermath analysis.  Both kind of analysese are required - offline analysis allows us to figure optimal parameters and setup, while online analysis is about really putting the system in real usage
2. **P300, SSVEP, Motor Imagery, Power Spectrum**:   These terms describe the paradigm of how brain signals are evoked.  There are different ways in which brain generates some patterns of signals (in volts).  For example, when we encounter some rare but interested stimuli, there will be a spike occuring around 300ms-500ms after the onset of that stimuli, this is called **P300**.  For another example, when we look at two stimuli flickering with different frequency, let's say 10 and 15hz respectively, we can detect which one users are looking at because similar, harmonic frequencies (e.g., 15hz, 30hz, 45hz, etc.) will be shown in users' EEG signals, this is called **SSVEP**.    When users are having high cognitive load, we can infer from power of each frequency band by performing Fourier Transform (or Wavelet Transform).  As a easy way to think about this, when you are stressed, your brain generates very high frequency signal, vice versa, when you are calm, most of your brain signals tend to fall in low frequency range.   This is called **Power spectral Density**.   Not surprisingly, P300, SSVEP, and PSD can all be triggered by audio/visual/haptic stimuli.   As for more user-generated signal rather than relying on external stimuli, brain signals can be evoked by asking the users to perform certain thought (e.g., thinking of moving left), which then developers can map these brain signals to commands.  This is called **Motor Imagery**
3. **Nyquist theorem**: describes a fundamental law in which the highest frequency component should be 1/2 * sampling rate.  For example, if our device is 250hz, then the highest frequency component we can investigate is 125hz.  This is because we need at least two samples per time point to make a judgement of that frequency rate.    In simple words, even our device sampling rate is 250hz, we can only investigate users' frequency band until 125hz.  Fortunately, in BCI, the interested frequency band is typically around 4 - 50 hz.
4. **Lab Streaming Layer**: a network protocol for streaming time-series data (e.g., eeg data, markers).  We are using pylsl python library.
5. **Markers**: markers refer to temporal events, such as image displaying.  We need to map markers to eeg signals, so we can infer what is happening when the stimuli (i.e., markers/events) are shown.  LSL can be used to push markers as well as eeg signals.
6. **Epochs**:  refers to windowed eeg signals that are event-bound.  For example, we may be interested to chop the eeg signal from 0.2 to 0.5s after the onset of the stimuli and check what is happening during that period.  MNE-python are commonly used to get epochs but coding from scratch can also be done (see P300 code).
7. **Signal filters (bandpass/notch filters)**:  refers to process of removing unwanted frequency band from the signals.  For example, in many countries, our electrical devices emit 50hz which can be artifact thus we may want to apply a **notch filter** of 50hz.  For another example, for P300, the frequency band in interest is typically around 1 to 30hz, thus we may want to perform a **bandpass filter** with low cut of 1 and high cut of 30hz.   Signal filtering can be done using the scipy.signal python library.
8. **Artifacts**: refers to noise of EEG signals.  Sources include eye blinks, muscle/head movements, random thoughts.  Can be partially fix by ICA (Independent Component Analysis) or simply restrict your participants (tell him/her do not move!) or use baseline corrections.  For ICA, using MNE-python can be easily be achieved.

All code will be centered around 3 typical steps
1. Data acquisition (Credit: https://github.com/NeuroTechX)
2. Experiment (stimuli) (Online/Offline experiment)
3. Analysis (Offline analysis)

Topics include:
1. **Psychological Experiment** - follows typical paradigm of showing stimuli and inferring user's state based on EEG signal, typically power spectrum (e.g., alpha).  Offline analysis based.  Useful for students to understand the basic setup of the EEG system.
2. **SSVEP Control** - aims to create an unsupervised online classification of three targets using Filter-bank Canonical Correlation Analysis.  Since this is unsupervised, offline analysis is done mainly to identify optimal parameters (e.g., epoch width).  Useful for students to understand the SSVEP paradigms.
3. **P300 Speller** - aims to create a 6 x 6 matrix speller using different stimuli variations; unsupervised method (e.g., discriminant analysis) will be demonstrated for online classification.  Useful for students to understand how ERP works.
4. **Motor Imagery Control** - aims to allow users to train left/right movement thoughts to control left/right movements
5. **Real-time Emotion Recognition** - aims to create a real-time (online) emotion classification system based on power spectrum.
6. **Thoughts reconstruction** - aims to perform thoughts reconstruction by translating EEG signals to audio/visual forms using GANs.

Things need to take precaution:
- To get a clean signal, it is important to **stay at a location free of electric artifacts**.  When you look at your brain signals using lsl-viewer, it should be around low frequency, around -10 or less.  If it is more, try make sure your feet/hand is touching the ground and see whether the volts changes.  Also, if your bluetooth receiver is near the power outlet, it can also increase the frequency significantly.  Try move to different locations that are free of power influences.  Last, even your feet/hand is grounded, make sure no electricity is on the ground!, e.g., leaving some plugs on the ground

## How to run

**P300**
   1. Run <code>python lsl-stream</code> on the background
   2. Run <code>python lsl-viewer</code> on another tab to check and then close
   3. Run <code>python offline_experiment.py</code>  (do not click space yet)
   4. Run <code>python lsl-record</code> on another tab; it should detect the marker stream from (iii)
   5. Press Start from (iii)
   6. Data can be found in 3-analysis/data; Open Offline.ipynb for offline analysis.  You should see a spike for targets, and vice versa for non-targers
   7. run <code> python online_experiment.py </code> (make sure to do this before step (viii), but do not press Start yet)
   8. Now, <code> python unsupervised_online_classifier.py </code>
   9. Now, press Start from (vii).  The code from (viii) will send the classification result to (vii)
