# EEG-Lab

# EEG Seizure Prediction Preprocessing Pipeline

This repository contains a **reproducible EEG preprocessing and dataset-generation pipeline**
designed for **seizure prediction models**, with a focus on **CNN / transfer-learning approaches**
using **time–frequency (spectrogram) representations**.

---

##  Requirements

- MATLAB (R2019b or newer recommended)
- EEGLAB (tested with v2025+)
- EEGLAB plugins:
  - **BIOSIG** (EDF import)
  - **CleanLine** (line-noise regression)

See `requirements.md` for installation notes.

---

##  Pipeline Overview

The pipeline performs the following steps:

1. **Load raw EEG (.edf) recordings**
2. **Preprocess EEG**
   - high-pass filter (1 Hz)
   - line-noise removal (CleanLine)
   - band-pass filter (0.5–40 Hz)
   - average referencing
3. **Window continuous EEG**
   - sliding windows (e.g., 5 s, 50% overlap)
4. **Label windows**
   - preictal / interictal
   - time-to-seizure (regression target)
   - supports multiple seizures per recording
5. **Convert windows to spectrograms**
   - log-power STFT
6. **Split by recording**
   - train on most recordings, save 1 for testing
7. **Normalize spectrograms**
   - TRAIN-only statistics
   - per-channel + per-frequency-band z-score
8. **Export tensors**
   - ready for CNN training

---

*quick clarification* 
- training data is used to learn normalization parameters
- testing data is normalized using those same parameters
- if test data were to influence normalization statistics, the performance of the model would be artificially improved which is not what we want


##  How to Run

### Edit experiment parameters
Open `run_pipeline.m` and modify only the **USER INPUTS** section:
```matlab
recordings = {...};
seizure_times = {...};
test_recording_id = ...;

don't edit anything else.

in the matlab command window, enter run_pipeline and you will recieve two 4d tensor outputs: train_cnn.h5 and test_cnn.h5
these .h5 files can be fed to the learning models


