%% run_pipeline.m
% Driver SCRIPT for EEG seizure CNN dataset generation
%
% This script:
%   1) loads EEG recordings
%   2) preprocesses EEG using EEGLAB-based steps
%   3) extracts labeled windows (preictal / interictal + time-to-seizure)
%   4) converts windows to time–frequency spectrograms
%   5) splits data by recording (train vs test)
%   6) normalizes spectrograms using TRAIN-ONLY statistics
%   7) exports tensors for CNN training/testing
%
% IMPORTANT:
% - This is a SCRIPT, not a function
% - You only edit the "USER INPUTS" section
% - All heavy logic lives in modular functions
%
% Pipeline structure mirrors common seizure CNN papers
% and is safe against train/test data leakage.

clear; clc;

%% =========================
% ADD PIPELINE TO MATLAB PATH
% =========================
% Assumes you are running this from the repo root
addpath(genpath(pwd));

%% =========================
% USER INPUTS (EDIT HERE)
% =========================

% ---- EEG recordings (same patient) ----
data_dir = fullfile(pwd, 'data', 'chb01');

recordings = {
    fullfile(data_dir, 'chb01_01.edf')
    fullfile(data_dir, 'chb01_03.edf')
    fullfile(data_dir, 'chb01_04.edf')
};

% ---- Seizure times per recording ----
% Each cell entry: [N x 2] matrix of [onset, end] in SECONDS
seizure_times = {
    [];              % chb01_01 (no seizures)
    [2996 3036];     % chb01_03 (reserved for testing)
    [1467 1494];     % chb01_04
};

% ---- Which recording to HOLD OUT for testing ----
% (Leave-one-recording-out evaluation)
test_recording_id = 2;

% ---- Output filenames ----
train_outfile = 'train_cnn.h5';
test_outfile  = 'test_cnn.h5';

%% =========================
% WINDOWING / LABELING PARAMETERS
% =========================
params.win_sec  = 5;        % window length (seconds)
params.step_sec = 2.5;      % step size (seconds)

params.preictal_horizon = 30 * 60;        % 30 minutes
params.buffer           = 5  * 60;        % 5 minutes (exclusion buffer)
params.postictal_gap    = 30 * 60;        % 30 minutes
params.interictal_gap   = 2  * 60 * 60;   % 2 hours

%% =========================
% SPECTROGRAM PARAMETERS
% =========================
spec_params.stft_win_sec  = 1.0;   % STFT window length (sec)
spec_params.stft_overlap = 0.5;   % fraction overlap
spec_params.stft_nfft    = 256;   % FFT size

%% =========================
% PREALLOCATE PER-RECORDING STORAGE (OPTIMIZED)
% =========================
% We store outputs per recording, then concatenate once.
% This avoids MATLAB dynamic array growth warnings
% and is much faster for large datasets.

num_rec = numel(recordings);

all_windows_cell = cell(num_rec,1);
all_labels_cell  = cell(num_rec,1);
all_tts_cell     = cell(num_rec,1);
meta_cell        = cell(num_rec,1);

fprintf('Processing %d recordings...\n', num_rec);

%% =========================
% PROCESS EACH RECORDING
% =========================
for r = 1:num_rec

    fprintf('--- Recording %d / %d: %s ---\n', ...
        r, num_rec, recordings{r});

    % ---- Load raw EEG ----
    EEG = pop_biosig(recordings{r});

    % ---- Preprocess EEG (filtering, referencing, etc.) ----
    EEG = preprocess_eeg(EEG);

    % ---- Attach recording ID for downstream splitting ----
    params.recording_id = r;

    % ---- Windowing + labeling ----
    % Outputs:
    %   w : cell array of windows [C x T]
    %   l : labels (preictal/interictal)
    %   t : time-to-seizure
    %   m : metadata (recording_id per window)
    [w, l, t, m] = extract_windows_labels( ...
        EEG, seizure_times{r}, params);

    % ---- Store per-recording outputs ----
    all_windows_cell{r} = w;
    all_labels_cell{r}  = l(:);
    all_tts_cell{r}     = t(:);
    meta_cell{r}        = m.recording_id(:);
end

%% =========================
% CONCATENATE ALL RECORDINGS
% =========================
% Single concatenation step (fast, memory-safe)

all_windows = vertcat(all_windows_cell{:});
all_labels  = vertcat(all_labels_cell{:});
all_tts     = vertcat(all_tts_cell{:});
meta.recording_id = vertcat(meta_cell{:});

fprintf('Total windows extracted: %d\n', numel(all_windows));

%% =========================
% CONVERT WINDOWS → NUMERIC TENSOR
% =========================
% From cell array {N} of [C x T] → numeric [N x C x T]

fprintf('Converting windows to numeric tensor...\n');

N = numel(all_windows);
[C, T] = size(all_windows{1});

Xwin = zeros(N, C, T, 'single');
for i = 1:N
    Xwin(i,:,:) = all_windows{i};
end

clear all_windows all_windows_cell;

%% =========================
% COMPUTE SPECTROGRAMS
% =========================
fprintf('Computing spectrograms...\n');

Xspec = compute_spectrograms(Xwin, EEG.srate, spec_params);
clear Xwin;

%% =========================
% SPLIT DATA BY RECORDING
% =========================
fprintf('Splitting data (test recording = %d)...\n', test_recording_id);

splits = split_by_recording(meta, test_recording_id);

Xtrain = Xspec(splits.train_idx,:,:,:);
Xtest  = Xspec(splits.test_idx,:,:,:);

ytrain = all_labels(splits.train_idx);
ytest  = all_labels(splits.test_idx);

ttstrain = all_tts(splits.train_idx);
ttstest  = all_tts(splits.test_idx);

clear Xspec all_labels all_tts;

%% =========================
% NORMALIZATION (TRAIN-ONLY)
% =========================
fprintf('Normalizing spectrograms (train-only statistics)...\n');

[Xtrain_norm, norm_params] = normalize_spectrograms(Xtrain, []);
Xtest_norm = normalize_spectrograms(Xtest, norm_params);

clear Xtrain Xtest;

%% =========================
% EXPORT TENSORS
% =========================
fprintf('Exporting tensors...\n');

export_tensor(Xtrain_norm, ytrain, ttstrain, train_outfile);
export_tensor(Xtest_norm,  ytest,  ttstest,  test_outfile);

fprintf('Pipeline complete.\n');
