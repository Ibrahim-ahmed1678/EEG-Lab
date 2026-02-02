function [windows, labels, tts, meta] = extract_windows_labels(EEG, seizures, params)
%EXTRACT_WINDOWS_LABELS
% Extract preictal and interictal windows from continuous EEG
%
% INPUTS
%   EEG      : EEGLAB struct (continuous, preprocessed)
%   seizures : [N x 2] array of [onset, end] in seconds
%              (can be empty if no seizures in recording)
%   params   : struct with fields:
%       win_sec
%       step_sec
%       preictal_horizon
%       buffer
%       postictal_gap
%       interictal_gap
%       recording_id
%
% OUTPUTS
%   windows : cell array, each [C x samples]
%   labels  : uint8 vector (0 = interictal, 1 = preictal)
%   tts     : single vector (seconds to next seizure, NaN for interictal)
%   meta    : struct with metadata fields
%       .window_end_time
%       .seizure_index
%       .recording_id

% -------------------------
% Setup
% -------------------------
fs = EEG.srate;

win_len  = round(params.win_sec  * fs);
step_len = round(params.step_sec * fs);

data = EEG.data;             % [C x T]
T = size(data,2);

starts = 1:step_len:(T - win_len + 1);
maxN   = numel(starts);

% -------------------------
% Preallocation
% -------------------------
windows = cell(maxN,1);
labels  = zeros(maxN,1,'uint8');
tts     = nan(maxN,1,'single');

meta.window_end_time = nan(maxN,1);
meta.seizure_index  = nan(maxN,1);
meta.recording_id   = nan(maxN,1);

cnt = 0;

% -------------------------
% Main loop
% -------------------------
for i = 1:maxN
    s0 = starts(i);
    s1 = s0 + win_len - 1;

    t_end = (s1 - 1) / fs;   % window end time (sec)
    w = data(:, s0:s1);

    label = -1;              % -1 = discard
    time_to_seizure = NaN;
    sz_idx = NaN;

    % -------------------------
    % Seizure-aware logic
    % -------------------------
    for k = 1:size(seizures,1)
        onset = seizures(k,1);
        endt  = seizures(k,2);

        % Exclude ictal + postictal
        if t_end >= onset && t_end <= endt + params.postictal_gap
            label = -1;
            break
        end

        % Preictal
        if t_end >= onset - params.preictal_horizon && ...
           t_end <  onset - params.buffer
            label = 1;
            time_to_seizure = onset - t_end;
            sz_idx = k;
            break
        end
    end

    % -------------------------
    % Interictal check
    % -------------------------
    if label == -1
        if isempty(seizures)
            % Recording has no seizures → all valid windows are interictal
            label = 0;
        else
            dist_on = abs(t_end - seizures(:,1));
            dist_en = abs(t_end - seizures(:,2));

            if all(dist_on >= params.interictal_gap) && ...
               all(dist_en >= params.interictal_gap)
                label = 0;
            else
                continue
            end
        end
    end

    % -------------------------
    % Store window
    % -------------------------
    cnt = cnt + 1;

    windows{cnt} = w;
    labels(cnt)  = label;
    tts(cnt)     = time_to_seizure;

    meta.window_end_time(cnt) = t_end;
    meta.seizure_index(cnt)  = sz_idx;
    meta.recording_id(cnt)   = params.recording_id;
end

% -------------------------
% Trim unused entries
% -------------------------
windows = windows(1:cnt);
labels  = labels(1:cnt);
tts     = tts(1:cnt);

meta.window_end_time = meta.window_end_time(1:cnt);
meta.seizure_index  = meta.seizure_index(1:cnt);
meta.recording_id   = meta.recording_id(1:cnt);

end
