% standard preprocessing for siezure prediction
% input: EEG (continuous, EEG struct)
% output: EEG (continuous, cleaned)
function EEG = preprocess_eeg(EEG)


% high pass at 1Hz for cleanline stability
EEG = pop_eegfiltnew(EEG, 'locutoff', 1);


% adaptive line noise removal around 60hz. same regressive technique used
% in the mdpi paper jackson shared
EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',1:EEG.nbchan ,'computepower',1,'linefreqs',60:60:180 , ...
    'newversion',0,'normSpectrum',0,'p',0.03,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype','Channels',...
    'taperbandwidth',2,'tau',100,'verb',1,'winsize',2,'winstep',0.01);


% bandpass filter from 0.5 to 40Hz
EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40);


% removes flat, noisy or disconnected channels. no ica so no risk of
% removing siezures
EEG = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', 5, ...
    'ChannelCriterion', 0.8, ...
    'LineNoiseCriterion', 4, ...
    'Highpass', 'off', ...
    'BurstCriterion', 'off', ...
    'WindowCriterion', 'off', ...
    'WindowCriterionTolerances', 'off');

% standard average rereference
EEG = pop_reref( EEG, []);

% just a consistency check on the datastructure
EEG = eeg_checkset(EEG);
end