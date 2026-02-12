function Xspec = compute_spectrograms(Xwin, fs, params)
% COMPUTE_SPECTROGRAMS
% Convert windowed EEG into CNN-ready spectrogram tensors
%
% INPUTS:
%   Xwin   : [Nwin x Nchan x Nsamples] time-domain windows
%   fs     : sampling rate (Hz)
%   params : struct with fields
%       .stft_win_sec     (e.g. 1.0)
%       .stft_overlap    (0–0.9, e.g. 0.5)
%       .stft_nfft       (e.g. 256)
%
% OUTPUT:
%   Xspec  : [Nwin x Nchan x Nfreq x Ntime] log-power spectrograms

% -----------------------------
% Parameters
% -----------------------------
stft_win     = round(params.stft_win_sec * fs);
stft_overlap = round(stft_win * params.stft_overlap);
nfft         = params.stft_nfft;

% -----------------------------
% Dimensions
% -----------------------------
[Nwin, Nchan, ~] = size(Xwin);

% Precompute spectrogram size
[S0, ~, ~] = spectrogram(squeeze(Xwin(1,1,:)), ...
                          stft_win, stft_overlap, nfft, fs);

[Nfreq, Ntime] = size(S0);

% -----------------------------
% Preallocate output tensor
% -----------------------------
Xspec = zeros(Nwin, Nchan, Nfreq, Ntime, 'single');

% -----------------------------
% Compute spectrograms
% -----------------------------
for w = 1:Nwin
    for ch = 1:Nchan

        signal = squeeze(Xwin(w, ch, :));

        [S, ~, ~] = spectrogram( ...
            signal, ...
            stft_win, ...
            stft_overlap, ...
            nfft, ...
            fs ...
        );

        % Log-power spectrogram (CNN-friendly)
        Xspec(w, ch, :, :) = log10(abs(S).^2 + eps);

    end
end

end
