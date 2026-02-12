function [Xnorm, norm_params] = normalize_spectrograms(X, norm_params)
%NORMALIZE_SPECTROGRAMS  Leakage-safe normalization for log-power spectrogram CNN inputs
%
% X shape: [N x C x F x T]  (windows, channels, freq bins, time bins)
% X should already be log-power (e.g., log10(power + eps)).
%
% If norm_params is empty/missing:
%   -> FIT on X (use TRAIN ONLY), returning norm_params
% If norm_params is provided:
%   -> APPLY to X (VAL/TEST) using stored params
%
% Implements:
%   1) Per-channel mean removal (computed from training data)
%   2) Per-channel, per-frequency-bin z-score (band-wise normalization)
%      i.e., mu/sigma are [C x F], computed over windows+time.

    if nargin < 2
        norm_params = [];
    end

    % Basic checks
    assert(ndims(X) == 4, 'X must be [N x C x F x T]');
    X = single(X);

    % Dimensions
    % N = size(X,1);
    C = size(X,2);
    F = size(X,3);
    % T = size(X,4);

    if isempty(norm_params)
        % =========================
        % FIT (TRAIN ONLY)
        % =========================

        % (A) Per-channel mean over all windows, freqs, time: mu_chan [C x 1]
        % mean over N,F,T -> dims [1 3 4]
        mu_chan = squeeze(mean(X, [1 3 4]));  % [C x 1]

        % Center by channel mean before band-wise stats
        % reshape mu_chan to [1 x C x 1 x 1] for broadcasting
        mu_chan_reshape = reshape(mu_chan, [1 C 1 1]);
        Xc = X - mu_chan_reshape;

        % (B) Band-wise (per-channel, per-frequency) mean/std over N and T
        % mean over N,T -> dims [1 4] leaving [C x F]
        mu_cf = squeeze(mean(Xc, [1 4]));          % [C x F]
        sigma_cf = squeeze(std(Xc, 0, [1 4]));     % [C x F]

        % Guard against zeros
        sigma_cf(sigma_cf < 1e-6) = 1e-6;

        norm_params.mu_chan = single(mu_chan);     % [C x 1]
        norm_params.mu_cf   = single(mu_cf);       % [C x F]
        norm_params.sigma_cf= single(sigma_cf);    % [C x F]

    else
        % =========================
        % APPLY (VAL/TEST)
        % =========================
        mu_chan  = norm_params.mu_chan;    % [C x 1]
        mu_cf    = norm_params.mu_cf;      % [C x F]
        sigma_cf = norm_params.sigma_cf;   % [C x F]

        % Sanity checks
        assert(numel(mu_chan) == C, 'norm_params.mu_chan channel count mismatch');
        assert(all(size(mu_cf) == [C F]), 'norm_params.mu_cf must be [C x F]');
        assert(all(size(sigma_cf) == [C F]), 'norm_params.sigma_cf must be [C x F]');
    end

    % =========================
    % APPLY NORMALIZATION
    % =========================

    % Center by per-channel mean
    mu_chan_reshape = reshape(mu_chan, [1 C 1 1]);
    Xc = X - mu_chan_reshape;

    % Band-wise z-score: subtract mu_cf, divide by sigma_cf (both [C x F])
    mu_cf_reshape    = reshape(mu_cf,    [1 C F 1]);
    sigma_cf_reshape = reshape(sigma_cf, [1 C F 1]);

    % Use implicit expansion (R2016b+) or bsxfun fallback
    try
        Xnorm = (Xc - mu_cf_reshape) ./ sigma_cf_reshape;
    catch
        Xnorm = bsxfun(@rdivide, bsxfun(@minus, Xc, mu_cf_reshape), sigma_cf_reshape);
    end

    Xnorm = single(Xnorm);
end
