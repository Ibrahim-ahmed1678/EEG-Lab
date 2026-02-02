function export_tensor(X, y, tts, outfile)

assert(ndims(X) == 4, 'X must be [N x C x F x T]');

if exist(outfile,'file'), delete(outfile); end

h5create(outfile, '/X', size(X), 'Datatype','single');
h5write(outfile, '/X', single(X));

h5create(outfile, '/y', size(y), 'Datatype','uint8');
h5write(outfile, '/y', uint8(y));

h5create(outfile, '/tts', size(tts), 'Datatype','single');
h5write(outfile, '/tts', single(tts));

fprintf('Saved %d samples → %s\n', size(X,1), outfile);
end
