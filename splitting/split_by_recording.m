function splits = split_by_recording(meta, test_recording_id)

is_test  = meta.recording_id == test_recording_id;
is_train = ~is_test;

splits.train_idx = find(is_train);
splits.test_idx  = find(is_test);

end
