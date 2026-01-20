import argparse
from process_electrodes import epoching, save_prepr,mvnn

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_ses', default=4, type=int)
parser.add_argument('--sfreq', default=100, type=int) # change resampling freq. through command line
parser.add_argument('--mvnn_dim', default='time', type=str)
parser.add_argument('--project_dir', default='../project_directory', type=str)

args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# Epoch and sort the data
# =============================================================================
epoched_test, _, ch_names, times = epoching(args, 'test', seed)
epoched_train, img_conditions_train, _, _ = epoching(args, 'training', seed)
#print(epoched_train)
print(print(f"Dimensions: {len(epoched_train)} x {len(epoched_train[0])}"))


#%%
whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
del epoched_test, epoched_train
#%%
# =============================================================================
# Merge and save the preprocessed data
## =============================================================================

print('>>> EEG Preprocessing Completed <<<')

save_prepr(args, whitened_test, whitened_train, img_conditions_train, ch_names,
	times, seed)
