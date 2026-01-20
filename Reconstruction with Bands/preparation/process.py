import os
from typing import Any
import pickle
os.environ['NPY_PICKLE_PROTOCOL'] = '5' # this did not work, solved
import mne
import numpy as np
from numpy import ndarray, dtype
from sklearn.utils import shuffle
from tqdm import tqdm



def epoching(args, data_part, seed):
    epoched_data = []
    img_conditions = []
    selected_ch_names = None

    for s in range(args.n_ses):
        eeg_dir = os.path.join('eeg_dataset', 'raw_data', f'sub-{args.sub:02}', f'ses-{s + 1:02}',
                               f'raw_eeg_{data_part}.npy')
        eeg_data = np.load(os.path.join(args.project_dir, eeg_dir), allow_pickle=True).item()

        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info)
        del eeg_data

        # Filter the raw data in bands sub 7 alpha GED
        #filtered_raw = raw.copy().filter(l_freq=9.567, h_freq=10.329, fir_design='firwin', verbose=True)
        #filtered_raw = raw.copy().filter(l_freq=11.440780, h_freq=13.00000, fir_design='firwin', verbose=True)
        #high gamma
        #filtered_raw = raw.copy().filter(l_freq= 55, h_freq= 95, fir_design='firwin', verbose=True)

        # Get events
        #events = mne.find_events(filtered_raw, stim_channel='stim')
        events = mne.find_events(raw, stim_channel='stim')

        # More specific regex pattern for O and P channels
        # Select only channels that start with O, PO, or P followed by a number or z
        #chan_idx = np.asarray(mne.pick_channels_regexp(filtered_raw.info['ch_names'],
        #                                               '^O[0-9z]|^PO[0-9z]|^P[0-9z]'))
        #17 electrodes
        #chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],'^O[0-9z]|^PO[0-9z]|^P[0-9z]'))
        # 8 channels own selection parieto-occipital regions
        #chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
        #                                               '^PO[0-9z]|^O[0-9z]'))
        #  me 8 electrodes
        #specific_channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4','PO8']
        # central electrodes Fpz, Fz, FCz, Cz, CPz, Pz, POz, Oz, O2, O1
        #specific_channels = ['Fp1', 'Fp2','AFz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'O2', 'O1']
        # 5 electrodes
        #specific_channels = ['PO7', 'PO3', 'POz', 'PO4', 'PO8']
        #m 3 electrodes
        #specific_channels = ['PO3', 'POz', 'PO4']
        #specific_channels = ['Fp1','AFz','Fp2']
        specific_channels = ['O1', 'Oz', 'O2']
        #specific_channels = ['P3', 'Pz', 'P4', 'POz']
        #3 eletrodes left leaving midline
        #specific_channels = ['O2', 'PO8', 'PO4']
        # 5 left electrodes ['P3', 'P1', 'O1', 'PO3' , 'PO7' ]
        #specific_channels = ['P3', 'P1', 'O1', 'PO3' , 'PO7' ]
        #5 right electrodes
        #specific_channels = ['P4', 'P2', 'O2', 'PO4' , 'PO8' ]
        #8 elctrodes front
        #specific_channels = ['Fp1','AFz','Fp2','AF7','AF3','FCz','AF4', 'AF8']

        #10 left
        #specific_channels = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'P1', 'P5', 'PO7', 'PO3', 'POz']
        #specific_channels = ['Pz', 'P4', 'P8', 'O2', 'Oz', 'P2', 'P6', 'PO8', 'PO4', 'POz']
        # 8 electrodes
        #specific_channels = ['Pz', 'P3', 'P4', 'P1', 'PO3', 'POz', 'PO4', 'P2']
        chan_idx = np.asarray(mne.pick_channels(raw.info['ch_names'], specific_channels))

        #chan_idx= np.asarray(mne.pick_channels_regexp(raw.info['ch_names']))


        #chan_idx = np.arange(len(filtered_raw.info['ch_names']))

        #new_chans = [filtered_raw.info['ch_names'][c] for c in chan_idx]
        new_chans = [raw.info['ch_names'][c] for c in chan_idx]
        print(f"Selected channels ({len(new_chans)}): {new_chans}")

        if selected_ch_names is None:
            selected_ch_names = new_chans
        elif set(selected_ch_names) != set(new_chans):
            raise ValueError(f"Channel mismatch between sessions. Session {s + 1} has different channels.")

        # Remove stim channel if present
        if 'stim' in new_chans:
            new_chans.remove('stim')

        #filtered_raw.pick_channels(new_chans)
        raw.pick_channels(new_chans)
        # Verify channel count after selection
        #print(f"Number of channels after selection: {len(filtered_raw.info['ch_names'])}")
        print(f"Number of channels after selection: {len(raw.info['ch_names'])}")

        # Remove target trials (event 99999)
        idx_target = np.where(events[:, 2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        # Epoch the filtered data
        #epochs = mne.Epochs(filtered_raw, events, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
        #del filtered_raw

        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
        del raw


        if args.sfreq < 1000:
            epochs.resample(args.sfreq)

        ch_names = epochs.info['ch_names']
        times = epochs.times

        data = epochs.get_data()
        events = epochs.events[:, 2]
        img_cond = np.unique(events)
        del epochs

        max_rep = 20 if data_part == 'test' else 2
        sorted_data = np.zeros((len(img_cond), max_rep, len(ch_names), data.shape[2]))

        for i in range(len(img_cond)):
            idx = np.where(events == img_cond[i])[0]
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)
        print(f" epoched_data : {epoched_data}")
        del sorted_data


    return epoched_data, img_conditions, ch_names, times
#%%
def mvnn(args, epoched_test, epoched_train):


    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(args.n_ses):
        session_data = [epoched_test[s], epoched_train[s]]

        ### Compute the covariance matrices ###
        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty((len(session_data),session_data[0].shape[2],
            session_data[0].shape[2]))
        for p in range(sigma_part.shape[0]):
            # Image conditions covariance matrix of shape:
            # Image conditions × EEG channels × EEG channels
            sigma_cond = np.empty((session_data[p].shape[0],
                session_data[0].shape[2],session_data[0].shape[2]))
            for i in tqdm(range(session_data[p].shape[0])):
                cond_data = session_data[p][i]
                # Compute covariace matrices at each time point, and then
                # average across time points
                if args.mvnn_dim == "time":
                    sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                        shrinkage='auto') for t in range(cond_data.shape[2])],
                        axis=0)
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif args.mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                        shrinkage='auto') for e in range(cond_data.shape[0])],
                        axis=0)
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # Average the covariance matrices across image partitions
        sigma_tot = sigma_part.mean(axis=0)
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

        ### Whiten the data ###
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
            session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
            session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
                @ sigma_inv).swapaxes(1, 2), session_data[1].shape))

    ### Output ###
    return whitened_test, whitened_train
#%%
# def save_prepr(args, whitened_test, whitened_train, img_conditions_train,
#     ch_names, times, seed):
#
#
#     import numpy as np
#     from sklearn.utils import shuffle
#     import os
#
#     ### Merge and save the test data ###
#     for s in range(args.n_ses):
#         if s == 0:
#             merged_test = whitened_test[s]
#         else:
#             merged_test = np.append(merged_test, whitened_test[s], 1)
#     del whitened_test
#     # Shuffle the repetitions of different sessions
#     idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
#     merged_test = merged_test[:,idx]
#     # Insert the data into a dictionary
#     test_dict = {
#         'preprocessed_eeg_data': merged_test,
#         'ch_names': ch_names,
#         'times': times
#     }
#     del merged_test
#     # Saving directories
#     save_dir = os.path.join(args.project_dir, 'eeg_dataset',
#         'gamma', 'sub-'+format(args.sub,'02'))
#     file_name_test = 'preprocessed_eeg_test_gamma.npy'
#     file_name_train = 'preprocessed_eeg_training_gamma.npy'
#     # Create the directory if not existing and save the data
#     if os.path.isdir(save_dir) == False:
#         os.makedirs(save_dir)
#     np.save(os.path.join(save_dir, file_name_test), test_dict)
#     del test_dict
#
#     ### Merge and save the training data ###
#     for s in range(args.n_ses):
#         if s == 0:
#             white_data = whitened_train[s]
#             img_cond = img_conditions_train[s]
#         else:
#             white_data = np.append(white_data, whitened_train[s], 0)
#             img_cond = np.append(img_cond, img_conditions_train[s], 0)
#     del whitened_train, img_conditions_train
#     # Data matrix of shape:
#     # Image conditions × EEG repetitions × EEG channels × EEG time points
#     merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
#         white_data.shape[2],white_data.shape[3]))
#     for i in range(len(np.unique(img_cond))):
#         # Find the indices of the selected category
#         idx = np.where(img_cond == i+1)[0]
#         for r in range(len(idx)):
#             if r == 0:
#                 ordered_data = white_data[idx[r]]
#             else:
#                 ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
#         merged_train[i] = ordered_data
#     # Shuffle the repetitions of different sessions
#     idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
#     merged_train = merged_train[:,idx]
#     # Insert the data into a dictionary
#     train_dict = {
#         'preprocessed_eeg_data': merged_train,
#         'ch_names': ch_names,
#         'times': times
#     }
#     del merged_train
#     # Create the directory if not existing and save the data
#     if os.path.isdir(save_dir) == False:
#         os.makedirs(save_dir)
#     np.save(os.path.join(save_dir, file_name_train),
#         train_dict)
#     del train_dict
#%%
def save_prepr(args, epoched_test, epoched_train, img_conditions_train, ch_names, times, seed):
    """Merge and save the EEG data of all sessions together."""

    ### Merge and save the test data ###
    for s in range(args.n_ses):
        if s == 0:
            merged_test = epoched_test[s]
        else:
            merged_test = np.append(merged_test, epoched_test[s], axis=1)
    del epoched_test

    # Verify channel count before saving
    print(f"Test data shape: {merged_test.shape}")  # Debug print
    assert merged_test.shape[2] == len(ch_names), f"Channel count mismatch: {merged_test.shape[2]} vs {len(ch_names)}"

    idx = shuffle(np.arange(merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:, idx]

    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names,
        'times': times
    }

    save_dir = os.path.join(args.project_dir, 'eeg_dataset',
                            'higamma', 'sub-' + format(args.sub, '02'))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #np.save(os.path.join(save_dir, 'preprocessed_eeg_test.npy'), test_dict)
    del test_dict, merged_test

    ### Merge and save the training data ###
    for s in range(args.n_ses):
        if s == 0:
            white_data = epoched_train[s]
            img_cond = img_conditions_train[s]
        else:
            white_data = np.append(white_data, epoched_train[s], axis=0)
            img_cond = np.append(img_cond, img_conditions_train[s], axis=0)

    # Verify channel count in training data
    #print(f"Training data shape: {white_data.shape}")  # Debug print
    assert white_data.shape[2] == len(ch_names), f"Channel count mismatch: {white_data.shape[2]} vs {len(ch_names)}"

    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1] * 2,
                             len(ch_names), white_data.shape[3]))

    for i in range(len(np.unique(img_cond))):
        idx = np.where(img_cond == i + 1)[0]
        for r in range(len(idx)):
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], axis=0)
        merged_train[i] = ordered_data

    idx = shuffle(np.arange(merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:, idx]

    train_dict = {
        'preprocessed_eeg_data': merged_train,
        'ch_names': ch_names,
        'times': times
    }

    #np.save(os.path.join(save_dir, 'preprocessed_eeg_training.npy'), train_dict)
    #np.savez_compressed(
    #    os.path.join(save_dir, 'preprocessed_eeg_training.npz'),
    #    data=train_dict,
    #    allow_pickle=True,
    #    pickle_protocol=5
    #)
