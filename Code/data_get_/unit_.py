import os
import mne
import numpy as np
import warnings

from . import selected_channel


# ! Data preprocessing
def process_data(file_path):
    mne.set_log_level('WARNING')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # read raw EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.resample(200.0)                                         # Resample to 200Hz
    raw.apply_function(lambda x: x * 1e6)                       # V -> μV
    for ch in raw.info['chs']:
        ch['unit'] = 105                                        # μV code
    raw.filter(l_freq=0.5, h_freq=45, method='fir',             # band filter [0.5, 40] Hz
                fir_window='hamming', n_jobs=2)
    selected_channels = selected_channel                        # selected channel
    available_channels = raw.info['ch_names']                   # isNon channel
    missing_channels = [ch for ch in selected_channels if ch not in available_channels]
    if missing_channels:
        print(f"file miss some channel :{missing_channels, skip.")
        return None
    else:
        raw_copy = raw.copy().pick(selected_channels)           # copy 
    data = raw_copy.get_data()                                  # (n_selected_channels, n_times)
    data_mean = np.mean(data, axis=1, keepdims=True)            # mean
    data_std = np.std(data, axis=1, keepdims=True)              # std
    data_normalized = (data - data_mean) / (data_std + 1e-10)   # Z-score 
    return data_normalized                                      # shape: (channel=16, n_times)


# ! read data len
def read_len_(dir_path='../Data_'):
    for i in range(1, 25):
        user_id = f"chb{i:02d}"
        user_dir = os.path.join(dir_path, user_id)
        for filename in ['label_0.npy', 'label_1.npy']:
            file_path = os.path.join(user_dir, filename)
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    # dim 2 len
                    if len(data.shape) < 2:
                        print(f"file {filename} data dim not enough, skip.")
                        continue
                    length = data.shape[1]
                    time_seconds = length / (200*60)  # Sampling rate 400 Hz
                    print(f"[{user_id}] file {filename}'s data len is: {time_seconds:.2f} min (sampling points: {length})")
                except Exception as e:
                    print(f"[{user_id}] read file {filename} error: {e}")
            else:
                print(f"[{user_id}] file {filename} no exist")




