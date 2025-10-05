import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import ast

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal, axis=0)

def normalize_signal(signal):
    std = np.std(signal, axis=0)
    std[std == 0] = 1e-8
    return (signal - np.mean(signal, axis=0)) / std

def segment_signal(signal, window_size=2000, overlap=0.2):
    step = int(window_size * (1 - overlap))
    return np.array([
        signal[start:start + window_size]
        for start in range(0, signal.shape[0] - window_size + 1, step)
    ])

def process_folder(folder_path, max_files=100):
    all_segments, record_ids = [], []
    record_files = [
        os.path.join(root, f[:-4])
        for root, _, files in os.walk(folder_path)
        for f in files if f.endswith('.hea')
    ]
    if max_files:
        record_files = record_files[:max_files]

    for rec in record_files:
        try:
            signals, fields = wfdb.rdsamp(rec)
            leads = fields['sig_name']
            idx = [leads.index(l) for l in ['V1', 'V5', 'II']]
            ecg = signals[:, idx]
            ecg = bandpass_filter(ecg)
            ecg = normalize_signal(ecg)
            segs = segment_signal(ecg)
            segs = segs[:5]
            if segs.ndim != 3 or segs.shape[0] == 0:
                continue
            all_segments.append(segs)
            record_ids.extend([os.path.basename(rec)] * segs.shape[0])
        except Exception as e:
            print(f"Skipped {rec}: {e}")
    return np.vstack(all_segments), record_ids
