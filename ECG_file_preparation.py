"""
Reads ECG signal from .mat file, filters it to "clean it up" and saves a a .wav
file.

Sep 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, wavfile

# %% read MAT-file

ecg_mat = loadmat('EMG_ECG')

fs = np.squeeze(ecg_mat['fs'])
y = np.squeeze(ecg_mat['y'])

y -= np.mean(y)

N = y.shape[0]

t = np.linspace(0, (N-1)/fs, N)

# plots raw signal
plt.figure()
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Raw signal')

# %% clean signal through filtering

# low pass order
N_lp = 4

# low pass cutoff freq
fc_lp = 15

# normalised cutoff frequency (to HALF of the sampling rate)
w0_lp = fc_lp/(fs/2.)

# create Butterworth low-pass filter
b_butter_lp, a_butter_lp = signal.butter(N_lp, w0_lp, btype='lowpass')

#
# high pass order
N_hp = 4

# high pass cutoff freq
fc_hp = 0.6

# normalised cutoff frequency (to HALF of the sampling rate)
w0_hp = fc_hp/(fs/2.)

# create Butterworth low-pass filter
b_butter_hp, a_butter_hp = signal.butter(N_hp, w0_hp, btype='highpass')

# apply the filters (use forward and backward filtering)
y_lp = signal.filtfilt(b_butter_lp, a_butter_lp, y)
y_lp_hp = signal.filtfilt(b_butter_hp, a_butter_hp, y_lp)

plt.figure()
plt.plot(t, y_lp_hp)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Filtered signal')

# create 16-bit wav file
N_bits = 16
wav_peak = 2**(N_bits-1) - 1
y_wavdata = np.int16((2*y_lp_hp)*wav_peak)

# save as .wav file
wavfile.write('ecg.wav', fs, y_wavdata)

# %%
"""
Heart Rate exercise
"""

from wavToFloat import wavToFloat

# read and check wav file
fs_wav, y_wav = wavfile.read('ecg.wav')
if y_wav.dtype is np.dtype(np.int16):
    print('File is 16-bit!')

# convert signal to float
y_ecg = wavToFloat(y_wav)

# plot signal and visually inspect for the first 5 peaks
plt.figure()
plt.plot(t, y_ecg)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('ECG signal')

# first 5 peaks are within the first 3 seconds (approx)
T_5peaks = 3.
N_5peaks = int(T_5peaks*fs)

t5 = t[:N_5peaks]
y_ecg5 = y_ecg[:N_5peaks]

plt.figure()
plt.plot(t5, y_ecg5, )
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('ECG signal (first 5 peaks)')

# find period for 1 peak (in sec and in samples)
T_1peak = T_5peaks/5
N_1peak = int(T_1peak*fs)

n_peaks = np.zeros(5, dtype='int')

for p in range(5):
    y_seg = y_ecg5[p*N_1peak:(p+1)*N_1peak]
    n_peaks[p] = np.argmax(y_seg) + int(N_1peak*p)

# mark peaks on plot
plt.plot(t[n_peaks], y_ecg[n_peaks], 'ro')

# find time instant of each peak
t_peaks = n_peaks/fs

#calculate average heart rate
T_avg = np.mean(np.diff(t_peaks))

print('The average heart rate is {:.1f} bpm'.format(60/T_avg))
