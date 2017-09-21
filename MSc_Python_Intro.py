"""
MSc Python Intro
"""



# %%

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# define a sampling frequency (in Hz)
fs = 44100

# sampling interval (in seconds)
dt = 1/fs

# define a total length for the time base (in seconds)
T_total = 1.
# create a time base using 'np.linspace' command
# - linspace syntax: (start, end, number_of_points)
t = np.linspace(0, T_total-dt, fs*T_total)

# open a new figure and plot the time base
plt.figure()
plt.plot(t)
plt.title("Time base vector")
plt.xlabel("samples")
plt.ylabel("time [s]")

# sine wave frequency
f0 = 500

# create the sine wave
x = np.sin(2*np.pi*f0*t)

plt.figure()
plt.plot(t, x)
plt.title("Sine wave")
plt.xlabel("time [s]")
plt.ylabel("amplitude")

plt.plot(t, x, "ro")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

from duplexAudio import duplexAudio
from wavToFloat import wavToFloat

# Scale the test signal to avoid clipping and numerical errors
gain = 0.5

# Set the block length for replaying the audio signal.
blockLength = 512

# Use the provided library function to play back the file
# using defaults for arguments (audio API, number of record channels (0), etc.)
# --> in Windows: audioApi = 'MME'
# --> in Ubuntu:  audioApi = 'ALSA'

duplexAudio(x*gain, fs, blockLength, audioApi='ALSA')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
import scipy.io.wavfile as wavio

# obtain the sampling frequency and the bitstream
fs, x_raw = wavio.read("file1.wav")

# Convert the samples to floating-point numbers
x_wav = wavToFloat(x_raw)

# Plot the waveform
t = np.linspace(0, (x_wav.shape[0]-1)/fs, x_wav.shape[0])

plt.figure()
plt.plot(t, x_wav, "b-")
plt.title("file1")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# %%

# end of bkgd noise - found by visual inspection
t1 = 0.065
n1 = int(t1*fs)

# end of 1st period of signal - found by visual inspection
t2 = 0.07
n2 = int(t2*fs)

x1 = x_wav[n1:n2]
t1 = t[n1:n2]

plt.figure()
plt.plot(t1, x1)

x_max = np.max(x1)
n_max = np.argwhere(x1 == x_max)

plt.plot(t1[n_max], x1[n_max], 'rs')


# %%
# fourier sine series of sawtooth wave

def bm_saw(M):
    m = np.arange(M)
    bm = -2*((-1)**m)/(np.pi*m)

    bm[0] = 0

    return bm


def fourier_sine(f0, bn, t):
    x = np.zeros(t.shape[0])

    for n, b in enumerate(bn):
        x += b*np.sin(2*np.pi*n*f0*t)

    return x


fs = 44100
dt = 1/fs
T_max = 0.4

f0 = 1000

t = np.linspace(0, T_max-dt, fs*T_max)

plt.figure()

for N_saw in range(1, 5):
    saw_sine = fourier_sine(f0, bm_saw(N_saw), t)
    plt.plot(t[:2*fs//f0], saw_sine[:2*fs//f0])

    duplexAudio(saw_sine, fs, 512, audioApi='ALSA')

# %%
# create array with a single zero
x_silence = np.zeros(1)

# record length in seconds
rec_length = 1

# record data and save in array 'y'
y = duplexAudio(x_silence, fs, blockLength, recordLength=rec_length,
                audioApi='ALSA')

# create vector of time samples
t_rec = np.linspace(0, rec_length-(1/fs), rec_length*fs)

# plot vector 'y'
plt.figure()
plt.plot(t_rec, y)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# play the recorded file 'y'
duplexAudio(y, fs, blockLength, audioApi='ALSA')

# FFT params
N_dft = 2048
df = fs/N_dft
freq = np.linspace(0, fs-df, N_dft)

y_f = np.fft.fft(y, n=N_dft)

# plot spectrum of recorded signal
plt.figure()
plt.semilogx(freq[:N_dft//2], 20*np.log10(np.abs(y_f[:N_dft//2])))

