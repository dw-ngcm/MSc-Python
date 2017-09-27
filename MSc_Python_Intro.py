"""
MSc Sound and Vibration Studies
Tutorial Session on Python Programming

28 Sep 2017
"""

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.1 - Conditional Statements

def traffic_light(load):
    """
    Returns a string denoting the colour of a traffic light in response
    to floating point number 'load'
    """
    if load ...
        outputString = 'green'
    elif ...
        outputString = ...
    ...
        ...
    return outputString


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.2 - Creating and Plotting Signals
# import the necessary modules
import numpy as np
import matplotlib.pyplot as plt

# define a sampling frequency (in Hz)
fs = 44100

# sampling interval (in seconds)
dt = 1/fs

# total length for the time base (in seconds)
T_total = 0.5

# create a time base using 'np.arange' command
# -> syntax: np.arange(start, end (not included), step_size)
t = np.arange(0, T_total, dt)

# open a new figure and plot the time base
plt.figure()
plt.plot(t)
plt.title("Time base vector")
plt.xlabel("Index")
plt.ylabel("Time [s]")

# %%
# sine wave frequency
f0 = 500

# create and plot the sine wave signal
x = np.sin(2*np.pi*f0*t)

plt.figure()
plt.plot(t, x)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.3 - Playing Signals with pyaudio/duplexaudio

# no need to import 'pyaudio' explicitly;
# 'duplexAudio' does that internally
from duplexAudio import duplexAudio

# Scale the test signal to avoid clipping and numerical errors
gain = 0.5

# Set the block length for replaying the audio signal.
blockLength = 512

# Use the provided library function to play back the file
# using defaults for arguments (audio API, number of record channels, etc.)
# --> in Windows: audioApi = "MME"
# --> in Ubuntu:  audioApi = "ALSA"
# --> in OSX:     audioApi = "CoreAudio"

duplexAudio(x*gain, fs, blockLength, audioApi="MME")

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.4 - Fourier sine series of sawtooth wave


def bm_saw(M):
    '''Create the first M fourier series coefficients for a sawtooth wave'''

    bm = np.zeros(M)
    m = np.arange(1, M)
    bm[1:M] = -2*((-1)**m)/(np.pi*m)

    return bm


def fourier_sine(f0, bm, t):
    x = np.zeros(t.shape[0])

    x += bm[0]*np.sin(2*np.pi*0*f0*t)

    return x


# %%
# FFT parameters

N_dft = 2048
df = fs/N_dft
freq = np.linspace(0, fs-df, N_dft)

# calculate FFT (freq domain)
X = np.fft.fft(x, n=N_dft)

# plot using log scale on freq axis and magnitude in decibels
plt.figure()
plt.semilogx(freq[:N_dft//2], 20*np.log10(np.abs(X[:N_dft//2])))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.5 - Reading wav files

import scipy.io.wavfile as wavio
from wavToFloat import wavToFloat

# obtain the sampling frequency and the bitstream
fs, x_raw = wavio.read("file1.wav")

# Convert the samples to floating-point numbers
x_wav = wavToFloat(x_raw)

# create the time base
t = np.arange(0, x_raw.shape[0]/fs, 1/fs)

plt.figure()
plt.plot(t, x_wav, "b-")
plt.title("file1")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 2.6 Array indices and Array Content

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Section 3 - Advanced Exercise - Estimating Heart Rate

