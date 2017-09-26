"""
MSc Python Intro
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# %%
# Functions and Control Flow
# Python functions are started using the def keyword,
# and typically return a value to the user.

# Complete this function so that it returns a string;
# 'green' when load is less than 0.7
# 'amber' when load is greater than or equal to 0.7, but less than 0.9
# 'red' when load is greater than or equal to 0.9.

# use the if, elif and else commands. Remember that Python uses indentation
# (tabs and spaces) to mark pieces of code inside defs, ifs, loops etc.

# test the function in the IPython Console by first executing this cell, then
# typing the name of the function with different numbers as the argument

def traffic_light(load):
    """Returns a string denoting the colour of a traffic light in response
    to floating point number load"""
    if ...
        outputString = 'green'
    elif  ...
        outputString = 'amber'
    ...
    ...
    return outputString

# %%

# define a sampling frequency (in Hz)
fs = 44100

# sampling interval (in seconds)
dt = 1/fs

# define a total length for the time base (in seconds)
T_total = 1.
# create a time base using 'np.linspace' command
# - linspace syntax: (start, end, number_of_points)
t = np.linspace(0, T_total-dt, fs*T_total)

# is this the best way to make a time vector?
# I prefer:
# t = np.arange(0,T_total,dt) [start,stop,step]
# as it's easier to remember and
# makes more semantic sense with the underlying data structure, i.e.
# every tick is dt in length, from 0, up to but not including T_total.
# np.diff(t)[0] = dt and max(t) = t[-1] = T-total-dt like before


# open a new figure and plot the time base
plt.figure(1)
plt.plot(t)
plt.title("Time base vector")
plt.xlabel("samples")
plt.ylabel("time [s]")

# sine wave frequency
f0 = 500

# create the sine wave
x = np.sin(2*np.pi*f0*t)

plt.figure(2)
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
# --> in OSX:     audioApi = 'CoreAudio'

duplexAudio(x*gain, fs, blockLength, audioApi='ALSA')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
import scipy.io.wavfile as wavio

from time import sleep

# obtain the sampling frequency and the bitstream
# if you get a 'FileNotFoundError',
# check if the current working directory is correct
fs, x_raw = wavio.read("file1.wav")

# Convert the samples to floating-point numbers
x_wav = wavToFloat(x_raw)

# Plot the waveform
t = np.linspace(0, (x_wav.shape[0]-1)/fs, x_wav.shape[0])

plt.figure(3)
plt.plot(t, x_wav, "b-")
plt.title("file1")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Can you write a line of code to listen to the file?
# Type duplexAudio? in the IPython console to receive the function signature
duplexAudio(...)

# Now, import file2.wav, convert it, and listen to it
fs, x2_raw = wavio.read("file2.wav")
x2_wav = wavToFloat(x2_raw)
duplexAudio(x2_wav,fs,512) # obfuscate this

# Do you recognise it?
# How has this signal been made?
# Can you transform this signal to turn it back into file1 again?
# Use the Python help browser or the internet to find appropriate functions

x2_wav_reversed = np.flipud(x2_wav) # reverse the signal
from scipy.signal import resample # resample it to twice the original sample rate
x2_wav_reversed_speed = resample(x2_wav_reversed, np.size(x2_wav_reversed)//2)

duplexAudio(x2_wav_reversed_speed, fs, 512) # obfuscate this - this is one way
# can also achieve speed-up by doubling the sample rate at playback.


x_wav_2 = np.tile(x_wav,2)

sleep(0.5)
duplexAudio(x_wav_2,fs,512)

sleep(0.5)
duplexAudio(x_wav,fs*2,512)

# %%

# end of bkgd noise - found by visual inspection
t1 = 0.065
n1 = int(t1*fs)

# end of 1st period of signal - found by visual inspection
t2 = 0.07
n2 = int(t2*fs)

x1 = x_wav[n1:n2]
t1 = t[n1:n2]

plt.figure(4)
plt.plot(t1, x1)

x_max = np.max(x1)
n_max = np.argwhere(x1 == x_max)

plt.plot(t1[n_max], x1[n_max], 'rs')


# %%
# fourier sine series of sawtooth wave

def bm_saw(M):
    '''Create the first M fourier series coefficients for a sawtooth wave'''
    bm = np.zeros(M)  # Make a new array for the M coefficients
    # the first bm coefficient is always zero,
    # so we only need to set the indices 1 to M-1
    # remember numpy arrays are indexed starting from zero
    m = np.arange(1, M)  # Make an array of numbers from 1 to M-1
    # Calculate b_m coefficients from eq. 1. to all m's simultaneously
    bm[1:M] = -2*((-1)**m)/(np.pi*m)
    return bm


def fourier_sine(f0, bn, t):
    '''Compute a signal x from sinusoidal fourier components

        f0 - scalar - fundamental frequency of sinusoidal components
        bn - array - fourier sine coefficients
        t - array - time vector
    '''
    x = np.zeros(t.shape[0]) # make an output array in the same shape as t

    for n, b in enumerate(bn):
        x += b*np.sin(2*np.pi*n*f0*t)

    return x

def saw(t,fsaw,a=1,p=0):
    '''Sawtooth wave generator
        Frequency fsaw
        Amplitude a
        Phase p (between 0 and 2pi)

        to match monotonic time base vector t
    '''
    return ((t+(p/fsaw)/(2*np.pi)) % ((1/fsaw)) - (1 / (2*fsaw))) * (2*a*fsaw)


fs = 44100
dt = 1/fs
T_max = 0.4

f0 = 400

t = np.linspace(0, T_max-dt, fs*T_max)

# or np.arange(0,T_max,dt)

#FFT parameters
N_dft = 2048
df = fs/N_dft
freq = np.linspace(0, fs-df, N_dft)



for N_saw in range(5):
    print('N_saw = {}'.format(N_saw))
    saw_sine = fourier_sine(f0, bm_saw(N_saw), t)
    plt.figure(5)
    plt.plot(t[:2*fs//f0], saw_sine[:2*fs//f0])

    SAW_SINE = np.fft.fft(saw_sine, n=N_dft)

    plt.figure(6)
    plt.semilogx(freq[:N_dft//2], 20*np.log10(np.abs(SAW_SINE[:N_dft//2])))
    duplexAudio(outputSignal=saw_sine,
                samplingFrequency=fs,
                blockLength=512,
                audioApi='ALSA')

saw_true = saw(t,f0,p=np.pi)
plt.figure(5)
plt.plot(t[:2*fs//f0],saw_true[:2*fs//f0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
SAW_TRUE = np.fft.fft(saw_true,n=N_dft)
plt.figure(6)
plt.semilogx(freq[:N_dft//2], 20*np.log10(np.abs(SAW_TRUE[:N_dft//2])),ls='--')
plt.xlabel('Frequency (Hz)')

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
plt.figure(7)
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
plt.figure(8)
plt.semilogx(freq[:N_dft//2], 20*np.log10(np.abs(y_f[:N_dft//2])))

