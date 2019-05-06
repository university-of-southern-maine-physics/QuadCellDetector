import numpy as np
import scipy.signal as signal
from scipy.signal import welch


def welch_psd(v, dt):
    sample_freq = 1 / dt
    f, psd = welch(v, sample_freq, window='hanning', nperseg=256,
                   detrend='constant')
    return f, psd


def periodogram_psd(v, fs):
    f, psd = signal.periodogram(v, fs,
                                detrend="constant", scaling='spectrum')
    return f, psd


def SIN_PATH(t, d0):
    """
    Describes the y values of a sine wave given a "position" t and detector
    diameter d0.

    """
    return 0.5 * d0 * np.sin(2 * np.pi * t / 2.0)


def CENTER_PATH(x, d0):
    return 0.0


def HALF_PATH(x, d0):
    return d0 / 4


def QUARTER_PATH(x, d0):
    return d0 / 8
