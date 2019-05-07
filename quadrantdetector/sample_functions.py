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


def sin_path(t, d0):
    """
    Describes the y values of a sine wave given a "position" t and detector
    diameter d0.

    """
    return 0.5 * d0 * np.sin(2 * np.pi * t / 2.0)


def center_path(x, d0):
    """
    Describes the y values of a a straight path across the center of a
    detector given a position x and detector diameter d0.

    """
    return 0.0


def half_path(x, d0):
    """
    Describes the y values of a a straight path halfway above the center of a
    detector given a position x and detector diameter d0.

    """
    return d0 / 4


def quarter_path(x, d0):
    """
    Describes the y values of a a straight path a quarter above the center of a
    detector given a position x and detector diameter d0.

    """
    return d0 / 8
