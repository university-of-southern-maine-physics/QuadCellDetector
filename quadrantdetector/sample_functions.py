def welch_psd(V, dt):
    sample_freq = 1/dt
    f, psd = welch(V, sample_freq, window='hanning', nperseg=256,
                   detrend='constant')
    return f, psd


def periodogram_psd(v, fs):
    f, psd = signal.periodogram(v, fs,
                                detrend="constant", scaling='spectrum')
    return f, psd


def SIN_PATH(t, d0):
    return 0.5*d0*np.sin(2*np.pi*t/2.0)


def CENTER_PATH(x, d0):
    return 0.0


def HALF_PATH(x, d0):
    return (d0/4)


def QUARTER_PATH(x, d0):
    return d0/8