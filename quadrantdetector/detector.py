#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:06:26 2018
This code defines functions needed to create a gaussian laser beam
and mask it according to the specifications of our quandrant cell photodiode.
@author: paul nakroshis
"""
import numpy as np
import numpy.ma as ma
from scipy.signal import welch
import scipy.signal as signal


def laser(x, y, x_c, y_c, σ):
    """
    This cute function uses the wondrousness of NumPy to produce a gaussian
    beam in one line of code. The resulting array will be masked in the
    compute_signals function to eliminate the beam outside the detector
    and also to account for the dead zones due to the gap.

    Parameters
    ----------
    x, y : array_like
        N by N numpy position arrays containing the detector grid points
    x_c, y_c : float
        x and y coordinates of the center of the laser spot
        (not necessarily on the detector!)
    σ : float
        the standard deviation for the gaussian beam; FWHM ~ 2.355σ

    Returns
    -------
    array_like
        NumPy array of normalized beam intensity values over the detector array
    """
    return 1/(2 * np.pi * σ**2) \
        * np.exp(-((x - x_c)**2 + (y - y_c)**2) / (2 * σ**2))


def n_critical(d0, δ):
    """
    This function computes the smallest even integer value for the number of
    cells, n_crit, is such that no more than 2 complete cells fall within the
    detector gap width δ (i.e. Δ = δ/2 yielding N = 2 d0/δ)
    The code makes sure that N is even.

    Parameters
    ----------
    d0 : float
        The diameter of the quadrant cell photodiode (in mm)
    δ : float
        The gap distance between the quadrants of the photodiode (in mm)

    Returns
    -------
    n_crit : int
        The critical number of cells (an even number)

    """
    n_crit = int(2 * d0 / δ)
    if n_crit % 2 != 0:
        n_crit = n_crit + 1
    return n_crit


def create_detector(n, d0, δ, ϵ=1e-14):
    """
    This routine creates the entire detector array. It does so by assuming a
    square array and eliminating chunks not within the circular detector
    boundary.

    Parameters
    ----------
    n : int
        Number of chucks to divide detector into
    d0 : float
        Diameter of full detector (in mm)
    δ : float
        Gap width between the quadrants of the detector (in mm)
    ϵ : float
        Fudge factor needed for roundoff error (default = 1e-14)

    Returns
    -------
    x, y : array_like
        2d arrays of x and y coordinates
    active_area : array_like
        2d array with effective area of each active cell

    Note
    ----
    From the Numpy masking module np.ma manual:

    A masked array is the combination of a standard numpy.ndarray
    and a mask. A mask is either nomask, indicating that no value
    of the associated array is invalid, or an array of booleans
    that determines for each element of the associated array whether
    the value is valid or not.

    When an element of the mask is False, the corresponding element
    of the associated array is valid and is said to be unmasked.

    When an element of the mask is True, the corresponding element
    of the associated array is said to be masked (invalid).

    In other words, for the code below, I want to create an array of
    values where the laser beam's intensity is specified for all points
    within the detector's active radius. The easiest way to do this is to
    assign the value TRUE to all points within this radius. BUT, in numpy's
    way of thinking, these points are to be neglected (masked); however,
    since the logical equivalent to TRUE is 1, a mutiplication of this mask
    with the x and y arrays will yield an array with the points outside the
    detector eliminated. Techically, I am using my mask in the OPPOSITE manner
    in which numpy intends.
    """

    Δ = d0 / n
    y, x = np.mgrid[-d0/2 + Δ/2:d0/2:Δ, -d0/2 + Δ/2:d0/2:Δ]
    # This computes the distance of each grid point from the origin
    # and then we extract a masked array of points where r_sqr is less
    # than the distance of each grid point from the origin:
    r_sqr = x**2 + y**2
    inside = ma.getmask(ma.masked_where(r_sqr <= (d0/2)**2, x))

    # This portion takes care of masking out elements of the detector where
    # the gap exists. It returns an array of light intensity over the detector.

    all_dead = (np.abs(x) + Δ/2 - ϵ > δ/2) & (np.abs(y) + Δ/2 - ϵ > δ/2)

    partial_dead_x_only = (np.abs(x) + Δ/2 - ϵ > δ/2) & \
                          (np.abs(x) - Δ/2 - ϵ < δ/2) & \
                          (np.abs(y) - Δ/2 - ϵ > δ/2)
    partial_dead_y_only = (np.abs(y) + Δ/2 - ϵ > δ/2) & \
                          (np.abs(y) - Δ/2 - ϵ < δ/2) & \
                          (np.abs(x) - Δ/2 - ϵ > δ/2)
    partial_dead_x_or_y = (1/Δ)*(np.abs(x) + Δ/2 - δ/2)*partial_dead_x_only +\
                          (1/Δ)*(np.abs(y) + Δ/2 - δ/2)*partial_dead_y_only

    partial_dead_x_and_y = (1/Δ**2) * (np.abs(x) + Δ/2 - δ/2)**2 * \
        (
            (np.abs(x) + Δ/2 - ϵ > δ/2) &
            (np.abs(x) - Δ/2 - ϵ < δ/2) &
            (np.abs(y) + Δ/2 - ϵ > δ/2) &
            (np.abs(y) + Δ/2 - ϵ > δ/2) &
            (np.abs(y) - Δ/2 - ϵ < δ/2) &
            (np.abs(x) + Δ/2 - ϵ > δ/2)
        )

    gap_mask = all_dead      # not stricly needed, but
    partial_mask = partial_dead_x_or_y + partial_dead_x_and_y
    partial_mask[partial_mask == 0] = 1
    active_area = inside * gap_mask * partial_mask * Δ**2

    return x, y, active_area


def compute_signals(n, d0, δ, beam, plot_signal=False):
    """
    This routine computes--for a given beam intensity--
    the sum, left-right, and top-bottom signals.

    Parameters
    ----------
    n : int
        Number of cells to divide diameter up into
    d0 : float
        Diameter of detector in mm
    δ : float
        Gap width between quadrants of detector in mm
    beam : array_like
        Array of laser beam intensity
    plot_signal : bool, optional
        Whether to produce a Matplotlib plot or not

    Returns
    -------
    sum_signal : float
        Sum of all 4 quadrants
    l_r : float
        Left minus right quadrants
    t_b : float
        Top minus bottom quadrants
    """
    x, y, area = create_detector(n, d0, δ)
    signal = beam * area
    sum_signal = np.sum(signal)
    x_shape, y_shape = signal.shape
    right = np.sum(signal[0:x_shape, int(y_shape/2):y_shape])
    left = np.sum(signal[0:x_shape, 0:int(y_shape/2)])
    bottom = np.sum(signal[0:int(x_shape/2), 0:y_shape])
    top = np.sum(signal[int(x_shape/2):x_shape, 0:y_shape])
    l_r = left - right
    t_b = top - bottom

    if plot_signal:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(signal, origin='lower',
                   extent=[-d0/2, +d0/2, -d0/2, d0/2], cmap=plt.cm.Reds)
        plt.xlabel('y axis (mm)', fontsize=18)
        plt.ylabel('x axis (mm)', fontsize=18)
        plt.grid(linestyle='-', linewidth=0.25)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.show()

    return sum_signal, l_r, t_b


def signal_over_path(n, d0, δ, xmax, σ, track, n_samples, ϵ=1e-14):
    """
    This routine executes the compute_signals function multiple times over
    a user specified path function and returns the path and the expected
    signals.

    Parameters
    ----------
    n : int
        Number of cells to divide diameter up into
    d0 : float
        Diameter of detector in mm
    δ : float
        Gap width between quadrants of detector in mm
    xmax : float
        horizontal domain for sweeping across detector x from -xmax to +xmax
    σ : float
        Width of gaussian beam
    track :
        A function describing path across detector
    n_samples : int
        Number of samples in domain; dx = 2 * xmax / n_samples
    ϵ : float
        Fudge factor needed for roundoff error (default = 1e-14)

    Returns
    -------
    xp : array_like
        list of x coordinates for path
    sum_signal : array_like
        List of the sums of all 4 quadrants
    l_r : array_like
        List of the left minus right quadrants
    t_b : array_like
        List of the top minus bottom quadrants
    """
    xp = np.linspace(-xmax, xmax, n_samples)   # create x coordinate array
    x, y, area = create_detector(n, d0, δ, ϵ)  # create detector array
    sum_sig = []
    l_r = []
    t_b = []
    for x_val in np.nditer(xp):
        beam = laser(x, y, x_val, track(x_val, d0), σ)
        s, l, t = compute_signals(n, d0, δ, beam, plot_signal=False)
        sum_sig.append(s)
        l_r.append(l)
        t_b.append(t)
    return xp, sum_sig, l_r, t_b


def signal_over_time(n, d0, δ, tmax, σ, track, n_samples, amplitude, ϵ=1e-14):
    """
    This routine executes the compute_signals function multiple times over
    a user specified TIME interval and returns the path and the expected
    signals.
    This routine is more relevant to someone using a quadrant cell
    detector as a way top measure zero crossings of torsional pendulum
    undergoing angular oscillations with amplitude significantly greater that
    the effective angular amplitude defined by the detector. That being said,
    this function allows the users to specify the amplitude to any value
    desired.


    Parameters
    ----------
    n : int
        The number of cells to divide diameter up into
    d0 : float
        Diameter of detector in mm
    δ : float
        Gap width between quadrants of detector in mm
    tmax : float
        Maximum time for simulation in seconds
    σ : float
        width of gaussian beam in mm
    track
        name of function describing path across detector
    n_samples : int
        number of samples in time domain; dt = 2*tmax/n_samples

    Returns
    -------
    tp : array_like
        List of time values in seconds
    xp : array_like
        :ist of x coordinates for path
    sum_signal : float
        Sum of all 4 quadrants
    l_r : float
        Left minus right quadrants
    t_b : float
        Top minus bottom quadrants

    Notes
    -----
    1. the track function specifies the y-coordinate of the spot
    center as it tracks across the detector.
    2. The period of the motion is set to mimic our pendulum with
    its current torsion fiber (40 seconds)
    """
    period = 40.00  # approx. period of our calibration ring pendulum
    tp = np.linspace(0, tmax, n_samples)  # create time array
    xp = amplitude*np.sin(2*np.pi*tp/period)  # create x coordinate array
    yp = track(tp, d0)

    x, y, area = create_detector(n, d0, δ, ϵ)  # create detector array
    sum_sig = []
    l_r = []
    t_b = []

    for x_val, y_val in np.nditer([xp, yp]):
        beam = laser(x, y, x_val, y_val, σ)
        s, l, t = compute_signals(n, d0, δ, beam, plot_signal=False)
        sum_sig.append(s)
        l_r.append(l)
        t_b.append(t)

    return tp, xp, sum_sig, l_r, t_b


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


def power_spectrum(n, d0, δ, tmax, σ, track_func, n_samples, amplitude):
    """
    This code uses the functions in physics.py to build a detector and
    simulate the motion of our laser on some path across the detector.

    """
#   Physical Paramaters for our detector system:
#
#    d0  # diameter of photocell in mm
#    δ   # gap between the 4 quadrants; also in mm
#    σ   # measured gaussian radius of our laser beam
#
#   Simulation Parameters:
#
#    n           # choose a reasonable minimum n value
#    ϵ           # fudge factor due to roundoff error in case where δ = 2Δ
    Δ = d0/n     # grid size for simulation
#    tmax        # maximum time to run for
#    n_samples   # number of samples for path in time
#    track_func  # path of laser across detector
#    amplitude   # amplitude of oscillation in mm
    print("Building: ", n, "by", n, " Array")
    print("Pixel Size: Δ = %.3f " % (Δ))

#   Now build the detector and return the detector response for a gaussian
#   beam centered at (xc, yc) illumninating the detector.

    tp, xp, s, lr, tb = signal_over_time(n, d0, δ, tmax, σ,
                                         track_func, n_samples, amplitude)
    f_lr, psd_lr = periodogram_psd(lr, n_samples/tmax)
    f_tb, psd_tb = periodogram_psd(tb, n_samples/tmax)
    return tp, xp, s, lr, tb, f_lr, psd_lr, f_tb, psd_tb


def plot_power_spectrum(tp, xp, s, lr, tb, f, psd, σ=0.32, fmax=0.25,
                        p_label='Left-Right'):
    """
    This routine plots the (already computed) detector signal and it's power
    spectrum.
    """

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.subplot(121)
    plt.plot(tp, lr, '-g', label=p_label)
    plt.ylabel(r'Detector Signals', fontsize=16, color='b')
    plt.xlabel(r'time (sec)', fontsize=16, color='b')
    plt.legend(fontsize=12, loc=(1.05, 0.875))
    plt.title(r'\textbf{Signals: laser diameter = %.3f}' % (σ),
              fontsize=12)
    tmax = tp[-1]
    plt.xlim(0, tmax)
    plt.subplot(122)
    plt.plot(f, np.sqrt(psd))
    plt.ylabel(r'Power Spectrum', fontsize=16)
    plt.xlabel(r'frequency (Hz)', fontsize=16)
    plt.grid()
    plt.xlim(0, fmax)

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.show()
    return None
