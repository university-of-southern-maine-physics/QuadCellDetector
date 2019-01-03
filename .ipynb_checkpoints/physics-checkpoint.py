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
from numba import jit
import time

def laser(x, y, x_c, y_c, σ):
    """
    This cute function uses the wondrousness of NumPy to produce a gaussian
    beam in one line of code. The resulting array will be masked in the
    compute_signals function to eliminate the beam outside the detector
    and also to account for the dead zones due to the gap.

    INPUTS:
    X, Y     :  N by N position arrays containing the detector grid points
    x_c, y_c :  x and y coordinates of the center of the laser spot
                (not necessarily on the detector!)
    σ        :  the standard deviation for the gaussian beam; FWHM ~ 2.355σ

    RETURNS:
    Array of normalized beam intensity values over the detector array
    """

    beam = 1/(2*np.pi*σ**2)*np.exp(-((x-x_c)**2 + (y-y_c)**2)/(2*σ**2))
    return beam


def n_critical(d0, δ):
    """
    This function computes the smallest even integer value for the number of
    cells, n_crit, is such that no more than 2 complete cells fall within the
    detector gap width δ (i.e. Δ = δ/2 yielding N = 2 d0/δ)
    The code makes sure that N is even.
    INPUTS:

    d0      : The diameter of the quadrant cell photodiode (in mm)
    δ       : The gap distance between the quadrants of the photodiode (in mm)

    RETURNS:
    n_crit  : the critical number of cells (an even number)

    """
    n_crit = int(2*d0/δ)
    if n_crit % 2 != 0:
        n_crit = n_crit + 1
    return n_crit


@jit
def create_detector(n, d0, δ, ϵ=1e-14):
    """
    This routine creates the entire detector array.
    It does so by assuming a square array and eliminating chunks
    not within the circular detector boundary.
    INPUTS:
    n : number of chucks to divide detector into
    d0: diameter of full detector (in mm)
    δ : gap width betweeb the quadrants of the detector (in mm)
    ϵ : fudge factor needed for roundoff error (default = 1e-14)

    RETURNS:
    x,y               : 2d arrays of x and y coorinates
    active_area       : 2d array with effective area of each active cell

    IMPORTANT CONCEPTUAL NOTE:
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
    #
    # The next two lines set up the grid; the entire routine is
    # designed so that the centers of the grid points are
    # symmetrically placed about the origin. The closed grid point
    # centers are always Δ/2 away from the x or y axes.
    #
    Δ = d0/n
    y, x = np.mgrid[-d0/2 + Δ/2:d0/2:Δ, -d0/2 + Δ/2:d0/2:Δ]
    # This computes the distance of each grid point from the origin
    # and then we extract a masked array of points where r_sqr is less
    # than the distance of each grid point from the origin:
    r_sqr = x**2 + y**2
    inside = ma.getmask(ma.masked_where(r_sqr <= (d0/2)**2, x))

    """
    This portion takes care of masking out elements of the detector where
    the gap exists. It returns an array of light intensity over the detector.
    """
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
    INPUTS:
    n            : number of cells to divide diameter up into
    d0           : diameter of detector in mm
    δ            : gap width between quadrants of detector in mm
    beam         : array of laser beam intensity
    plot_signal  : Boolean value whether to produce a plot or not

    RETURNS:
    sum_signal : sum of all 4 quadrants
    l_r        : left minus right quadrants
    t_b        : top minus bottom quadrants
    """
    import numpy as np
    x, y, area = create_detector(n, d0, δ)
    signal = beam*area
    sum_signal = np.sum(signal)
    x_shape, y_shape = signal.shape
    right = np.sum(signal[0:x_shape, int(y_shape/2):y_shape])
    left = np.sum(signal[0:x_shape, 0:int(y_shape/2)])
    bottom = np.sum(signal[0:int(x_shape/2), 0:y_shape])
    top = np.sum(signal[int(x_shape/2):x_shape, 0:y_shape])
    l_r = left - right
    t_b = top - bottom
    if(plot_signal):
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


@jit
def signal_over_path(n, d0, δ, xmax, σ, track, n_samples,  ϵ = 1e-14):
    """
    This routine executes the compute_signals function multiple times over
    a user specified path function and returns the path and the expected
    signals.
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


@jit
def signal_over_time(n, d0, δ, ϵ, tmax, σ, track, n_samples, amplitude):
    """
    This routine executes the compute_signals function multiple times over
    a user specifiedtime interval and returns the path and the expected
    signals.
    """
    period = 40.00  # approx. period of our calibration ring pendulum
    tp = np.linspace(0, tmax, n_samples)     # create time array
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


def periodogram_psd(v, dt):
    sample_freq = 1/dt
    f, psd = signal.periodogram(v, sample_freq,
                                detrend="constant", scaling='spectrum')
    return f, psd


def SIN_PATH(x, d0):
    return 0.5*d0*np.sin(2*np.pi*x/0.5)


def CENTER_PATH(x, d0):
    return 0.0

@jit
def HALF_PATH(x, d0):
    return (d0/4)


def THREEQUARTER(x, d0):
    return 3*d0/8

def power_spectrum(n, d0, δ, ϵ, tmax, σ, track_func, n_samples, amplitude):
    """
    This code uses the functions in physics.py to build a detector and
    simulate the motion of our laser on some path across the detector.
    """
    import time
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
    Δ = d0/n    # grid size for simulation
#    tmax        # maximum time to run for
#    n_samples   # number of samples for path in time
#    track_func  # path of laser across detector
#    amplitude   # amplitude of oscillation in mm
    print("Building: ", n, "by", n, " Array")
    print("Pixel Size: Δ = %.3f " % (Δ))

#   Now build the detector and return the detector response for a gaussian
#   beam centered at (xc, yc) illumninating the detector.

    start_time = time.time()
    tp, xp, s, lr, tb = signal_over_time(n, d0, δ, ϵ, tmax, σ,
                                        track_func, n_samples, amplitude)
    f, psd = periodogram_psd(lr, tmax/n_samples)
    print("Runtime = %s seconds ---" % round(time.time() - start_time, 2))
    return tp, xp, s, lr, tb, f, psd
    
def plot_power_spectrum(tp, xp, s, lr, tb, f, psd):
    import matplotlib.pyplot as plt
    σ = 0.32
    plt.rcParams["figure.figsize"] = [16, 9]	
    plt.subplot(121)
    plt.plot(tp, lr, '-g', label='left-right')
    plt.ylabel(r'Detector Signals', fontsize=16, color='b')
    plt.xlabel(r'time (sec)', fontsize=16, color='b')
    plt.legend(fontsize=12, loc=(1.05, 0.875))
    plt.title(r'\textbf{Left - Right Signals: laser diameter = %.3f}' % (σ),
              fontsize=12)
    tmax = tp[-1]
    plt.xlim(0, tmax)
    #np.savetxt('L-R.txt', list(zip(tp, lr)), fmt='%.4f')
    #np.savetxt('psd.txt', list(zip(f, np.sqrt(psd))), fmt='%.4f')
    plt.subplot(122)
    plt.plot(f, np.sqrt(psd))
    plt.ylabel(r'Power Spectrum', fontsize=16)
    plt.xlabel(r'frequency (Hz)', fontsize=16)
    plt.grid()
    plt.xlim(0, 0.2)
    
    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.show()
    return 