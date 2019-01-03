#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:06:26 2018
This code simulates the response of our quadrant cell photodiode to a gaussian
laser beam passing over the detector on a path specified by the user.
@author: paul nakroshis
"""

import physics as phy
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
plt.rcParams["figure.figsize"] = [8, 6]				
#import matplotlib
#matplotlib.use('TkAgg')



def main():
    """
    This code uses the functions in physics.py to build a detector and
    simulate the motion of our laser on some path across the detector.
    """
#   Physical Paramaters for our detector system:

    d0 = 7.80   # diameter of photocell in mm
    δ = 0.042   # gap between the 4 quadrants; also in mm
    σ = 0.320   # measured gaussian radius of our laser beam

#   Simulation Parameters:

    n = phy.n_critical(d0, δ)  # choose a reasonable minimum n value
    ϵ = 1E-14         # fudge factor due to roundoff error in case where δ = 2Δ
    Δ = d0/n          # grid size for simulation
    
    tmax = 600
    n_samples = 10*tmax   # number of samples for path in time
    track_func = phy.CENTER_PATH
    amplitude = 0.5  # amplitude of oscillation in mm
    print("Building: ", n, "by", n, " Array")
    print("Pixel Size: Δ = %.3f " % (Δ))

#   Now build the detector and return the detector response for a gaussian
#   beam centered at (xc, yc) illumninating the detector.

    start_time = time.time()
    tp, xp, s, lr, tb = phy.signal_over_time(n, d0, δ, ϵ, tmax, σ,
                                             track_func, n_samples, amplitude)
    f, psd = phy.periodogram_psd(lr, tmax/n_samples)

    print("Runtime = %s seconds ---" % round(time.time() - start_time, 2))
    #plt.figure(figsize=(8, 8))
    plt.rcParams["figure.figsize"] = [10, 6]	
    plt.subplot(121)
    plt.plot(tp, lr, '-g', label='left-right')
    plt.ylabel(r'Detector Signals', fontsize=16, color='b')
    plt.xlabel(r'time (sec)', fontsize=16, color='b')
    plt.legend(fontsize=12, loc=(1.05, 0.875))
    plt.title(r'\textbf{Left - Right Signals: laser diameter = %.3f}' % (σ),
              fontsize=12)
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
    

if __name__ == "__main__":
    main()
