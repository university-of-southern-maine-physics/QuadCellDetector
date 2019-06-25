import pytest
import numpy as np
import quadrantdetector.detector as qd
import quadrantdetector.sample_functions as qsf
from scipy import integrate
import math as m
from pytest import approx

detector_n = 2000  # cells in simulation
detector_diameter = 10  # diameter in mm
max_gap = np.sqrt(2) * detector_diameter/2  # this is the largest possible gap size


def intensity(y, x, sigma):
    """ Computes the intensity of a Gaussian beam centered on the origin at the position (x,y)"""
    return (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))


def total_signal(delta, sigma, R):
    """ Computes the theoretical intensity by integration;
        Assumes a centered beam.
    """
    signal = 4 * integrate.dblquad(intensity,
                                   delta/2, np.sqrt(R**2 - 0.25*delta**2),
                                   delta/2, lambda x: np.sqrt(R**2 - x**2),
                                   args=(sigma,))[0]
    
    return signal


@pytest.fixture(scope='session')
def get_detectors():
    """
    Returns a list of 20 detectors with increasingly large gaps, the last
    being a gap of size max_gap = np.sqrt(2)*detector_diameter/2.
    """
    return [(gap, qd.create_detector(detector_n, detector_diameter, gap))
            for gap in np.linspace(0, 0.99 * max_gap, 20)]


def test_detector_init(get_detectors):
    for gap, detect in get_detectors:
        assert detect.shape == (detector_n, detector_n)

        density = detector_n / detector_diameter

        midpoint = int(detector_n / 2)
        # Positive value describing the number of zero-only cells in the width
        # of the detector gap.
        gap_width = min(int(gap * density / 2), int(detector_n / 2))

        # Sum the 2d areas that should all be zero. Remember, end bounds for
        # slicing are not inclusive.
        vertical_block = detect[midpoint - gap_width: midpoint + gap_width, :]
        horizontal_block = detect[:, midpoint - gap_width: midpoint + gap_width]

        # Common-sense check to make sure we did our slicing right.
        assert vertical_block.shape[0] == 2 * gap_width \
            and horizontal_block.shape[1] == 2 * gap_width

        # Must be an all-zero area.
        assert vertical_block.sum() == 0 and horizontal_block.sum() == 0


def test_laser(get_detectors):
    # this test creates a gaussian beam centered on the detector
    # and looks at the sum signal as produced by our computational model
    # of the detector and compares it to the theoretical model for the same beam.
    # The theoretical integral is relatively easy to compute in this case, but is 
    # not so easily computed when the beam is not centered.
    #
    # start by creating a ranged of beam radii, and then compute the sum signal
    # by our computational model:
    sigma_min = 0.01
    sigma_max = 20
    sigma_step = 0.5
    for sigma in np.arange(sigma_min, sigma_max, sigma_step):
        for gap, detect in get_detectors:
            laser = qd.laser(detector_diameter, detector_n, 0, 0, sigma)
            sum_s = np.sum(laser * detect)
            # now test so see that this is approximately equal to the theoretical value
            assert m.fabs(sum_s - total_signal(gap, sigma, detector_diameter / 2)) < 0.00001


def test_compute_signals(get_detectors):
    for gap, detect in get_detectors:
        # get_detectors created a sequence of detectors centered, so all
        # signals should be symmetric.
        sum_signal, lr_signal, tb_signal = qd.compute_signals(
            qd.laser(detector_diameter, detector_n, 0, 0, 2.0),
            detect)
        assert sum_signal >= lr_signal and sum_signal >= tb_signal


def test_signal_over_path():
    # Deal with our linear tracks first.
    for track_func in [qsf.center_path, qsf.half_path, qsf.quarter_path]:
        # Run a track with a detector of diameter 10mm, from -20mm to +20mm
        # WLOG, sigma = 1, and we take 20 points along the track.
        x_positions, sum_signals, lr_signals, tb_signals = \
            qd.signal_over_path(detector_n, detector_diameter, 0, 20, 1, track_func, 40)

        for curr_x, curr_sum, curr_lr, curr_tb in zip(x_positions, sum_signals,
                                                      lr_signals, tb_signals):
            # In all cases...
            assert (curr_sum > curr_lr or abs(curr_sum - curr_lr) < 1e-6) \
               and (curr_sum >= curr_tb or abs(curr_sum - curr_tb) < 1e-6)

            # Check sums based on position.
            if curr_x >= -detector_diameter and curr_x <= detector_diameter:
                assert curr_sum > 0
            else:
                assert curr_sum < 0.1

            if track_func == qsf.center_path:
                assert 0.1 > curr_tb > -0.1

            if track_func == qsf.half_path:
                if curr_x < 3 and -3 < curr_x:
                    # Should be noticably biased towards the top
                    assert 1 > curr_tb > 0.75

            if track_func == qsf.quarter_path:
                if curr_x < 3 and -3 < curr_x:
                    # Should be somewhat biased towards the top
                    assert 1 > curr_tb > 0.5


def test_signal_over_time():
    # Temp variables while this function gets built
    gap = 0
    amplitude = detector_diameter/2
    period = 8
    max_time = 8
    sigma = 1
    num_samples = 41

    for track_func in [qsf.center_path, qsf.half_path, qsf.quarter_path]:
        time_vals, x_vals, sum_s, lr_s, tb_s = qd.signal_over_time(
            detector_n, detector_diameter, gap,
            amplitude, period, max_time, sigma,
            track_func, num_samples)

        # Basic dimentionality and type check
        assert time_vals.shape == x_vals.shape == sum_s.shape == lr_s.shape\
            == tb_s.shape
            

        if track_func == qsf.center_path: 
            # Check case of small vertically centered spot which starts centered
            # on the detector and moves sinusoidally as Amplitude*sin(2*pi*t/period)
            # Consequently, all signals should be identical at t = 0 and t = period
            assert sum_s[0] == approx(sum_s[-1], abs=1e-6) 
            assert lr_s[0] == approx(lr_s[-1],  abs=1e-6)
            assert tb_s[0] == approx(tb_s[-1], abs=1e-6) 
            # now check that the sum signal and top-bottom signals at t=period/4 and 
            # t = 3*period/4 should be unchanged. Also the left - right signal should 
            # switch sign
            assert sum_s[10] == approx(sum_s[30],  abs=1e-6)
            assert tb_s[10] == approx(tb_s[30],  abs=1e-6)
            assert lr_s[10] == approx(-lr_s[30],  abs=1e-6)        
            
    # Now change the amplitude to have an amplitude 10 times larger than the detector
    # diameter.
    amplitude = 10*detector_diameter
    time_vals, x_vals, sum_s, lr_s, tb_s = qd.signal_over_time(
        detector_n, detector_diameter, gap,
        amplitude, period, max_time, sigma,
        track_func, num_samples)
    # Check case of small vertically centered spot which starts centered
    # on the detector and moves sinusoidally as Amplitude*sin(2*pi*t/period)
    # Consequently, all signals should be identical at t = 0 and t = period
    assert sum_s[0] == approx(sum_s[-1], abs=1e-6) 
    assert lr_s[0] == approx(lr_s[-1],  abs=1e-6)
    assert tb_s[0] == approx(tb_s[-1], abs=1e-6) 
    # now check that the sum signal and top-bottom signals at t=period/4 and 
    # t = 3*period/4 should be unchanged. Also the left - right signal should 
    # switch sign; also, all of these signals should be very small, as these 
    # points correspond to beams centered far from detector.
    assert sum_s[10] == approx(sum_s[30],  abs=1e-6) == approx(0, abs=1e-6)
    assert tb_s[10] == approx(tb_s[30],  abs=1e-6) == approx(0, abs=1e-6)
    assert lr_s[10] == approx(-lr_s[30],  abs=1e-6)   == approx(0, abs=1e-6)      
    
    
        