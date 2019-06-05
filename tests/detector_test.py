import pytest
import numpy as np
import quadrantdetector.detector as qd
import quadrantdetector.sample_functions as qsf
from scipy import integrate

axis_size = 1000  # cells
detector_size = 16  #  diameter in mm


def intensity(y, x, sigma):
    """ Computes the intensity of a Gaussian beam centered on the origin at the position (x,y)"""
    return (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))


def total_signal(delta, sigma, R):
    """ Computes the theoretical intensity by sub-tracing off the signal lost due to the detector's
    finite size and the gap between the quadrants.
    """
    signal = integrate.dblquad(intensity, 0, R, 0,
                               lambda x: np.sqrt(R**2 - x**2), args=(sigma,))[0]
    gap1 = integrate.dblquad(intensity, 0, delta/2,  0,
                             lambda x: np.sqrt(R**2 - x**2), args=(sigma,))[0]
    gap2 = integrate.dblquad(intensity, 0, delta/2, 0, delta/2,
                             args=(sigma,))[0]
    return max(4 * signal - 8 * gap1 - 4 * gap2, 0)


@pytest.fixture(scope='session')
def get_detectors():
    """
    Returns a list of 100 detectors with increasingly large gaps, the last
    being gaps larger than the actual detector.
    """
    return [(gap, qd.create_detector(axis_size, detector_size, gap, outer_circular_mask=True))
            for gap in np.linspace(0, 1, 5)] \
     #   + [(gap, qd.create_detector(axis_size, detector_size, gap))
     #      for gap in np.linspace(1, 11, 50)]


def test_detector_init(get_detectors):
    for gap, detect in get_detectors:
        assert detect.shape == (axis_size, axis_size)

        density = axis_size / detector_size

        midpoint = int(axis_size / 2)
        # Positive value describing the number of zero-only cells in the width
        # of the detector gap.
        gap_width = min(int(gap * density / 2), int(axis_size / 2))
    
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
    # Test a laser on an empty grid.
    # In every laser on a detector, the sum of every point should be
    # approximately 1 when sigma << detector diameter.
    # Clearly, this only holds when the gap is small, otherwise the sum will
    # be even smaller.
    for sigma in np.arange(0.01, 7.02, 0.25):
        for gap, detect in get_detectors:
            laser = qd.laser(detect, detector_size / axis_size, 0, 0, sigma)
            sum_s = np.sum(laser * detect)

            # Note that when sum_s is increasingly large, we approach the expected value.
            # When it decreases, we fall away from our expected value.
            print(sum_s,',', sum_s - total_signal(gap, sigma, detector_size / 2))


def test_compute_signals(get_detectors):
    for gap, detect in get_detectors:
        # get_detectors created a sequence of detectors centered, so all
        # signals should be symmetric.
        sum_signal, lr_signal, tb_signal = qd.compute_signals(
            qd.laser(detect, detector_size / axis_size, 0, 0, 2.0), detect)
        assert sum_signal >= lr_signal and sum_signal >= tb_signal


def test_signal_over_path():
    # Deal with our linear tracks first.
    for track_func in [qsf.center_path, qsf.half_path, qsf.quarter_path]:
        # Run a track with a detector of diameter 10mm, from -20mm to +20mm
        # WLOG, sigma = 1, and we take 20 points along the track.
        x_positions, sum_signals, lr_signals, tb_signals = \
            qd.signal_over_path(axis_size, detector_size, 0, 20, 1, track_func, 40)

        for curr_x, curr_sum, curr_lr, curr_tb in zip(x_positions, sum_signals,
                                                      lr_signals, tb_signals):
            # In all cases...
            assert (curr_sum > curr_lr or abs(curr_sum - curr_lr) < 1e-6) \
               and (curr_sum >= curr_tb or abs(curr_sum - curr_tb) < 1e-6)

            # Check sums based on position.
            if curr_x >= -detector_size and curr_x <= detector_size:
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

