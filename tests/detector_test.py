
import pytest
import numpy as np
from quadrantdetector import detector


@pytest.fixture(scope='session')
def get_detectors():
    """
    Returns a list of 200 detectors with increasingly large gaps, the last
    being gaps larger than the actual detector.
    """
    return [(gap, detector.create_detector(1000, 20, 1e-12))
            for gap in np.linspace(1e-14, 21, 200)]
