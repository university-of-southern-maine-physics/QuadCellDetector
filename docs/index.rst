.. quadcelldetector documentation master file, created by
   sphinx-quickstart on Mon May  6 12:03:04 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to `quadcelldetector`
============================================

`QuadCellDetector` is a python package that allows one to simulate the response of a quadrant cell photodiode to the passage of a Gaussian laser beam across its surface.

The goals of this package are to allow the user to  

1.  Simulate the response of a given circular photodiode to the passage of a Gaussian beam that traverses the detector in *any* user specified path.  
2.  Use this simulation to match the output of the detector when an actual beam is manually swept across the detector; since the only adjustable parameter in this simulation is the width of the Gaussian beam, is is a simple matter to vary this parameter until one obtains a match to experimental data.  
3.  Use the simulation to be able to study the frequency content of a given spot's path across the detector.   

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :glob:

    How To Install <install>
    API Documentation <modules>


