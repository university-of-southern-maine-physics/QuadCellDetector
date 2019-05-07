How To Install
==============

Latest Version
--------------

To get the latest version from Github, you can use the following terminal
commands:

.. code-block:: shell

    git clone https://github.com/university-of-southern-maine-physics/QuadCellDetector.git
    cd QuadCellDetector
    pip install .

Modifying Documentation
-----------------------

Unless you are taking this package and modifying it to make it your own, you
will never need to touch this part.

First, you may want to install the plugins this documentation requires. In the
QuadCellDetector directory, you can use the following terminal commands

.. code-block:: shell

    cd docs
    pip install -r requirements.txt

to get a basic working installation.



If you want to modify the documentation (after you forked the project, for instance),
we use `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ with a collection of plugins specified
in `docs/conf.py <https://github.com/university-of-southern-maine-physics/QuadCellDetector/blob/master/docs/conf.py>`_.
We rely heavily on Automodule, but you don't have to!

Documentation is currently written in NumPy style, and Napoleon has been set
up to ignore other styles. You probably don't want to change this unless you
plan on using mixed documentation styles (bad idea), or you feel like
rewriting all of the documentation.

