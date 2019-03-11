from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='quadcelldetector',
      description='A library to simulate the electronic output resulting from the \ 
                   passage of a gaussian beam over a quadrant cell photodiode',
      long_description=long_description,
      long_description_content_type="text/markdown",
      version='0.1',
      url='https://github.com/paulnakroshis/QuadCellDetector',
      author='Paul Nakroshis',
      author_email="author@example.com",
      packages=['quadcelldetector'],
      licence='BSD 3-Clause License',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD 3-Clause License',
          'Programming Language :: Python :: 3'
      ],
      install_requires=[
                        'scipy',
                        'numpy',
                        'matplotlib'
                       ],
      zip_safe=False)
