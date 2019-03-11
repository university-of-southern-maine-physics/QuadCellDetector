from setuptools import setup

setup(name='quadcelldetector',
      description='A library to simulate the electronic output resulting from the \ 
                   passage of a gaussian beam over a quadrant cell photodiode',
      version='0.1',
      url='https://github.com/paulnakroshis/QuadCellDetector',
      author='Paul Nakroshis',
      packages=['quadcelldetector'],
      licence='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      install_requires=[
                        'scipy',
                        'numpy',
                        'matplotlib'
                       ],
      zip_safe=False)
