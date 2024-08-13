from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(
    name='lldm',
    version='1.0',
    description=(
        'This is the official implementation for the paper: '
        '"Long Term Time Series Forecasting With Latent Linear Operators".'
    ),
    author='anonymouscodesubmission88',
    author_email='anonymouscodesubmission88@gmail.com',
    url='https://github.com/anonymouscodesubmission88/lldm',
    license='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Time-Series Forecasting',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    package_dir={'lldm': os.path.join(cwd, 'lldm')},
    packages=find_packages(
        exclude=[
            'data',
            'logs',
        ]
    ),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch',
        'torchvision',
        'tensorboard',
        'tqdm',
        'h5py',
        'pandas',
        'seaborn',
    ],
)
