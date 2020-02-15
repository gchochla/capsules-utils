'''Setup script
Usage: pip install .
To install development dependencies too, run pip install -e .[dev]
'''

from setuptools import find_packages
from setuptools import setup

from _version import __version__

setup(
    name='capsules_utils',
    version=__version__,
    packages=find_packages(),
    scripts=[],
    author='Georgios Chochlakis',
    url='https://github.com/gchochla/capsules_utils',
    install_requires=[],
    extras_require={
        'dev': [
            'torch',
            'torchvision',
            'matplotlib',
            'numpy',
        ]
    }
)