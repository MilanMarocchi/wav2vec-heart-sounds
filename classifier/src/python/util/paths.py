"""
    paths.py

    returns the path to the matlab code
    AUthor : Milan Marocchi
"""

import os

# The paths of this project
UTIL_PATH = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = os.path.dirname(UTIL_PATH)
PROCESSING_PATH = os.path.join(PYTHON_PATH, "processing")
ROOT = os.path.dirname(PYTHON_PATH)
MATLAB_PATH = os.path.abspath(os.path.join(ROOT, "matlab"))

# path to ephnogram and mit
EPHNOGRAM = '/home/mmaro/dev/heart_proj/data/ephnogram/WFDB/'
MIT = '/home/mmaro/dev/heart_proj/data/mit-bih-noise-stress-test-database-1.0.0'
