"""
This script generates the rim crest, floor, and rim flank outlines of craters with diameters defined by diameter_array.
Note that:
    By setting the random seed to a certain value, the same crater will be generated each time. If random craters need
    to be generated, just comment out the random seed.
"""
import numpy as np
from utilities import create_synthetic_outlines
if __name__ == '__main__':
    np.random.seed(42)
    diameter_start = 10
    diameter_number = 10
    diameter_end = 100
    diameter_array = np.linspace(diameter_start, diameter_end, diameter_number)
    create_synthetic_outlines(diameter_array)

