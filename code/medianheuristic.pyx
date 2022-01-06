# distutils: language = c++

import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cimport medianheuristic

@cython.boundscheck(False)
@cython.wraparound(False)

def cy_medianHeuristic(X):
    return cpp_medianHeuristic(X)

