from libcpp.vector cimport vector

cdef extern from "cpp_median.cpp":
    pass

cdef extern from "cpp_median.h" namespace "mhnamespace":
    double cpp_medianHeuristic(vector[double]);
