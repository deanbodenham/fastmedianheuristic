from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

# including the bit about 'NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'
# to avoid warning after a compile; 
# solution found: https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

setup(
    ext_modules = cythonize([Extension("medianheuristic",
                             sources=["medianheuristic.pyx"],
                             include_dirs=[numpy.get_include()],
                             language='c++')], 
                             language_level=3)
)


#                              define_macros=[('NPY_NO_DEPRECATED_API')]
#define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
