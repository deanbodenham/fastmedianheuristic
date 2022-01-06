# Fast Median Heuristic Computation for Univariate Data

Implementation of `O(n log n)` algorithm for computing the median heuristic
for univariate data.

## Definition

Given **univariate** data `z_1, z_2, ..., z_n`, one can define the
following estimate of scale:

```
z_med = median {|z_i - z_j| : i < j, i,j = 1, 2, ..., n}
```

This estimator (or a scaled version of it) has been defined in the past in
the statistics literature; see Shamos (1976), Bickel and Lehmann (1979) and
Rousseuw and Croux (1993).



## Example

In Python:

```
import numpy as np
from medianheuristic import cy_medianHeuristic

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

mh = cy_medianHeuristic(X)
print("\nmedian heuristic:")
print(mh)
```

## Requirements

 - Python 3
 - Cython 
 - C++

First build/compile the module in the `code` 
subfolder using:
```
./run_build.sh
```

## Application

This quantity can be used when setting the value of the kernel parameter
in the MMD two sample tests. For the Laplacian kernel,

```
k(z_i, z_j) = exp ( -beta [z_i - z_j] )
```
one can set `beta = 1/z_med`.


For the Gaussian kernel,

```
k(z_i, z_j) = exp ( -alpha [z_i - z_j]^2 ),
```

one can simple set `alpha = 1/z_med^2`.

(Sorry about the formulae, unfortunately GitHub does not seem to support LaTeX
formulae yet.)


## Implementation

A naive implementation is clearly `O(n^2)` if all paris of differences
are computed; however, it is possible to compute `z_med` in `O(n log n)`.

The core method is in C++, and this is wrapped in Cython for Python.

It must be stressed that this is only applicable to univariate data;
Johnson and Mizoguchi (1978) provide an argument for the *d*-dimensional
case being `O(d n^d log n)`, but this is actually solving a more general problem,
and it is unclear if this bound can be improved on for the
*d*-dimensional median heuristic.



## References


 - Shamos, M. I. (1976) *Geometry and Statistics: Problems at the Interface*, in **Algorithms and Complexity**, Academic Press, Inc. [see Theorem 3.6]


 - Johnson, D. B. and Mizoguchi, T. (1978), Selecting the *K*th element in *X+Y* and *X_1+X_2+...+X_m*, SIAM Journal on Computing, 7, 2, 147-153.


 - Bickel, P. J. and Lehmann, E. L. (1979) *Descriptive statistics for nonparametric models IV. Spread*, in **Jaroslav Hajek Memorial Volume**, 519-526, Springer.
    [see Example 9]


 - Croux, C. and Rousseeuw, P. J. (1992) *Time-efficient algorithms for two highly robust estimators of scale*, in **Computational Statistics**, 411-428, Springer.


 - Rousseeuw, P. J. and Croux, C. (1993) *Alternatives to the median absolute deviation*, Journal of the American Statistical Association, 88, 424, 1273-1283.
