import numpy as np

from medianheuristic import cy_medianHeuristic

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

mh = cy_medianHeuristic(X)
print("\nmedian heuristic:")
print(mh)
