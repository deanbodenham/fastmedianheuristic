import numpy as np

def py_naiveMedianHeuristic(X):
    '''Naive method for finding n(n-1)/2th smallest element in 
       { |X-j - X_i| : i < j }
       Will be O( (n^2) log (n) ) because it is based on sorting
       the n(n-1)/2 elements

       Parameters:
            X (list) Array of numbers.


       Details:
            Uses sorting with mergesort on the distinct pairs
            (think: upper triangle).
    
       Returns:
            Absolute value of Kth smallest difference; among all distinct
            pairs of X, i.e. Kth smallest element in { |x_i - x_j| | i < j}.
    '''
    n = len(X)
    # number of nonzero elements; must be even
    nn = n * (n-1) // 2
    # median index
    m = nn // 2
    # -1 for python index, if perfect division
    r = nn % 2
    if (r == 0):
        m = m-1
    A = [0 for i in range(nn)]

    # fill up array A
    index = 0
    for i in range(1, n):
        for j in range(i):
            A[index] = abs(X[i] - X[j])
            index += 1
    # will use sort instead of select!
    A = np.sort(A)
    return( A[m] )

