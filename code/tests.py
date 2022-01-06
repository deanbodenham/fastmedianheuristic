import unittest
import numpy as np
from medianheuristic import cy_medianHeuristic

from naive_medianheuristic import py_naiveMedianHeuristic

from random import random
from random import seed

class medianHeuristicTests(unittest.TestCase):

    def test_mh_1(self):
        """Computing median heuristic
        """
        #print("test2.4")
        X = (np.array([1.12668299, -0.49653829, 0.51318307, 0.32565421, -0.22344055, -0.61588253, -0.28538632, -1.07281643, -0.65512623, 0.03339469, 1.35947697, -0.45829236, 2.11488536, 1.96697442, 3.36539357, -0.25610955, 3.57348141, 1.92885565, -0.80836105, 1.2439888]))
        ans = cy_medianHeuristic(X)
        soln = 1.41206931
        # check solution
        self.assertAlmostEqual(ans, soln, places=5, msg=None, delta=None)


    def test_mh_2(self):
        """Computing median heuristic
        """
        #np.random.seed(2)
        X = np.random.normal(0, 1, 200)
        #X =(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        soln = py_naiveMedianHeuristic(X)
        ans = cy_medianHeuristic(X)
        # check solution
        self.assertAlmostEqual(ans, soln, places=5, msg=None, delta=None)


    def test_median_heuristic_croux_rousseuw_2(self):
        n = 20
        X = [0.0 for i in range(n)]
        # uniform[a, b)
        a = -2
        b = 2
        seed(1)
        for i in range(n):
            X[i] = a + (b-a)*random()

        soln = py_naiveMedianHeuristic(X[:])
        ans = cy_medianHeuristic(X)
        self.assertAlmostEqual(ans, soln, places=5, msg=None, delta=None)


    def test_median_heuristic_croux_rousseuw_3(self):
        n = 3 
        X = [0.0 for i in range(n)]
        # uniform[a, b)
        a = -2
        b = 2
        seed(1)
        for i in range(n):
            X[i] = a + (b-a)*random()

        soln = py_naiveMedianHeuristic(X[:])
        ans = cy_medianHeuristic(X)
        self.assertAlmostEqual(ans, soln, places=5, msg=None, delta=None)


    def test_median_heuristic_croux_rousseuw_5(self):
        '''Now for a numpy vector
        '''
        n = 22
        np.random.seed(1)
        X = np.random.normal(n, 2, 4)

        soln = py_naiveMedianHeuristic(X[:])
        ans = cy_medianHeuristic(X)
        #print(ans)
        #print(soln)

        self.assertTrue(ans==soln)

if __name__ == '__main__':
    unittest.main()

