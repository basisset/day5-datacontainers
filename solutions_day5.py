#!/usr/bin/env python3

import numpy as np
from scipy import linalg

A = np.matrix('1 2 3; 4 5 6; 7 8 9')
b = np.array([1, 2, 3])

print(A)
#print(linalg.solve(A,b))


