#!/usr/bin/env python3

import numpy as np
from scipy import linalg
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt

A = np.matrix('1 2 3; 4 5 6; 7 8 9')
b = np.array([1, 2, 3])


#print(linalg.solve(A,b)) ### c) Cannot be solved because its a singular matrix
### d) Equation cannot be solved because of previous answer

B = np.random.random((3,3))
b_2 = np.random.random((3,1))
#sol =  linalg.solve(B,b_2)
#print(f'Solution of B x = b_2 is:\n {sol}')

sol2 = np.matmul(linalg.inv(B),b_2)

print(f'Solution of B x = b_2 is::\n {sol2}')
print(f'B times x is:\n {np.matmul(B,sol2)}')

### Solving eigenvalue problem

eigenvals, eigenvecs = linalg.eig(A)
print(f'Eigenvalues of A are:\n {eigenvals}')
print(f'Eigenvectors (per column) of A are:\n {eigenvecs}')

print(f'Determinant of A: {linalg.det(A)}') ### Inversi is not defined since determinant is zero

print(f'Norm of A to order of 1: {linalg.norm(A,1)}')

print(f'Norm of A to order of 2: {linalg.norm(A,2)}')

print(f'Norm of A to order to infinity: {linalg.norm(A,np.inf)}')

### Statistics a)

fig, (ax1,ax2,ax3)= plt.subplots(3,1)

mu = 0.6
N = 1000

r = poisson.rvs(mu, size = N)
pmf = poisson.pmf(r, mu)
ax1.plot(pmf, label = 'poisson pmf')
cdf = poisson.cdf(r,mu)
ax2.plot(cdf, label = 'poisson cdf')
ax3.hist(r)

fig2, (ax4,ax5,ax6) = plt.subplots(3,1)

### b)
samples = norm.rvs(mu, size = N)
pdf_norm = norm.pdf(samples, mu)
ax4.plot(pdf_norm, label = 'normal pdf')
cdf_norm = norm.cdf(samples, mu)
ax5.plot(cdf_norm, label = 'normal cdf')
ax6.hist(samples)
#plt.show()

### c)









