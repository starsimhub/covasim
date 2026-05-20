'''
Explore different methods of goodness-of-fit calculation.
'''

import pylab as pl
import numpy   as np
import covasim as cv

np.random.seed(1)
x1 = abs(np.cumsum(np.random.randn(100)))
x2 = abs(np.cumsum(np.random.randn(100)))

e1 = cv.compute_gof(x1, x2) # Default, normalized absolute error
e2 = cv.compute_gof(x1, x2, normalize=False, use_frac=True) # Fractional error
e3 = cv.compute_gof(x1, x2, normalize=False, use_squared=True) # Squared error
e4 = cv.compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # MSE
e5 = cv.compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method

fig = pl.figure(figsize=(14,20), dpi=180)

pl.subplot(4,1,1)
pl.plot(x1, label='Actual')
pl.plot(x2, label='Predicted')
pl.legend()

pl.subplot(4,1,2)
pl.plot(e1, label='Default')
pl.legend()

pl.subplot(4,1,3)
pl.plot(e2, label='Fractional')
pl.legend()

pl.subplot(4,1,4)
pl.plot(e3, label='Squared error')
pl.axhline(e4, c=[0.1, 0.6, 0.1], lw=4, alpha=0.5, label='MSE')
pl.axhline(e5, c=[0.3, 0.1, 0.0], lw=1, alpha=1.0, label='Scikit-learn MSE')
pl.legend()