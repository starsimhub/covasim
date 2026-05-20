import numba as nb
import numpy   as np
import sciris as sc

@nb.njit((nb.int32, nb.int32))
def nb32_choose(max_n, n):
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nb.int64, nb.int64))
def nb64_choose(max_n, n):
    return np.random.choice(max_n, n, replace=False)


def np_choose(max_n, n):
    return np.random.choice(max_n, n, replace=False)


#%% Configure

n = int(1e4)
m = 10000
s = 10

#%% Results

# Numba 0.48: Elapsed time: 0.0184 s
# Numba 0.49: Elapsed time: 0.942 s
sc.tic()
for i in range(n):
    nb32_choose(m, s)
sc.toc()

# Numba 0.48: Elapsed time: 0.0186 s
# Numba 0.49: Elapsed time: 0.946 s
sc.tic()
for i in range(n):
    nb64_choose(m, s)
sc.toc()

# Numba 0.48: Elapsed time: 1.40 s
# Numba 0.49: Elapsed time: 1.42 s (should be the same)
sc.tic()
for i in range(n):
    np_choose(m, s)
sc.toc()