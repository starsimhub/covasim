import numba as nb
import numpy   as np
import sciris as sc

@nb.njit((nb.int32,), cache=True)
def nb_poisson(rate): return np.random.poisson(rate, 1)[0]
def np_poisson(rate): return np.random.poisson(rate, 1)[0]

@nb.njit((nb.float32, nb.float32[:]), cache=True)
def nb_binomial_filter(prob, arr): return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]
def np_binomial_filter(prob, arr): return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]

@nb.njit((nb.int32, nb.int32), cache=True)
def nb_choose(max_n, n): return np.random.choice(max_n, n, replace=False)
def np_choose(max_n, n): return np.random.choice(max_n, n, replace=False)

@nb.njit((nb.int32, nb.int32), cache=True) # This hugely increases performance
def nb_choose_r(max_n, n): return np.random.choice(max_n, n, replace=True)
def np_choose_r(max_n, n): return np.random.choice(max_n, n, replace=True)


#%% Configure

repeats = int(1e4)
p_rate  = 5
prob    = 0.5
arr     = np.array(np.random.random(int(10e3)), dtype=np.float32)
max_n   = 10000
n       = 20

def time_both(label, nb_func, np_func, *args):
    sc.heading(label)
    with sc.Timer(label='Numba'):
        for i in range(repeats):
            out1 = nb_func(*args)
    with sc.Timer(label='Numpy'):
        for i in range(repeats):
            out2 = np_func(*args)
    return out1, out2

po1, po2 = time_both('poisson',  nb_poisson,         np_poisson,         p_rate)
fo1, fo2 = time_both('filter',   nb_binomial_filter, np_binomial_filter, prob, arr)
co1, co2 = time_both('choose',   nb_choose,          np_choose,          max_n, n)
ro1, ro2 = time_both('choose_r', nb_choose_r,        np_choose_r,        max_n, n)


#%% Example output
'''
——————————————————————————————
poisson
——————————————————————————————

Elapsed time for Numba: 0.00265 s
Elapsed time for Numpy: 0.0401 s



——————————————————————————————
filter
——————————————————————————————

Elapsed time for Numba: 0.939 s
Elapsed time for Numpy: 1.13 s



——————————————————————————————
choose
——————————————————————————————

Elapsed time for Numba: 0.0196 s
Elapsed time for Numpy: 1.43 s



——————————————————————————————
choose_r
——————————————————————————————

Elapsed time for Numba: 0.00635 s
Elapsed time for Numpy: 0.166 s
'''