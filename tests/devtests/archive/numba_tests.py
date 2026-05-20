'''
Speed and implementation tests
'''

import numpy as np
import pylab as pl
import sciris as sc
import numba as nb

# Set Numpy and Numba types (must match)
nptype = np.float32
nbtype = nb.float32

def mult(a, b):
    return (a * b).sum()

def cond(a, b):
    return np.logical_and(a>nptype(0.5), b>nptype(0.5)).sum()

@nb.njit((nbtype[:], nbtype[:]))
def mult_jit(a, b):
    return (a * b).sum()

@nb.njit((nbtype[:], nbtype[:]))
def cond_jit(a, b):
    return np.logical_and(a>nptype(0.5), b>nptype(0.5)).sum()


class NumbaTests(sc.prettyobj):
    def __init__(self, n=400e3, npts=1000, keys=None):
        if keys is None:
            keys = ['a','b']
        self.n = int(n)
        self.npts = npts
        self.keys = keys
        self.states = {key:np.array(np.random.random(self.n), dtype=nptype) for key in self.keys}
        self.results = np.zeros(self.npts)

    @property
    def a(self):
        return self.states['a']

    @property
    def b(self):
        return self.states['b']

    def next_mult(self, t):
        self.results[t] += mult(self.a, self.b)
        return

    def next_cond(self, t):
        self.results[t] += cond(self.a, self.b)
        return

    def next_mult_jit(self, t):
        self.results[t] += mult_jit(self.a, self.b)
        return

    def next_cond_jit(self, t):
        self.results[t] += cond_jit(self.a, self.b)
        return

    def run(self, which='mult', jit=True):
        for t in range(self.npts):
            if   which=='mult' and     jit: self.next_mult_jit(t)
            elif which=='mult' and not jit: self.next_mult(t)
            elif which=='cond' and     jit: self.next_cond_jit(t)
            elif which=='cond' and not jit: self.next_cond(t)
            else: raise Exception
        return


if __name__ == '__main__':

    ns = (1+np.arange(10))*100e3

    fig = pl.figure(figsize=(14,22))

    count = 0
    for which in ['mult', 'cond']:
        for jit in [0]: #[0,1]:
            print(which, jit)

            ts = []
            for n in ns:
                nt = NumbaTests(n=n)
                sc.tic()
                nt.run(which=which, jit=jit)
                t = sc.toc(output=True)
                ts.append(t)
                print(n, t)

            count += 1
            pl.subplot(4,2,count)
            pl.scatter(ns/1e6, ts)
            sc.setylim()
            pl.xlabel('Number of points (millions)')
            pl.ylabel('Calculation time')
            pl.title(f'Which = {which}, jit={jit}')

            count += 1
            pl.subplot(4,2,count)
            pl.scatter(ns/1e6, np.array(ts)/ns*1e6)
            sc.setylim()
            pl.xlabel('Number of points (millions)')
            pl.ylabel('Calculation time per point (ns)')

    pl.show()