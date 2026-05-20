import covasim as cv
import matplotlib.pyplot as plt
import numpy as np

dur = cv.make_pars()['dur']

N = 50000
fig = plt.figure(figsize=(16,16))
for idx, (k, d) in enumerate(dur.items()):
    ax = plt.subplot(3,3,idx+1)
    vals = [cv.utils.sample(**d) for _ in range(N)]
    ax.hist(vals, density=True, bins=range(int(np.ceil(max(vals)+1))), align='left')
    mu = np.mean(vals)
    ax.plot([mu]*2, ax.get_ylim(), 'r')
    ax.set_title(f'{k}: {d["dist"]}({d["par1"]},{d["par2"]})')
plt.suptitle('Default distributions in Covasim', fontsize=20)
plt.show()
