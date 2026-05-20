#!/bin/bash

set -e
cd $(dirname "$0")

# Run examples
CMD='covasim'
echo "running ${CMD}"; eval $CMD

CMD='covasim --interv=True'
echo "running ${CMD}"; eval $CMD

CMD='covasim --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"'
echo "running ${CMD}"; eval $CMD

CMD='covasim --interv=True --do_save=True'
echo "running ${CMD}"; eval $CMD

CMD='covascens'
echo "running ${CMD}"; eval $CMD

CMD="covascens --basepars \"{pop_size:5000}\" --metapars \"{n_runs:3, noise:0.1, noisepar:beta, rand_seed:1, quantiles:{low:0.1, high:0.9}}\""
echo "running ${CMD}"; eval $CMD

CMD='covascens --interv_day=35 --interv_eff=0.7'
echo "running ${CMD}"; eval $CMD

CMD='covascens --do_save=True'
echo "running ${CMD}"; eval $CMD