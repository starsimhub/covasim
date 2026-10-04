---
name: covasim-migrate-v3-v4
description: Use when migrating code written for Covasim v3 (3.1.x or earlier) to Covasim v4, which is built on Starsim — including scripts, custom interventions and analyzers, and saved populations or sims. Also use when v3-era Covasim code fails or gives different results on v4.
---

# Migrating Covasim v3 code to v4

Covasim v4 is built on Starsim, but keeps the v3 API: most v3 scripts run unchanged. The goal of a migration is to change as few lines as possible, and to check that the results still match. Don't rewrite v3 code into Starsim style unless asked; `cv.Sim(pars)`, `sim['beta']`, `apply(self, sim)` and so on are all still supported (see `covasim/compat.py`).

The rules are in `docs/migrate3to4.md` in the Covasim repository (https://docs.covasim.org/migrate3to4.html). Read it before changing anything by hand.

## Steps

1. **Record the v3 results first, if v3 can still be run.** Before changing anything, run the code on v3 and save the numbers that matter (e.g. `sim.summary`, or the values plotted) for several seeds. v4 results are not identical to v3 for the same seed, so compare means over seeds, not single runs. To run v3 next to an installed v4, extract it somewhere (`git archive main | tar -x -C /tmp/cov-v3` in the Covasim repository) and run with `PYTHONPATH=/tmp/cov-v3`.
2. **Run the migration script without changing anything:** `covasim-migrate3to4 <files or folders>`. It prints a diff of the mechanical changes, then `CHECK` lines for anything that may need a change by hand.
3. **Apply the mechanical changes:** add `--apply`. These are safe: `sim.t` → `sim.ti`, `sim.pars['key']` → `sim['key']`, `sim.results['transtree']` → `sim.transtree`, and `sc.findinds(people.x)` → `cv.true(people.x)`. The files should be under version control, since they are changed in place.
4. **Go through each `CHECK` line.** Many need no change; the script can't tell. Decide using the rules below, and change only what is needed.
5. **Run the code on v4** with `COVASIM_WARNINGS=error`, fix what fails using `docs/migrate3to4.md`, and compare with the v3 results from step 1. If the code can't be run (e.g. missing data), say so, and say which changes are untested.
6. **Report** what was changed mechanically, what was changed by hand and why, what was left alone, and how the results compare.

## Judgment calls

### Agents who have died

This is the change most likely to give wrong results silently. In v4, agents who die are removed from the per-agent arrays, so `sim.people.age`, `sim.people.exposed`, `len(sim.people)` etc. only cover agents who are alive, and a position in an array is not a UID. The exceptions are `sim.people.dead` and `sim.people.date_dead`, which cover every agent ever created.

- Indexing by UID always works, for agents alive or dead: `people.age[cv.true(people.dead)]` needs no change.
- Positions are not UIDs: `sc.findinds(...)`, `np.nonzero(...)` and `np.where(...)` on a people array → `cv.true(...)` (or `.uids`).
- Combining `dead` elementwise with another array, or assuming a fixed length, needs `.raw`, which has the values for every agent: `people.age[people.dead]` → `people.age.raw[:people.n_uids][people.dead]`; `n_doses.append(people.doses.copy())` (to stack into a 2D array) → `people.doses.raw.copy()`.
- An array built with `len(sim.people)` and then indexed by UID is too short once anyone has died: `vals = np.ones(len(sim.people)); vals[people.true('severe')] *= 2` → either size it with `sim.people.n_uids`, or (for `subtarget` functions, which pair `inds` with `vals`) use `inds = sim.people.indices()` and boolean masks, e.g. `vals[np.asarray(people.severe)] *= 2`.
- If nobody dies in the sim, none of this matters; don't change code for a case that can't happen, but do check whether it can.

### Custom interventions and analyzers

- `apply(self, sim)`, `initialize(self, sim)` and `finalize(self, sim)` still work. Leave them.
- Starsim reserves the attributes `t`, `pars`, `sim`, `dists` and `results` on modules. `self.t = []` → `self.tvec = []` (and update where it's read). The script flags every `self.results = ...`, including in classes that aren't Covasim modules, where no change is needed.
- A `finalize(self)` without the `sim` argument must call `super().finalize()`.
- Module names must be unique across interventions, analyzers and networks. A clash from default names gives a warning and a rename; fix it by passing `label=`.
- Function interventions and analyzers (`def f(sim): ...`, lambdas) work as before.

### Populations

- `pop_type='synthpops'`, `popfile=`, `cv.make_people()` and v3 `.ppl` files are not supported. If the code only needs a realistic population, use `pop_type='hybrid'`.
- To keep a specific saved population, export it with v3 (each agent's age and sex, and `p1`, `p2`, `beta` for each layer) and rebuild it in v4: `people = cv.People(n)`, set `people.age.default` and `people.female.default` to functions returning the stored values, and pass `people=people, networks=[cv.Layer(name='h', edges=dict(p1=p1, p2=p2, beta=beta)), ...]` to `cv.Sim()`. Edits made to `sim.people.contacts` after `sim.init_people()` in v3 should be made to the exported arrays instead.
- Custom layer classes that subclass `cv.Layer` as a dict (`self['p1'] = ...`) need rewriting: `cv.Layer` is now an `ss.Network`, with edges in `layer.edges.p1` etc.

### Saved sims

- v3 sims saved without people load with `cv.load()`, and their results, parameters and dates can be read; they can't be rerun or plotted with `sim.plot()`. To rerun, recreate the sim from its parameters.
- v3 sims saved with people (`keep_people=True`) can't be loaded. Rerun them, or load them in v3 and export what's needed.

### Results

- By-variant results are `(npts, n_variants)` rather than `(n_variants, npts)`: `[1,:]` → `[:,1]`, or use the variant name (`.delta`).
- `sim.results` is an `ss.Results`, not a dict. Reading with `sim.results['cum_infections']` works; anything that needs a real dict should use `sc.objdict(sim.results)`.
- `sim.summary` keys work with or without the `covid_` prefix.

### Other

- `sim.pars` passed around as an object (flagged by the script): it only has the Starsim sim parameters. Pass the sim and read `sim['key']`, or use `sim.diseases.covid.pars`.
- `cv.Sim(version=...)` for versions before 2.1.0: see the rule in `docs/migrate3to4.md` for reproducing the old duration distributions.
- `reset_seed` and re-seeding during a run have no effect, since each distribution has its own random number stream. Code that did this to compare scenarios with the same random numbers no longer needs to: v4 does that by default.

## What not to do

- Don't change code the script didn't flag and that runs correctly.
- Don't replace v3 names with Starsim ones (`pop_size` → `n_agents`, `apply` → `step`) as part of a migration.
- Don't add type annotations, reformat, or tidy unrelated code.
- Don't conclude the results match (or don't) from one seed: an epidemic growing exponentially amplifies small differences, so compare means and spreads over several seeds.
