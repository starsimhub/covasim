# Migrating from Covasim v3 to v4

Covasim v4 is built on the [Starsim](https://starsim.org) framework: `cv.Sim` is a subclass of `ss.Sim`, the population is an `ss.People`, contact layers are `ss.Network`s, and the COVID disease logic is in a single `cv.COVID` module (an `ss.Infection`). The v3 API is preserved, so most v3 code runs on v4 without changes. This guide lists what does need to change, and how to make those changes.

## What stays the same

- **Creating and running a sim**: `cv.Sim(pars)`, `cv.Sim(pop_size=20e3, ...)`, `sim.initialize()`, `sim.run()`, `sim.plot()`, `sim.save()` and `cv.load()`.
- **Parameters**: the v3 parameter names (`pop_size`, `pop_infected`, `pop_type`, `n_days`, `start_day`, `beta`, `pop_scale`, `rescale`, `use_waning`, `location` etc.), read and set with `sim['key']`.
- **Results**: `sim.results['cum_infections']`, `sim.summary['cum_infections']`, and per-agent states such as `sim.people.exposed`.
- **Interventions**: `cv.test_prob`, `cv.test_num`, `cv.contact_tracing`, `cv.vaccinate_prob`, `cv.vaccinate_num`, `cv.vaccinate`, `cv.simple_vaccine`, `cv.change_beta`, `cv.clip_edges`, `cv.dynamic_pars` and `cv.sequence`, as well as custom interventions with an `apply(self, sim)` method.
- **Analyzers**: `cv.snapshot`, `cv.age_histogram`, `cv.daily_age_stats`, `cv.nab_histogram` and `cv.TransTree`, as well as custom analyzers.
- **Variants and immunity**: `cv.variant`, waning immunity and cross-immunity (`use_waning`, which is on by default as in v3).
- **Multiple runs and fitting**: `cv.MultiSim`, `cv.Scenarios`, `cv.parallel`, `cv.Fit` and `cv.Calibration`.

What is supported for backwards compatibility is in `covasim/compat.py`.

## How to migrate

The rules below are the changes that v3 code needs in order to run on v4, found by running the v3 test suite, tutorials and examples against v4. Rules marked *mechanical* are applied by the migration script; rules marked *judgment* can change results without raising an error, so need checking by hand.

```bash
covasim-migrate3to4 my_script.py           # Show the changes for one file
covasim-migrate3to4 my_folder --apply      # Change every .py file and notebook in a folder
```

The script also lists the lines that may need one of the other rules. The same functions are available in Python as `cv.migrate3to4`, e.g. `cv.migrate3to4.migrate('my_folder')`. For an AI assistant doing a migration, `docs/migrate3to4/SKILL.md` has instructions.

Results are not identical to v3 for the same `rand_seed`, since v4 uses Starsim's random number streams (one for each distribution) rather than a single global stream. To check that a migrated script gives the same answers, compare the means over several seeds rather than single runs.

## Removed

- `cv.ParsObj`, `cv.BaseSim`, `cv.BasePeople`, `cv.Person`: the v3 base classes are gone; `cv.Sim` and `cv.People` are now based on `ss.Sim` and `ss.People`. There is no v4 equivalent; code that subclasses these needs rewriting.
- `pop_type='synthpops'`: SynthPops is not supported; use `pop_type='hybrid'`.
- `popfile`, `cv.make_people()`, `cv.make_randpop()`, `cv.make_synthpop()`: v3 populations can't be loaded or created; let `cv.Sim` create the population. To reuse a saved v3 (e.g. SynthPops) population, export each agent's age and sex and each layer's `p1`, `p2` and `beta` arrays with v3; then in v4, create `people = cv.People(n)`, set `people.age.default` and `people.female.default` to functions that return the stored values (e.g. `def get_ages(n): return ages[:n]`), and pass `people=people, networks=[cv.Layer(name='h', edges=dict(p1=p1, p2=p2, beta=beta)), ...]` to `cv.Sim()`.
- `cv.migrate3to4()`: v3 sims can be loaded with `cv.load()` and their results, parameters (`sim['key']`), dates (`sim.day()`, `sim.datevec`), interventions and analyzers read, but they can't be migrated, rerun or plotted, and v3 sims saved with their people (or v3 `.ppl` files) can't be loaded.
- Parameters from Covasim versions before 2.1.0 (`cv.Sim(version=...)`). Versions before 2.1.0 differ from 2.1.0 only in their duration distributions, whose `par2` is the square root of the 2.1.0 value, so e.g. use `version='2.1.0'` with `dur` from `cv.make_pars(version='2.1.0')['dur']` with each `par2` replaced by its square root.

## Changed

- `sim.pars['key']` → `sim['key']` (*mechanical*). `sim.pars` now holds the Starsim sim parameters (e.g. `n_agents` rather than `pop_size`); the COVID parameters are in `sim.diseases.covid.pars`. `sim['key']` gets and sets the v3 parameters wherever they are stored. Unmigrated, setting a COVID parameter with `sim.pars['key'] = value` raises an error once the sim is initialized (`Key "key" not found`); before then it has no effect, without an error.
- Agents who have died are removed from the per-agent arrays (*judgment*). As in Starsim, `sim.people.age`, `sim.people.exposed`, `sim.people.date_exposed` etc., `len(sim.people)` and `sim.n` only include agents who are alive, so their length shrinks as agents die, and a position in an array is not a UID. The exceptions are `sim.people.dead` and `sim.people.date_dead`, which cover every agent ever created (so `sim.people.dead.sum()` is the number of deaths, and `cv.true(sim.people.dead)` is their UIDs). Indexing by UID works for any agent, alive or dead, so `people.age[cv.true(people.dead)]` needs no change. What needs migrating is code that assumes a fixed length or combines arrays elementwise with `dead`: use `.raw` for the values for every agent, e.g. `people.age[people.dead]` → `people.age.raw[people.dead]`, `np.arange(len(sim.people))` → `sim.people.indices()` (the UIDs of agents alive), `np.zeros(len(sim.people))` indexed by UID → `np.zeros(sim.people.n_uids)`. Use `.uids` or `cv.true()` to get UIDs rather than `sc.findinds()` or `np.nonzero()`, which return positions.
- `people.get()`, `people.person_keys()`, `people.dur_keys()`, `people.to_arr()`, `people.to_list()`, `people.from_list()`, `people._resize_arrays()`, `people.make_edgelist()`: v3 internals with no v4 equivalent. Use `people.keys()`, `people.to_df()`, and `people.person(uid)` instead.
- `cv.Result(...)` → `ss.Result(...)`: `cv.Result` is now Starsim's `ss.Result`, which has a different signature.
- `sim.validate_pars()`, `sim.init_*()` (e.g. `init_people`, `init_immunity`): v3 internals of initialization. Parameters are validated when the sim is created or initialized.
- Sims are saved with their people by default (`sim.save()`; use `keep_people=False` to remove them), so a loaded sim that has been run needs `sim.initialize(reset=True)` before it can be rerun.
- `sim.run(reset_seed=True)` and `sim.set_seed()` part-way through a run have no effect on the results: v4 uses a separate random number stream for each distribution (common random numbers), so results only depend on `rand_seed`.
- `sim.people.t = sim.t` (to continue running a sim after re-initializing it): v3 internal; use `sim.initialize(reset=True)` instead.
- `daily_age.results[date]` → `daily_age.age_results[date]` (or `daily_age.to_df()`): Starsim reserves `results` for time series.
- `snapshot` values are `cv.PeopleSnapshot` objects (full-length arrays indexed by UID, supporting indexing, `true()`, `false()`, `defined()` and `count()`) rather than copies of `People`; other `People` methods (e.g. contacts) aren't available on them.
- `sim.results['gen_time']` → `sim.gen_time`, and `sim.results.transtree`/`agehist` → `sim.transtree`/`sim.agehist` (with `output=False`) (*mechanical*): Starsim results can only hold time series.
- `TransTree.detailed` columns for `date_known_contact` are NaN, since v4 doesn't record that date.
- `msim.combine()` doesn't merge people; the combined sim keeps the first sim's people.
- `Scenarios(sim=<an already-run sim>)` can't be rerun; pass an unrun sim.
- Scenarios can't change parameters that determine how the sim is built (e.g. `use_waning`, `pop_type`); set these on the base sim.
- `cv.Calibration(..., n_workers=...)` defaults to the number of CPUs, and the calibration database is stored in a temporary folder unless `name` or `db_name` is given.
- Plot titles use the v4 result labels (e.g. "New infections" rather than "Number of new infections"), so `log_scale=[title]` lists may need updating.
- `sim.t` → `sim.ti` (*mechanical*) in custom interventions and analyzers, e.g. `if sim.t == 10:` → `if sim.ti == 10:`, `self.results[key][sim.t]` → `...[sim.ti]`, and `sim.t % 7` → `sim.ti % 7`. In v4, `sim.t` is the Starsim timeline (`ss.Timeline`) rather than the integer day, so comparing it with a number or using it as an index raises an error.
- Custom `finalize(self)` methods in the Starsim form (without the `sim` argument) must call `super().finalize()`; v3-style `finalize(self, sim)` methods don't need to.
- In custom interventions and analyzers, attributes Starsim reserves (`t`, `pars`, `sim`, `dists`, `results`) can't be set (e.g. `self.t = []` → `self.tvec = []`). Module names must be unique across interventions, analyzers, and networks: a module with a clashing default name (e.g. a class `A`, named `a` like the random network) is renamed with a warning, but explicitly set names that clash raise an error.
- Transmission is 1–2% lower than in v3 for the same parameters (*judgment*), because of a correction to the viral load. `viral_dist` gives a high viral load for the first `frac_time` (30%) of the infectious period, up to `high_cap` (4) days. In v3, a rounding error (comparing a double-precision fraction with a single-precision threshold) made this a day longer for people infectious for exactly 10 days (4 days rather than 3) and for most people infectious for 14 days or more (5 rather than 4); v4 uses the intended durations. The difference compounds while an epidemic is growing: e.g. about 10% fewer infections after four months in one calibrated model. Models calibrated with v3 should be recalibrated, usually by increasing `beta` by 1–2%; there is no option to reproduce the v3 behavior.
- Results by variant: `new_symptomatic_by_variant` and `new_severe_by_variant` (and their cumulative versions) are counted on the day people become symptomatic or severe, as for the results that aren't by variant; v3 counted them on the day of infection. `prevalence_by_variant` is the number infected with each variant divided by the number alive, so it sums to `prevalence`; v3 used the number of new infections. No changes to code are needed, but the values differ from v3.
- `cv.historical_wave()` doesn't include the seed infections in the wave. v3 did, then infected them again on day 0, which counted them twice in `new_infections` and `cum_infections` on day 0 (so v3's values are higher by `pop_infected`), and listed them twice in the transmission log.
- Results by variant have time as the first axis, i.e. shape `(npts, n_variants)` rather than `(n_variants, npts)`: e.g. `sim.results['variant']['new_infections_by_variant'][1,:]` → `[:,1]` (or `.values.T`), or use the variant name, e.g. `...['new_infections_by_variant'].delta`.

## Where things are in v4

None of this is needed to run v3 code, but it helps when reading v4 code or writing new code:

- The COVID parameters are in `sim.diseases.covid.pars`, and the sim parameters (with the Starsim names, e.g. `n_agents`) are in `sim.pars`; `sim['key']` reads and sets both.
- The COVID results are in `sim.diseases.covid.results`, and are also available as `sim.results[key]`. Results by variant are in `sim.results['variant']`. The keys of `sim.summary` have the module name as a prefix (e.g. `covid_cum_infections`), but can also be used without it.
- `sim.init()` is the Starsim name for `sim.initialize()`; both work.

See the [Starsim documentation](https://docs.starsim.org) for more on the framework.
