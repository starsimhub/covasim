# Migrating from Covasim v3 to v4

Covasim v4.0 is a substantial internal change: the model is now built on the
[Starsim](https://starsim.org) framework. `cv.Sim` is a subclass of `ss.Sim`, the agent population
is an `ss.People`, contact layers are `ss.Network`s, and the COVID disease logic lives in a single
`cv.COVID(ss.Infection)` module. Despite the new foundation, **the public Covasim API is preserved**:
the great majority of v3 scripts run unchanged, or with only the small adjustments documented below.

This guide covers what stays the same, what changed, how to convert a v3 script, and the features
that are not yet ported.

---

## 1. What stays the same

These work in v4 exactly as in v3:

- **Building a sim.** Both the dict form and the keyword form are supported, and an explicit keyword
  overrides the same key in the dict:

  ```python
  import covasim as cv

  # dict form (canonical v3 usage)
  pars = dict(pop_size=20e3, pop_infected=100, pop_type='hybrid', n_days=120, use_waning=True)
  sim = cv.Sim(pars)

  # keyword form
  sim = cv.Sim(pop_size=20e3, pop_infected=100, pop_type='hybrid', n_days=120, use_waning=True)

  sim.run()
  ```

- **Interventions** — `cv.test_prob`, `cv.test_num`, `cv.contact_tracing`, `cv.vaccinate_prob`,
  `cv.vaccinate_num`, `cv.vaccinate`, `cv.simple_vaccine`, `cv.change_beta`, `cv.clip_edges`,
  `cv.dynamic_pars`, and `cv.sequence` all keep their v3 signatures.
- **Analyzers** — `cv.snapshot`, `cv.age_histogram`, `cv.daily_age_stats`, `cv.nab_histogram`, and
  `cv.TransTree`.
- **Multi-run and analysis tools** — `cv.MultiSim`, `cv.Scenarios`, `cv.parallel`, `cv.Fit`,
  `cv.Calibration`.
- **Variants and waning immunity** — `cv.variant`, co-circulating strains, neutralizing-antibody
  waning, and cross-immunity (gated by `use_waning`, exactly as in v3).
- **Save / load** — `sim.save()`, `cv.Sim.load()`, `cv.save()`, `cv.load()`, and `sc.load()`.
- **Plotting** — `sim.plot()` (a headline multi-panel view), `fit.plot()`, `transtree.plot()`.

## 2. What changed

### 2.1 Results are namespaced, with a top-level bridge

The canonical home for results is the disease module:

```python
sim.diseases.covid.results['cum_infections']   # canonical
sim.results['cum_infections']                  # bridged alias (works too)
```

For backwards compatibility, every top-level COVID result is also referenced at `sim.results[key]`,
so v3-style `sim.results['cum_deaths']` continues to resolve. The by-variant results live under
`sim.results['variant'][key]` (a 2D `time × variant` array).

`sim.summary` keys are **namespaced** by module (`covid_cum_deaths`, `h_n_edges`, …) rather than the
bare v3 names. If you parsed `sim.summary` by key, update the keys accordingly.

### 2.2 Random numbers and exact reproducibility

Starsim uses **per-distribution common random numbers (CRN)** rather than Covasim v3's single global
NumPy/Numba stream. As a result, **v4 results are not bit-for-bit identical to v3** for the same seed.
This is expected. Equivalence is validated statistically: the migration ships multi-seed z-score
*parity gates* that confirm v4 and v3.1.8 agree within sampling noise on the headline metrics.

### 2.3 `sim.init()` (with `sim.initialize()` retained)

Starsim's initializer is `sim.init()`. The v3 name `sim.initialize()` is retained as an alias, so
existing calls keep working. (`sim.run()` still initializes automatically if needed.)

### 2.4 Inputs are deep-copied

As in v3, `cv.Sim` deep-copies the interventions / analyzers / diseases you pass in. The *live*
objects after a run are on the sim:

```python
snap = cv.snapshot(days=[30])
sim  = cv.Sim(pop_size=10e3, analyzers=snap)
sim.run()
snap_live = sim.analyzers['snapshot']   # NOT the `snap` you constructed
```

### 2.5 Parameter access

Parameters are stored on `sim.pars` (sim-level), `sim.diseases.covid.pars` (disease), and the network
objects. Broad v3-style `sim['beta']` item access is **not** fully supported in v4.0 — read from the
specific object instead (e.g. `sim.diseases.covid.pars.beta`). Use `sim.export_pars()` to snapshot the
configuration to JSON.

## 3. Parameter remapping

| v3 | v4 location | Notes |
|---|---|---|
| `pop_size` | `cv.Sim(pop_size=...)` | Becomes `n_agents` internally |
| `pop_infected` | `cv.Sim(pop_infected=...)` | Exact seed count at t=0 |
| `pop_type` | `cv.Sim(pop_type=...)` | `'random'` or `'hybrid'` (`'synthpops'` not yet ported) |
| `n_days` | `cv.Sim(n_days=...)` | Mapped to `dur=ss.days(n_days)`, `dt=ss.days(1)` |
| `start_day` | `cv.Sim(start_day=...)` | Mapped to `start=ss.date(...)` |
| `beta` | `cv.Sim(beta=...)` | Per-layer weights carried on the disease's `beta` dict |
| `use_waning` | `cv.Sim(use_waning=...)` | Gates the NAb/waning + cross-immunity engine |
| `pop_scale` / `total_pop` | `cv.Sim(pop_scale=...)` / `total_pop=...` | Absolute scaling; pass at most one |
| natural-history params (`rel_severe_prob`, durations, …) | `sim.diseases.covid.pars` / `cv.dynamic_pars` | Set per-run via interventions, as in v3 |

## 4. Converting a script

**v3:**
```python
import covasim as cv

pars = dict(pop_size=50e3, pop_infected=100, n_days=90, pop_type='hybrid')
sim = cv.Sim(pars, interventions=cv.test_prob(symp_prob=0.1))
sim.initialize()
sim.run()
print(sim.summary['cum_infections'])
sim.plot()
```

**v4:**
```python
import covasim as cv

pars = dict(pop_size=50e3, pop_infected=100, n_days=90, pop_type='hybrid')
sim = cv.Sim(pars, interventions=cv.test_prob(symp_prob=0.1))
sim.run()                                       # init() runs automatically
print(sim.summary['covid_cum_infections'])      # namespaced key
sim.plot()
```

The only required changes are the **namespaced summary key** and (optionally) dropping the explicit
`initialize()` call. Everything else is identical.

## 5. Not yet ported

- **`pop_type='synthpops'`** (the SynthPops population backend, including the LTCF layer). Use
  `'hybrid'` or `'random'` for now.
- **`bin/covasim`** — the command-line wrapper is retired in v4.0; use the Python API directly.
- **Dynamic rescaling** (`rescale`, `make_naive`/`make_nonnaive`). v4 uses absolute agent counts;
  static `pop_scale` / `total_pop` are supported.
- **Full TransTree graph plotting** — `cv.TransTree` reconstructs the tree and computes offspring/R0,
  and `transtree.plot()` shows the offspring + timing distributions, but the full NetworkX graph view
  is not yet ported.
- **Loading pre-v4 pickles** — v3 `.sim`/`.scens` files will generally not unpickle under the new
  object model. Re-run from parameters, or keep v3 installed to read old files.

## 6. Getting help

See the [Starsim documentation](https://starsim.org) for the underlying framework, the Covasim
tutorials under `docs/`, and the `migration_plan/` folder in the repository for the full
milestone-by-milestone record of the port.
