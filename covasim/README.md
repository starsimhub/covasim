# Parameters

This file describes each of the input parameters in Covasim. Note: the overall infection rate can be explored using `sim.results['doubling_time']` and `sim.results['r_eff']` (a higher infection rate means lower doubling times and higher *R_eff*), as well as by simply looking at the epidemic curves.

## Population parameters

- `pop_size` = Number of agents, i.e., people susceptible to SARS-CoV-2
- `pop_infected` = Number of initial infections
- `pop_type` = What type of population data to use -- 'random' (fastest), 'synthpops' (best), 'hybrid' (compromise), 'rank' (contacts mostly with people who live nearby; see `cv.RankNet`)
- `clustered` = For `hybrid`: whether schools and workplaces are random networks, as in v3 (False, the default), or clusters whose members all know each other (True)
- `location` = What location to load data from -- default Seattle

## Geography parameters

These say where people live (`people.x` and `people.y`, in degrees of longitude and latitude), and which region (`people.region`) and sewershed (`people.sewershed`) they are in; see `geography.py`.

- `boundary_shapefile` = Optional path to a boundary file (GeoJSON/shapefile with `region_code`, `region_name`, and optionally `population`), used to assign regions and locations; None means a single region with no locations
- `sewershed_file` = Optional path to a sewershed polygon file (GeoJSON/shapefile with a `sid` property); None means one whole-population catchment
- `people_coords` = Optional array of shape (pop_size, 2) with the longitude and latitude of each person
- `density_raster` = Path to a population-density GeoTIFF (people per pixel), used to place people when `pop_type='rank'` (inside `boundary_shapefile`, if given); None uses a 10 x 10 km test square
- `density_bbox` = Optional (lon_min, lat_min, lon_max, lat_max) window of that raster to read

## Simulation parameters

- `start_day` = Start day of the simulation
- `end_day` = End day of the simulation
- `n_days` = Number of days to run, if end_day isn't specified
- `rand_seed` = Random seed, if None, don't reset
- `verbose` = Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (more), 2 (everything)

## Rescaling parameters

- `pop_scale` = Factor by which to scale the population -- e.g. 1000 with pop_size = 10e3 means a population of 10m
- `scaled_pop` = The total scaled population, i.e. the number of agents times the scale factor; alternative to pop_scale
- `rescale` = Enable dynamic rescaling of the population
- `rescale_threshold` = Fraction susceptible population that will trigger rescaling if rescaling
- `rescale_factor` = Factor by which we rescale the population

## Basic disease transmission

- `beta` = Beta per symptomatic contact; absolute
- `n_imports` = Average daily number of imported cases (actual number is drawn from Poisson distribution)
- `beta_dist` = Distribution to draw individual level transmissibility; see <https://wellcomeopenresearch.org/articles/5-67>
- `viral_dist` = The time varying viral load (transmissibility); estimated from Lescure 2020, Lancet, <https://doi.org/10.1016/S1473-3099(20)30200-0>
- `asymp_factor` = Multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility: <https://www.sciencedirect.com/science/article/pii/S1201971220302502>
- `shedding` = Viral shedding into wastewater by day since becoming infectious, drawn for each person from [shedding_hub](https://github.com/shedding-hub/shedding-hub) (the arguments of its `select()`, and the `dispersion`); None to not calculate it
- `evo_pars` = Sequence evolution and variant fitness (disabled by default): `enable`, the genome length `L`, the `reference` sequence (or a FASTA file), the molecular clock rate `mol_clock_rate`, the substitution model `sub_model`, and the fitness model (`fitness_model`, `fitness_data_path`, `fitness_scale`); see `sequence_evolution.py` and `fitness.py`

## Network parameters

- `contacts` = The number of contacts per layer
- `dynam_layer` = Which layers are dynamic
- `beta_layer` = Transmissibility per layer

## Multi-strain parameters

- `n_imports` = Average daily number of imported cases (actual number is drawn from Poisson distribution)
- `n_strains` = The number of strains circulating in the population

## Immunity parameters

- `use_waning` = Whether to use dynamically calculated immunity
- `nab_init` = Parameters for the distribution of the initial level of log2(nab) following natural infection, taken from fig1b of <https://doi.org/10.1101/2021.03.09.21252641>
- `nab_decay` = Parameters describing the kinetics of decay of nabs over time, taken from fig3b of <https://doi.org/10.1101/2021.03.09.21252641>
- `nab_kin` = Constructed during sim initialization using the nab_decay parameters
- `nab_boost` = Multiplicative factor applied to a person's nab levels if they get reinfected. \# TODO: add source
- `nab_eff` = Parameters to map nabs to efficacy
- `rel_imm_symp` = Relative immunity from natural infection varies by symptoms
- `immunity` = Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py

## Strain-specific parameters

- `rel_beta` = Relative transmissibility varies by strain
- `rel_imm_strain` = Relative own-immunity varies by strain

## Time for disease progression

- `exp2inf` = Duration from exposed to infectious; see Lauer et al., <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/>, subtracting inf2sym duration
- `inf2sym` = Duration from infectious to symptomatic; see Linton et al., <https://doi.org/10.3390/jcm9020538>
- `sym2sev` = Duration from symptomatic to severe symptoms; see Linton et al., <https://doi.org/10.3390/jcm9020538Duration> from severe symptoms to requiring ICU; see Wang et al., <https://jamanetwork.com/journals/jama/fullarticle/2761044Duration> from severe symptoms to requiring ICU

## Time for disease recovery

- `asym2rec` = Duration for asymptomatic people to recover; see Wölfel et al., <https://www.nature.com/articles/s41586-020-2196-x>
- `mild2rec` = Duration for people with mild symptoms to recover; see Wölfel et al., <https://www.nature.com/articles/s41586-020-2196-x>
- `sev2rec` = Duration for people with severe symptoms to recover, 22.6 days total; see Verity et al., <https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf>
- `crit2rec` = Duration for people with critical symptoms to recover, 22.6 days total; see Verity et al., <https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf>
- `crit2die` = Duration from critical symptoms to death, 17.8 days total; see Verity et al., <https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf>

## Severity parameters

- `rel_symp_prob` = Scale factor for proportion of symptomatic cases
- `rel_severe_prob` = Scale factor for proportion of symptomatic cases that become severe
- `rel_crit_prob` = Scale factor for proportion of severe cases that become critical
- `rel_death_prob` = Scale factor for proportion of critical cases that result in death
- `prog_by_age` = Whether to set disease progression based on the person's age
- `prognoses` = The actual arrays of prognoses by age; this is populated later

## Efficacy of protection measures

- `iso_factor` = Multiply beta by this factor for diganosed cases to represent isolation; set below
- `quar_factor` = Quarantine multiplier on transmissibility and susceptibility; set below
- `quar_period` = Number of days to quarantine for; assumption based on standard policies

## Events and interventions

- `interventions` = The interventions present in this simulation; populated by the user
- `analyzers` = Custom analysis functions; populated by the user
- `timelimit` = Time limit for the simulation (seconds)
- `stopping_func` = A function to call to stop the sim partway through

## Health system parameters

- `n_beds_hosp` The number of hospital (adult acute care) beds available for severely ill patients (default is no constraint)
- `n_beds_icu` The number of ICU beds available for critically ill patients (default is no constraint)
- `no_hosp_factor` Multiplier for how much more likely severely ill people are to become critical if no hospital beds are available
- `no_icu_factor` Multiplier for how much more likely critically ill people are to die if no ICU beds are available
