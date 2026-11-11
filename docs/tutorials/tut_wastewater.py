"""
Covasim for wastewater: step by step tutorial (script version of tut_wastewater.ipynb).

Runs from any directory; paths are resolved relative to this file. Cells are marked with
"# %%" so editors such as VS Code and Spyder can run them one at a time.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) # So the relative data paths below resolve

plot_dir = 'tut_outputs' # All plots are saved here as PDFs instead of being shown
os.makedirs(plot_dir, exist_ok=True)

# %% [markdown]
# # Covasim for wastewater step by step tutorial
# 1. To run the following code, first make sure covasim and its dependencies are installed.
# Please follow the instructions here https://github.com/andersen-lab/covasim4wastewater to install.
# (Full covasim tutorial available at : https://docs.covasim.org/tutorials/)
# 2. To start, we run a basic simulation: 20,000 people with 20 initial infections for 100 days. The contact network is "rank", meaning people mostly contact others who live near them: the chance of a contact falls with how many people live closer (the nearest person has rank 1). People are placed using a population-density raster, so they cluster where people actually live. There are no imported cases.
# 3. People are placed on a map using the San Diego files in `san-diego-data/` (built from ZIP-code polygons and a ZIP-to-treatment-plant table by `make_sewershed_map.py`):
#    * `sandiego_zip_boundaries.geojson` — one region per ZIP code, with `region_code` and `region_name`. People are assigned to regions and given home coordinates inside them. An optional `population` column weights the regions by population; this file has none, so regions are weighted by area (and covasim warns about it).
#    * `sandiego_sewersheds.geojson` — one catchment polygon per treatment plant, with a `sid` property (the plant name). Each person is assigned the sewershed containing their home; people outside every sewershed get -1.
#    * `san_diego_county_ppp_2020.tif` — WorldPop 2020 people per ~100 m pixel (EPSG:4326), passed as `density_raster`. Agents are placed in proportion to this raster, but only inside the region polygons. Make it with `clip_worldpop.py`.
#
#    Both are optional (`boundary_shapefile=None`, `sewershed_file=None` by default, giving one region and one catchment). Supply your own files in the same format to model a different place.
#
# > **Note:** Run this notebook from the `docs/tutorials/` directory so the relative paths below resolve.

# %%
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt

sim = cv.Sim(
    pop_size=20_000,
    pop_type='rank',
    density_raster='san-diego-data/san_diego_county_ppp_2020.tif',
    pop_infected=20,
    n_days=100,
    n_imports=0,
    boundary_shapefile='san-diego-data/sandiego_zip_boundaries.geojson',
    sewershed_file='san-diego-data/sandiego_sewersheds.geojson',
)
sim.run()

# %% [markdown]
# ## Wastewater viral loads
#
# To estimate how much each infected individual is shedding to wastewater,
#  we use the shedding hub https://shedding-hub.github.io, "woelfel2020virological" dataset.
#
# For each individual, the full `pop_size × n_days` shedding matrix is stored in `sim.results['viral_shedding_hist']`
# (memory grows with population × days, so keep this in mind for very large runs).
# The following plot shows how much each of the first 100 agents is shedding (cell color) over the simulation (x axis time, y axis agent).

# %%
viral_shedding = sim.results['viral_shedding_hist']
im = plt.imshow(
    viral_shedding[:100],  # first 100 agents only
    aspect='auto',   # prevents squashing
    origin='lower',  # agent 0 at bottom
    cmap='viridis',
)
plt.colorbar(im, label='Viral shedding')
plt.xlabel('time')
plt.ylabel('agent')
plt.title('Viral shedding by agent and time (first 100 agents)')
plt.tight_layout()
plt.savefig(f'{plot_dir}/01_shedding_heatmap.pdf')
plt.close()

# %% [markdown]
# We then look at the overall shedding (total for all agents) and the shedding in each sewershed at each time point. This is the signal a wastewater treatment plant would see for its catchment.

# %%
n_agents = sim.people.n_uids # Agents ever created; viral_shedding_hist has a row for each, including those who died
sewersheds = sim.people.sewershed.raw[:n_agents] # .raw: every agent, not only those alive at the end
plt.figure(figsize=(12, 6))
for sid in np.unique(sewersheds):
    mask = sewersheds == sid
    name = sim.sewershed_labels.get(sid, 'Outside all sewersheds') # IDs of sewersheds in the file -> names; 0 and -1 are not in the file
    plt.plot(viral_shedding[mask].sum(axis=0), label=f'{name} ({mask.sum()} people)')
plt.plot(viral_shedding.sum(axis=0), linewidth=2, color='red', label='Total')
plt.xlabel('Day')
plt.ylabel('Total viral shedding')
plt.title('Total viral shedding by sewershed')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/02_shedding_by_sewershed.pdf')
plt.close()

# %% [markdown]
# ## Genomic simulation
# In this section, we use Covasim to simulate how different SARS-CoV-2 variants—specifically **wild-type, AY.4, BA.2, and XBB.1.5**—spread over time. We track these variants using consensus FASTA files and plot their infectious counts on both absolute and proportional scales.
#
# ### How it works
#
# * **Seeding variants:** We introduce each new variant using a `founding_fasta` file, which defines its specific mutation profile when it enters the population.
# * **Tracking infections:** To track variant proportions over time, we pull daily counts directly from `sim.results['variant']['n_infectious_by_variant']`—Covasim's built-in tracker for daily infectious cases per variant.
# * **Wastewater sampling:** The `WastewaterSampler` tool isn't needed to plot these infection numbers; it's used only to generate the haplotype-resolved data required for Bygul (covered in the next section).
#
# > **Note:** Make sure to run this notebook from the `docs/tutorials/` directory so Covasim can find the FASTA files located in `sars-2-data/variants/`.

# %%


# Paths are relative to docs/tutorials/ — run the notebook from that directory.
ref_fasta         = 'sars-2-data/NC_045512_Hu-1.fasta'
ay4_fasta         = 'sars-2-data/variants/AY.4_consensus.fasta'
ba2_fasta         = 'sars-2-data/variants/BA.2.9_consensus.fasta'
xbb15_fasta       = 'sars-2-data/variants/XBB.1.5.18_consensus.fasta'
artic_primers     = 'sars-2-data/ARTIC_V4-1.bed'

# Variants — rel_beta is NOT hard-coded.  evo_pars (below) derives the
# transmission advantage from each variant's founding mutations by looking up
# every SNP in the Bloom lab nucleotide fitness landscape and computing
# rel_beta = exp(Σ log_fitness_i × fitness_scale).
ay4 = cv.variant(
    label          = 'AY.4',
    days           = 30, # introduce AY.4 on day 30
    n_imports      = 50,
    founding_fasta = ay4_fasta,
)
ba2 = cv.variant(
    label          = 'BA.2',
    days           = 60, # introduce BA.2 on day 60
    n_imports      = 50,
    founding_fasta = ba2_fasta,
)
xbb15 = cv.variant(
    label          = 'XBB.1.5',
    days           = 100, # introduce XBB.1.5 on day 100
    n_imports      = 50,
    founding_fasta = xbb15_fasta, # the variant's sequence when first introduced
)

# WastewaterSampler captures each unique evolved haplotype so simulate_sample()
# can feed the mixture to Bygul for amplicon-read synthesis.
# Sampled after XBB.1.5's introduction (day 100) while all four variants are
# likely still co-circulating.
_haplotype_days = list(range(110, 151, 5))   # days 110, 115, 120 … 150
ww = cv.WastewaterSampler(
    days  = _haplotype_days,
    label = 'WastewaterHaplotype',
)

sim = cv.Sim(
    dict(
        pop_size     = 20_000,
        pop_infected = 20, #initial infections
        start_day    = '2022-10-01',
        end_day      = '2023-04-01',
        evo_pars = dict( #evolutionary parameters for the simulation
            enable            = True,
            reference         = ref_fasta,
            mol_clock_rate    = 1e-6,
            fitness_model     = 'bloom_nt',           # Bloom lab nt_fitness.csv https://github.com/jbloomlab/SARS2-mut-fitness
                                                      # (the default fitness_data_path is the copy in the repository's data folder)
            fitness_scale     = 0.05,                 # multiplies log-fitness sum before exp()
        ),
    ),
    variants  = [ay4, ba2, xbb15],
    analyzers = [ww],
    verbose   = 0,
)
sim.run()

# %% [markdown]
# Read more about sequence evolution : https://github.com/andersen-lab/covasim4wastewater/blob/main/docs/tutorials/tut_sequence_evol.ipynb
# The following code calculates the relative beta for each variant and shows the SNP counts compared to the founding sequence.

# %%
# Show the rel_beta multiplier that was derived from each variant's founding mutations.
# This makes the FASTA → fitness → transmissibility pathway explicit.
tracker       = sim.people.sequence_tracker
fitness_model = tracker.fitness_model

print(f'{"variant":12s}  {"founding SNPs":>13s}  {"rel_beta (×wt)":>15s}')
print('-' * 46)
# Wild-type carries no founding mutations → reference fitness = 1.0
for label, muts in [('wild', frozenset())] + list(tracker.variant_founding_mutations.items()):
    mult = fitness_model.compute_fitness(muts)
    print(f'{label:12s}  {len(muts):>13d}  {mult:>15.4f}')

# %% [markdown]
# The following code generates plots that show variant specific case counts at each time point. We can also see the prportions of infected people by each variant. The plot shows how each specific variant takes over based on the day they are introduced. e.g. 'AY.4': 30, 'BA.2': 60, 'XBB.1.5': 100 (dashed lines)

# %%
variant_names = list(sim['variant_map'].values())   # e.g. ['wild', 'AY.4', 'BA.2', 'XBB.1.5']
n_steps = sim.npts

# Per-variant infectious counts from the standard Covasim result.
# Shape: (n_variants, n_steps), i.e. the transpose of the result, which has one column per variant.
n_inf = sim.results['variant']['n_infectious_by_variant'].values.T
total = n_inf.sum(axis=0)
total[total == 0] = 1  # avoid division by zero on days with no infections

loads = {v: n_inf[i] for i, v in enumerate(variant_names)}
props = {v: n_inf[i] / total for i, v in enumerate(variant_names)}

# Color palette (extend automatically for any extra variants)
palette = {'wild': '#4C72B0', 'AY.4': '#DD8452', 'BA.2': '#C44E52', 'XBB.1.5': '#55A868'}
_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, v in enumerate(variant_names):
    palette.setdefault(v, _defaults[i % len(_defaults)])

tvec       = np.arange(n_steps)
intro_days = {'AY.4': 30, 'BA.2': 60, 'XBB.1.5': 100}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# -- Absolute infectious counts -----------------------------------------------
for vname in variant_names:
    ax1.plot(tvec, loads[vname], label=vname, color=palette[vname], linewidth=1.8)
for vname, day in intro_days.items():
    ax1.axvline(day, color=palette[vname], linestyle=':', linewidth=1.2, alpha=0.7)
ax1.set_ylabel('Infectious agents')
ax1.set_title('Infectious count by variant — absolute')
ax1.legend(framealpha=0.9)

# -- Scaled to 1 (stacked area) -----------------------------------------------
stacked = np.array([props[v] for v in variant_names])
ax2.stackplot(tvec, stacked,
              labels=variant_names,
              colors=[palette[v] for v in variant_names],
              alpha=0.85)
for vname, day in intro_days.items():
    ax2.axvline(day, color='white', linestyle=':', linewidth=1.5, alpha=0.9)
ax2.set_xlabel('Simulation day')
ax2.set_ylabel('Proportion of infectious agents')
ax2.set_title('Variant proportion over time — scaled to 1')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{plot_dir}/03_variants_counts_and_proportions.pdf')
plt.close()

# %% [markdown]
# ## Amplicon sequencing snapshot with Bygul
#
# `WastewaterSampler` records every unique evolved haplotype present in the
# infectious pool on each sampling day.  At a time-point where wild-type, AY.4,
# BA.2, and XBB.1.5 all co-circulate the mixture mirrors what a real wastewater
# sequencing lab would see.
#
# `simulate_sample()` writes each haplotype to a temporary FASTA file and calls
# [Bygul](https://github.com/andersen-lab/Bygul)'s `simulate_proportions` CLI,
# which tiles ARTIC V4.1 amplicons across each genome and produces synthetic
# paired-end reads at the observed mixture proportions — giving a realistic
# amplicon-sequencing dataset that reflects the simulated epidemiology.
#
# > **Requires** the `bygul` package: `pip install bygul`  (installing through conda is recommended conda install bioconda::bygul)
# > BED file and reference FASTA are read from `sars-2-data/` via the paths
# > set in the earlier cell (`artic_primers`, `ref_fasta`).

# %%
import glob, os, dataclasses

# From the candidate haplotype days, pick the first where all four variants
# each hold >1% of signal.  Fall back to day 120 if none qualifies.
ww = sim.get_analyzer('WastewaterHaplotype')
mixed_day = next(
    (t for t in sorted(ww.samples)
     if ww.samples[t] is not None
     and all(props[v][t] > 0.01 for v in ['wild', 'AY.4', 'BA.2', 'XBB.1.5'])),
    120,
)

print(f'Sampling day {mixed_day}  ({sim.date(mixed_day)})')
print('  Variant-level proportions at that day:')
for v in ['wild', 'AY.4', 'BA.2', 'XBB.1.5']:
    print(f'    {v:10s}  {props[v][mixed_day]:.1%}')

sample = ww.samples.get(mixed_day)
if sample is None:
    print('\nNo infectious agents were present on that day.')
else:
    print(f'\n  Unique haplotypes in pool : {sample.n_genotypes}')

    # Rank haplotypes by proportion and keep the top 5.
    # Renormalize so the subset proportions still sum to 1.0 — Bygul requires this.
    order     = sorted(range(sample.n_genotypes),
                       key=lambda i: sample.proportions[i], reverse=True)[:5]
    top_seqs  = [sample.mutation_sets[i]   for i in order]
    top_loads = [sample.raw_loads[i]   for i in order]
    top_props = [sample.proportions[i] for i in order]
    top_total = sum(top_props)
    top_props = [p / top_total for p in top_props]
    top_props[-1] = 1.0 - sum(top_props[:-1])   # absorb floating-point residual

    print(f'  Top-5 proportions (renormalized): {[round(p, 4) for p in top_props]}')
    print(f'  These cover {top_total:.1%} of the full pool by viral shedding')

    # Replace the stored sample with the trimmed version so simulate_sample() sees it
    ww.samples[mixed_day] = dataclasses.replace(
        sample,
        mutation_sets = top_seqs,
        proportions   = top_props,
        raw_loads       = top_loads,
        n_genotypes     = len(top_seqs),
    )

# artic_primers and ref_fasta were set in the earlier cell
outdir = './amplicon_reads/'

try:
    ww.simulate_sample(
        day             = mixed_day,
        primers         = artic_primers,
        reference       = ref_fasta,
        outdir          = outdir,
        readcnt         = 300,
        redo            = True,
        simulation_mode = 'amplicon',
        seed            = 42,
    )
    fastq_files = (
        glob.glob(os.path.join(outdir, '**/*.fastq*'), recursive=True) +
        glob.glob(os.path.join(outdir, '*.fastq*'))
    )
    print(f'\nBygul wrote {len(fastq_files)} FASTQ file(s) to {outdir!r}:')
    for f in sorted(set(fastq_files))[:12]:
        print(f'  {os.path.relpath(f)}  ({os.path.getsize(f)/1024:.1f} kB)')
except ImportError as exc:
    print('\n[Bygul not installed — skipping read simulation]')
    print(str(exc))
    print('Install: pip install bygul   (https://github.com/andersen-lab/Bygul)')

# %% [markdown]
# ## Clinical and wastewater data streams together
#
# The same simulation can produce both a clinical stream (sequences from diagnosed cases) and a wastewater stream (shedding-weighted haplotype mixtures per sewershed). Here we add testing, a `ClinicalSequencer`, and one `WastewaterSampler` per sewershed (`sid=`; `sid=None` samples everyone).
#
# > Covasim copies analyzers when the sim is created, so always read results back with `sim.get_analyzer(label)`, not from the original objects.

# %%
sample_days = list(range(40, 100, 10))
dual = cv.Sim(
    dict(
        pop_size     = 20_000,
        pop_infected = 20,
        n_days       = 100,
        pop_type     = 'rank',
        density_raster = 'san-diego-data/san_diego_county_ppp_2020.tif',
        evo_pars     = dict(enable=True, reference=ref_fasta, mol_clock_rate=1e-6,
                            fitness_model='bloom_nt', fitness_scale=0.05),
        boundary_shapefile = 'san-diego-data/sandiego_zip_boundaries.geojson',
        sewershed_file     = 'san-diego-data/sandiego_sewersheds.geojson',
    ),
    variants      = [ay4],
    interventions = [cv.test_prob(symp_prob=0.2, start_day=0)],
    analyzers     = [cv.ClinicalSequencer(days=sample_days, n_samples=50, label='clinical')]
                  + [cv.WastewaterSampler(days=sample_days, sid=s, label=f'ww_sewershed_{s}') for s in (1, 2, 3)],
    verbose       = 0,
)
dual.run()

clinical = dual.get_analyzer('clinical')
for day in sample_days:
    row = f'day {day}: clinical n={len(clinical.samples.get(day, [])):3d}'
    for s in (1, 2, 3):
        smp = dual.get_analyzer(f'ww_sewershed_{s}').samples.get(day)
        row += f' | sewershed {s}: ' + (f'{smp.n_genotypes} haplotypes' if smp else 'no shedders')
    print(row)

# %% [markdown]
# Both streams can be mapped side by side, each in its own geography. From the same simulation, the left panel shows cumulative diagnosed cases per 1,000 residents (clinical), grouped by the ZIP codes in `sandiego_zip_boundaries.geojson`. The right panel shows cumulative viral shedding per capita (wastewater), i.e. per resident, grouped by the sewersheds in `sandiego_sewersheds.geojson` (ZIP codes outside every sewershed are not counted). Both colour scales start at 0.

# %%
import geopandas as gpd

sewershed_gdf = gpd.read_file('san-diego-data/sandiego_sewersheds.geojson')
# people.sewershed holds integer IDs; sim.sewershed_labels gives {integer ID: original sid}
sewershed_gdf['sid_num'] = sewershed_gdf['sid'].map({sid: i for i, sid in dual.sewershed_labels.items()})
region_gdf = gpd.read_file('san-diego-data/sandiego_zip_boundaries.geojson')
region_gdf['region_num'] = region_gdf['region_code'].astype(int) # people.region holds the integer region codes

from matplotlib.patches import Patch
outline = gpd.GeoSeries([region_gdf.union_all()], crs=region_gdf.crs) # The whole area covered by the boundary file

def add_unmonitored_outline(ax):
    """Hatch the whole area, beneath the sewersheds, so the areas that are not monitored by wastewater stand out"""
    outline.plot(ax=ax, facecolor='white', edgecolor='0.6', hatch='///', linewidth=0.8, zorder=0)
    ax.legend(handles=[Patch(facecolor='white', edgecolor='0.6', hatch='///', label='Not in a sewershed')], loc='upper left', bbox_to_anchor=(0, 0.02), frameon=False, fontsize=8)

people = dual.people
shedding_by_agent = dual.results['viral_shedding_hist'].sum(axis=1) # Cumulative shedding per agent, for every agent ever created
n_agents = people.n_uids
sewershed_all = people.sewershed.raw[:n_agents] # .raw: every agent, including those who died, to match the rows of viral_shedding_hist
diagnosed = ~np.isnan(people.date_diagnosed) # Everyone alive at the end who was ever diagnosed

# Wastewater: cumulative shedding per capita (per resident) in each sewershed
for sid in sewershed_gdf['sid_num']:
    sewershed_gdf.loc[sewershed_gdf['sid_num'] == sid, 'shedding_per_capita'] = shedding_by_agent[sewershed_all == sid].mean()
sewershed_gdf['label'] = sewershed_gdf['sid'] # Plant name

# Clinical: diagnosed cases per region
for code in region_gdf['region_num']:
    in_region = people.region == code
    row = region_gdf['region_num'] == code
    region_gdf.loc[row, 'cases'] = diagnosed[in_region].sum()
    region_gdf.loc[row, 'cases_per_1000'] = 1000*diagnosed[in_region].sum()/in_region.sum() if in_region.any() else np.nan # ZIPs with nobody in the simulation are left blank

fig, (ax_cl, ax_ww) = plt.subplots(1, 2, figsize=(14, 7))
panels = [(ax_cl, region_gdf, 'cases_per_1000', 'Cumulative diagnosed cases per 1,000 residents', 'Clinical: by ZIP code'),
          (ax_ww, sewershed_gdf, 'shedding_per_capita', 'Cumulative viral shedding per capita', 'Wastewater: by sewershed')]
for ax, gdf, column, label, title in panels:
    gdf.plot(column=column, cmap='YlOrRd', vmin=0, edgecolor='0.3', legend=True,
             legend_kwds={'label': label, 'orientation': 'horizontal', 'shrink': 0.7}, ax=ax)
    if 'label' in gdf.columns: # Label the few sewersheds; there are too many ZIP codes to label
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            dark = row[column] > 0.5*gdf[column].max() # White text on dark cells, black on light ones
            ax.annotate(row['label'], xy=(c.x, c.y), ha='center', fontsize=7, fontweight='bold', color='white' if dark else 'black')
    ax.set_title(title)
    ax.set_axis_off()
add_unmonitored_outline(ax_ww)
plt.tight_layout()
plt.savefig(f'{plot_dir}/04_clinical_vs_wastewater_map.pdf')
plt.close()

# %% [markdown]
# Finally, the two surveillance modalities over time. For this animation we use a smaller simulation of 1,000 agents so that every agent can be drawn. The left panel shows each agent at their home location, coloured by their current viral load (grey = not infectious). The middle panel shows clinical data (new diagnoses by region) and the right panel wastewater (viral shedding per capita by sewershed). Each frame is one day, with 7-day means for the two maps. The colour scales are fixed across frames so changes over time are visible.

# %%
import pandas as pd
from matplotlib.colors import Normalize
from PIL import Image

class RecordViralLoad(cv.Analyzer):
    """Save every agent's viral load (relative transmissibility, 0 if not infectious) on every day"""

    def initialize(self, sim):
        super().initialize(sim)
        self.viral_load = np.zeros((sim['pop_size'], sim.npts)) # One row per agent (by UID), including those who die

    def apply(self, sim):
        self.viral_load[sim.people.auids, self.today] = sim.people.viral_load.values # The agents alive today

# A small simulation (1,000 agents) with the same boundaries, sewersheds and testing
small = cv.Sim(
    pop_size           = 1000,
    pop_infected       = 20,
    n_days             = 100,
    pop_type           = 'rank',
    density_raster     = 'san-diego-data/san_diego_county_ppp_2020.tif',
    boundary_shapefile = 'san-diego-data/sandiego_zip_boundaries.geojson',
    sewershed_file     = 'san-diego-data/sandiego_sewersheds.geojson',
    interventions      = [cv.test_prob(symp_prob=0.2, start_day=0)],
    analyzers          = [RecordViralLoad(label='viral_load')],
    verbose            = 0,
)
small.run()
viral_load_hist = small.get_analyzer('viral_load').viral_load # (n_agents, n_days)
small_people = small.people
small_diagnosed = ~np.isnan(small_people.date_diagnosed)
small_sewershed = small_people.sewershed.raw[:small_people.n_uids] # Every agent, to match the rows of the histories
n_days = small.npts
shedding_hist = small.results['viral_shedding_hist'] # (n_agents, n_days)

# Daily values, with rows in the same order as the polygons in each file:
# new diagnoses per 1,000 residents per region (clinical) and shedding per capita per sewershed (wastewater)
clinical_ts = np.zeros((len(region_gdf), n_days))
wastewater_ts = np.zeros((len(sewershed_gdf), n_days))
for k, code in enumerate(region_gdf['region_num']):
    in_region = small_people.region == code
    diag_days = small_people.date_diagnosed[in_region & small_diagnosed].astype(int)
    clinical_ts[k] = 1000*np.bincount(diag_days[diag_days < n_days], minlength=n_days)/in_region.sum() if in_region.any() else np.nan # ZIPs with nobody in the simulation are left blank
for k, sid in enumerate(sewershed_gdf['sid_num']):
    wastewater_ts[k] = shedding_hist[small_sewershed == sid].mean(axis=0)

def smooth(ts):
    return pd.DataFrame(ts.T).rolling(7, min_periods=1).mean().to_numpy().T # 7-day trailing mean

clinical_ts, wastewater_ts = smooth(clinical_ts), smooth(wastewater_ts)

# Agent colours: viral load is 0 for anyone who is not infectious, who are drawn grey. They are a separate scatter below
# (lower zorder) the infectious agents, and below the map lines (zorder 1: grey agents, 2: ZIP outlines, 3: infectious agents).
agent_cmap = plt.get_cmap('RdPu')
agent_norm = Normalize(vmin=0, vmax=viral_load_hist.max())
agent_xy = np.column_stack((small_people.x.raw[:small_people.n_uids], small_people.y.raw[:small_people.n_uids])) # Every agent, to match the rows of viral_load_hist

def agents_on_day(day):
    """ Locations of the non-infectious agents, and locations and viral loads of the infectious ones """
    load = viral_load_hist[:, day]
    infectious = load > 0
    return agent_xy[~infectious], agent_xy[infectious], load[infectious]

fig, (ax_agents, ax_clinical, ax_wastewater) = plt.subplots(1, 3, figsize=(15, 5.5))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9, wspace=0.02) # Tight spacing between the three maps
region_gdf.boundary.plot(ax=ax_agents, color='0.5', linewidth=1, zorder=2) # ZIP code outlines for context
not_infectious, infectious_xy, infectious_load = agents_on_day(0)
grey_scatter = ax_agents.scatter(not_infectious[:, 0], not_infectious[:, 1], color='lightgrey', s=12, zorder=1)
agent_scatter = ax_agents.scatter(infectious_xy[:, 0], infectious_xy[:, 1], c=infectious_load, cmap=agent_cmap, norm=agent_norm, s=12, zorder=3)
fig.colorbar(agent_scatter, ax=ax_agents, orientation='horizontal', shrink=0.7, label='Viral load (relative transmissibility)')
ax_agents.set_title('Agents')
ax_agents.set_axis_off()

panels = [(ax_clinical, region_gdf, clinical_ts, 'Clinical (by ZIP code)', 'New diagnoses per 1,000 residents'),
          (ax_wastewater, sewershed_gdf, wastewater_ts, 'Wastewater (by sewershed)', 'Viral shedding per capita')]
for ax, gdf, ts, title, label in panels:
    gdf['value'] = np.nan_to_num(ts[:, 0]) # Plot without gaps so every polygon is in one collection; blanks come from NaN in the later frames
    gdf.plot(column='value', cmap='YlOrRd', vmin=0, vmax=np.nanmax(ts), edgecolor='0.3', ax=ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, np.nanmax(ts)), cmap='YlOrRd'), ax=ax, orientation='horizontal', shrink=0.7, label=label)
    ax.set_title(title)
    ax.set_axis_off()

add_unmonitored_outline(ax_wastewater) # Added after the sewersheds, which the animation updates through ax.collections[0]
xmin, ymin, xmax, ymax = region_gdf.total_bounds # The same map extent in every panel, so the panels line up
for ax in (ax_agents, ax_clinical, ax_wastewater):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def update(day):
    not_infectious, infectious_xy, infectious_load = agents_on_day(day)
    grey_scatter.set_offsets(not_infectious)
    agent_scatter.set_offsets(infectious_xy)
    agent_scatter.set_array(infectious_load)
    for ax, gdf, ts, title, label in panels:
        ax.collections[0].set_array(ts[:, day])
    fig.suptitle(f'Day {day}')

def save_gif(fig, update, n_frames, path, duration=100):
    """
    Draw every frame (update(i) redraws frame i), then save one GIF that uses a single colour palette for all of them.
    (Matplotlib's PillowWriter builds a new 256-colour palette for each frame, which makes colours shift slightly between frames.)
    """
    frames = []
    for i in range(n_frames):
        update(i)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)

    sample = frames[::5] # The palette is built from a stack of every 5th frame
    stack = Image.new('RGB', (sample[0].width, sample[0].height*len(sample)))
    for i, frame in enumerate(sample):
        stack.paste(frame, (0, i*frame.height))
    palette = stack.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

save_gif(fig, update, n_days, f'{plot_dir}/05_clinical_vs_wastewater_over_time.gif')

# %% [markdown]
# ## Phylogeny from the transmission tree
#
# Covasim records who infected whom in `sim.people.infection_log`. With sequence evolution on, each entry also stores the mutations that arose along that transmission (`n_mutations`), so the transmission tree doubles as a phylogeny of the virus. Here we draw it with [baltic](https://github.com/evogytis/baltic), for a small outbreak of 1,000 agents. It starts from 3 wild-type infections, and AY.4 arrives later, on day 30, as 3 imports.
#
# * Every infection is a node, joined to the infection of its source that was active on that day. People can be infected more than once, so each reinfection is its own node.
# * Every infected person is a tip, drawn as a dot on the day they were infected. The people they go on to infect branch off at that same day, so every infection appears whole on the day it happens.
# * Branches are coloured by variant. Grey branches are ancestors of more than one variant, which is only the root where the separate seed infections join.

# %%
import collections
import baltic as bt
from matplotlib.lines import Line2D

# A small outbreak: 3 wild-type infections on day 0, then AY.4 arrives on day 30 as 3 imports
ay4_day = 30
tree_ay4 = cv.variant(label='AY.4', days=ay4_day, n_imports=3, founding_fasta=ay4_fasta)
outbreak = cv.Sim(
    pop_size     = 1000,
    pop_infected = 3,
    n_days       = 100,
    rand_seed    = 3,
    evo_pars     = dict(enable=True, reference=ref_fasta, mol_clock_rate=1e-6,
                        fitness_model='bloom_nt', fitness_scale=0.05),
    variants     = [tree_ay4],
    verbose      = 0,
)
outbreak.run()
log = outbreak.people.infection_log
n_days = outbreak.npts

# Each infection_log entry is one infection. Find the infection of the source that was active on a given day
infections = collections.defaultdict(list) # agent -> [(date, index in log)], oldest first
for k, entry in enumerate(log):
    infections[int(entry['target'])].append((entry['date'], k))

def active_infection(agent, day):
    return [k for date, k in infections[agent] if date <= day][-1]

parent = [None if e['source'] is None else active_infection(int(e['source']), e['date']) for e in log]
date = np.array([e['date'] for e in log])
n_muts = np.zeros(len(log)) # Mutations between the seed infection and each infection
for k, entry in enumerate(log): # The log is in time order, so a parent always comes before its children
    n_muts[k] = (n_muts[parent[k]] if parent[k] is not None else 0) + entry['n_mutations']
children = collections.defaultdict(list)
for k, p in enumerate(parent):
    if p is not None:
        children[p].append(k)
roots = [k for k in range(len(log)) if parent[k] is None]

ROOT_LEN = 0.001 # The seeds join at the root with a branch this long, so that no tree ever has branch lengths that are all 0 (baltic would rescale it)

def to_newick(day, measure):
    """ Newick string of the infections up to `day`. Every infected person is a tip, on the day they were infected.
    measure='days' or 'mutations' sets what branch lengths mean """
    at = lambda k: date[k] if measure == 'days' else n_muts[k] # Where an infection sits on the x axis
    def subtree(k, parent_at):
        kids = [c for c in children[k] if date[c] <= day]
        if not kids:
            return f'tip{k}:{at(k) - parent_at}'
        parts = [subtree(c, at(k)) for c in kids] + [f'tip{k}:0'] # The infected person is also a tip, where their lineage splits
        return f'({",".join(parts)}):{at(k) - parent_at}'
    root_at = min(at(r) for r in roots) - ROOT_LEN # The seed infections are dated day -1, the step before the first
    return '(' + ','.join(subtree(r, root_at) for r in roots if date[r] <= day) + '):0;' # The seeds and imports join at the root

# Tips are always drawn in the same order, so that the tree in the GIF grows without shuffling
tip_order = []
def collect_tips(k):
    for c in children[k]:
        collect_tips(c)
    tip_order.append(f'tip{k}')
for r in roots:
    collect_tips(r)
tip_y = {name: y for y, name in enumerate(tip_order)}
tip_variant = {f'tip{k}': e['variant'] for k, e in enumerate(log)}

def branch_colour(branch):
    variants = {tip_variant[name] for name in branch.leaves} if branch.is_node() else {tip_variant[branch.name]}
    return palette[variants.pop()] if len(variants) == 1 else '0.5'

def draw_tree(ax, day, measure, point_size):
    tree = bt.make_tree(to_newick(day, measure), 'divergence')
    tree.traverse_tree()
    y_cache = {}
    def y_of(branch): # Tips keep their place; a node sits halfway between its descendants
        if id(branch) not in y_cache:
            y_cache[id(branch)] = tip_y[branch.name] if branch.is_leaf() else np.mean([y_of(c) for c in branch.children])
        return y_cache[id(branch)]
    tree.plot_tree(ax, colourFxn=branch_colour, width=0.5, autoSort=False, yCoordinateFxn=y_of)
    tree.plot_points(ax, targetFxn=lambda k: k.is_leaf(), colourFxn=branch_colour, yCoordinateFxn=y_of, pointSize=point_size, outline=False)
    ax.set_ylim(-1, len(tip_order))
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    return tree

legend_handles = [Line2D([], [], color=palette[v], label=v) for v in ('wild', 'AY.4')] + [Line2D([], [], color='0.5', label='mixed')]

# %% [markdown]
# **All tips.** The tree of everyone infected during the outbreak. Left, branch lengths are time (the x axis is the simulation day). Right, branch lengths are the number of mutations. Mutations are rare at this clock rate (`mol_clock_rate=1e-6` per site per day), so most branches have none.

# %%
fig, (ax_days, ax_muts) = plt.subplots(1, 2, figsize=(14, 12))
for ax, measure, title, xlabel in [(ax_days, 'days', 'Time-scaled', 'Simulation day'), (ax_muts, 'mutations', 'Mutations', 'Mutations from the seed infection')]:
    newick = to_newick(n_days - 1, measure)
    with open(f'{plot_dir}/06_transmission_phylogeny_{measure}.nwk', 'w') as f:
        f.write(newick)
    tree = draw_tree(ax, n_days - 1, measure, point_size=3)
    ax.set_title(f'{title} ({len(tree.get_external())} infected agents)')
    ax.set_xlabel(xlabel)
plt.tight_layout(rect=(0, 0.03, 1, 1)) # Leave room for the legend under the trees
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False)
plt.savefig(f'{plot_dir}/06_transmission_phylogeny.pdf')
plt.close()

# %% [markdown]
# **The outbreak over time.** The tree on each day, with everyone infected so far as a tip, next to the number of infectious agents of each variant. Each frame is one day.

# %%
n_infectious = outbreak.results['variant']['n_infectious_by_variant'].values.T # (variant, day)
fig, (ax_tree, ax_counts) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw=dict(width_ratios=[2, 1]))
for v, name in outbreak['variant_map'].items():
    ax_counts.plot(np.arange(n_days), n_infectious[v], color=palette[name], label=name)
ax_counts.set_xlim(0, n_days - 1)
ax_counts.set_xlabel('Simulation day')
ax_counts.set_ylabel('Infectious agents')
ax_counts.legend(frameon=False)
ax_counts.axvline(ay4_day, color=palette['AY.4'], linestyle=':', linewidth=1.2) # AY.4 arrives
today = ax_counts.axvline(0, color='0.3', linewidth=1)
for side in ('top', 'right'):
    ax_counts.spines[side].set_visible(False)

def update(day):
    ax_tree.clear()
    draw_tree(ax_tree, day, 'days', point_size=6)
    ax_tree.set_xlim(0, n_days - 1) # The same time axis in every frame
    ax_tree.set_xlabel('Simulation day')
    ax_tree.set_title(f'Infected so far: {sum(date <= day)}')
    today.set_xdata([day, day])
    fig.suptitle(f'Day {day}')

save_gif(fig, update, n_days, f'{plot_dir}/07_transmission_phylogeny_over_time.gif')
