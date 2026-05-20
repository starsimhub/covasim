'''
Benchmarking and validation of new vs. old transmission tree code.
'''

import covasim as cv
import pandas as pd
import sciris as sc
import numpy as np

# Whether to validate (slow!)
validate = 1

# Create a sim
sim = cv.Sim(pop_size=20e3, n_days=100).run()
people = sim.people


sc.heading('Built-in implementation (Numpy)...')

tt = sim.make_transtree()

sc.tic()
tt.make_detailed(sim.people)
sc.toc()


sc.heading('Manual implementation (pandas)...')

sc.tic()

# Convert to a dataframe and initialize
idf = pd.DataFrame(sim.people.infection_log)

# Initialization
src = 'src_'
trg = 'trg_'
attrs = ['age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_severe', 'date_critical', 'date_known_contact']
quar_attrs = ['date_quarantined', 'date_end_quarantine']
date_attrs = [attr for attr in attrs if attr.startswith('date_')]
is_attrs = [attr.replace('date_', 'is_') for attr in date_attrs]

n_people = len(people)
ddf = pd.DataFrame(index=np.arange(n_people))

# Handle indices
trg_inds    = np.array(idf['target'].values, dtype=np.int64)
src_inds    = np.array(idf['source'].values)
date_vals   = np.array(idf['date'].values)
layer_vals  = np.array(idf['layer'].values)
src_arr     = np.nan*np.zeros(n_people)
trg_arr     = np.nan*np.zeros(n_people)
infdate_arr = np.nan*np.zeros(n_people)
src_arr[trg_inds]  = src_inds
trg_arr[trg_inds]  = trg_inds
infdate_arr[trg_inds] = date_vals
ts_inds = sc.findinds(~np.isnan(trg_arr) * ~np.isnan(src_arr))
v_src = np.array(src_arr[ts_inds], dtype=np.int64)
v_trg = np.array(trg_arr[ts_inds], dtype=np.int64)
vinfdates = infdate_arr[v_trg] # Valid target-source pair infection dates
ainfdates = infdate_arr[trg_inds] # All infection dates

# Populate main things
ddf.loc[v_trg, 'source'] = v_src
ddf.loc[trg_inds, 'target'] = trg_inds
ddf.loc[trg_inds, 'date'] = ainfdates
ddf.loc[trg_inds, 'layer'] = layer_vals

# Populate from people
for attr in attrs+quar_attrs:
    ddf.loc[:, trg+attr] = people[attr][:]
    ddf.loc[v_trg, src+attr] = people[attr][v_src]

# Replace nan with false
def fillna(cols):
    cols = sc.promotetolist(cols)
    filldict = {k:False for k in cols}
    ddf.fillna(value=filldict, inplace=True)
    return

# Pull out valid indices for source and target
ddf.loc[v_trg, src+'is_quarantined'] = (ddf.loc[v_trg, src+'date_quarantined'] <= vinfdates) & ~(ddf.loc[v_trg, src+'date_quarantined'] <= vinfdates)
fillna(src+'is_quarantined')
for is_attr,date_attr in zip(is_attrs, date_attrs):
    ddf.loc[v_trg, src+is_attr] = (ddf.loc[v_trg, src+date_attr] <= vinfdates)
    fillna(src+is_attr)

ddf.loc[v_trg, src+'is_asymp'] = np.isnan(ddf.loc[v_trg, src+'date_symptomatic'])
ddf.loc[v_trg, src+'is_presymp'] = ~ddf.loc[v_trg, src+'is_asymp'] & ~ddf.loc[v_trg, src+'is_symptomatic']
ddf.loc[trg_inds, trg+'is_quarantined'] = (ddf.loc[trg_inds, trg+'date_quarantined'] <= ainfdates) & ~(ddf.loc[trg_inds, trg+'date_end_quarantine'] <= ainfdates)
fillna(trg+'is_quarantined')

sc.toc()


sc.heading('Original implementation (dicts)...')

sc.tic()

detailed = [None]*len(sim.people)

for transdict in sim.people.infection_log:

    # Pull out key quantities
    ddict  = {}
    source = transdict['source']
    target = transdict['target']

    # If the source is available (e.g. not a seed infection), loop over both it and the target
    if source is not None:
        stdict = {'src_':source, 'trg_':target}
    else:
        stdict = {'trg_':target}

    # Pull out each of the attributes relevant to transmission
    attrs = ['age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_end_quarantine', 'date_severe', 'date_critical', 'date_known_contact']
    for st,stind in stdict.items():
        for attr in attrs:
            ddict[st+attr] = people[attr][stind]
    if source is not None:
        for attr in attrs:
            if attr.startswith('date_'):
                is_attr = attr.replace('date_', 'is_') # Convert date to a boolean, e.g. date_diagnosed -> is_diagnosed
                if attr == 'date_quarantined': # This has an end date specified
                    ddict['src_'+is_attr] = ddict['src_'+attr] <= transdict['date'] and not (ddict['src_'+'date_end_quarantine'] <= ddict['date'])
                elif attr != 'date_end_quarantine': # This is not a state
                    ddict['src_'+is_attr] = ddict['src_'+attr] <= transdict['date'] # These don't make sense for people just infected (targets), only sources

        ddict['src_'+'is_asymp']   = np.isnan(people.date_symptomatic[source])
        ddict['src_'+'is_presymp'] = ~ddict['src_'+'is_asymp'] and ~ddict['src_'+'is_symptomatic'] # Not asymptomatic and not currently symptomatic
    ddict['trg_'+'is_quarantined'] = ddict['trg_'+'date_quarantined'] <= transdict['date'] and not (ddict['trg_'+'date_end_quarantine'] <= ddict['date']) # This is the only target date that it makes sense to define since it can happen before infection

    ddict.update(transdict)
    detailed[target] = ddict

sc.toc()


if validate:
    sc.heading('Validation...')
    sc.tic()
    for i in range(len(detailed)):
        sc.percentcomplete(step=i, maxsteps=len(detailed)-1, stepsize=5)
        d_entry = detailed[i]
        df_entry = ddf.iloc[i].to_dict()
        tt_entry = tt.detailed.iloc[i].to_dict()
        if d_entry is None: # If in the dict it's None, it should be nan in the dataframe
            for entry in [df_entry, tt_entry]:
                assert np.isnan(entry['target'])
        else:
            dkeys  = list(d_entry.keys())
            dfkeys = list(df_entry.keys())
            ttkeys = list(tt_entry.keys())
            assert dfkeys == ttkeys
            # assert all([dk in dfkeys for dk in dkeys]) # The dataframe can have extra keys, but not the dict
            for k in dkeys:
                if k in dfkeys: # Skip variant key
                    v_d = d_entry[k]
                    v_df = df_entry[k]
                    v_tt = tt_entry[k]
                    try:
                        assert np.isclose(v_d, v_df, v_tt, equal_nan=True) # If it's numeric, check they're close
                    except TypeError:
                        if v_d is None:
                            assert all(np.isnan([v_df, v_tt])) # If in the dict it's None, it should be nan in the dataframe
                        else:
                            assert v_d == v_df == v_tt # In all other cases, it should be an exact match
    sc.toc()
    print('\nValidation passed.')


print('Done.')