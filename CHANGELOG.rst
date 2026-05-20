==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1



~~~~~~~~~~~
Coming soon
~~~~~~~~~~~

Covasim is currently in maintenance phase, and there are no further major releases planned. If there is a specific bugfix or feature you would like to see, please `create an issue <https://github.com/InstituteforDiseaseModeling/covasim/issues/new/choose>`_.


~~~~~~~~~~~~~~~~~~~~~~~
Latest versions (3.1.x)
~~~~~~~~~~~~~~~~~~~~~~~

Version 3.1.7 (2026-02-18)
--------------------------
- Fixed a deprecation in NumPy 2.4 that affcted interventions.
- *GitHub info*: PR `1452 <https://github.com/amath-idm/covasim/pull/1452>`_


Version 3.1.6 (2024-01-28)
--------------------------
- Reverted to ``multiprocess`` instead of ``concurrent.futures`` as the default parallelization method to prevent run failures on Macs. To continue using ``concurrent.futures``, add an argument to ``run()``, e.g. ``msim.run(par_args={'parallelizer':'fast'})`` (``'fast'`` defaults to ``concurrent.futures``, while ``'robust'`` defaults to ``multiprocess``; see all options `here <https://sciris.readthedocs.io/en/latest/api/_autosummary/sciris.sc_parallel.parallelize.html#sciris.sc_parallel.parallelize>`_).
- ``get_intervention()`` and ``get_analyzer()`` now return a match if there is an exact match for a class name. Previously, this only used ``isinstance()``, which sometimes failed during parallel processing since these required pickling and unpickling the intervention, and upon unpickling Python would (incorrectly) consider it to be a different class.
- *GitHub info*: PR `1439 <https://github.com/amath-idm/covasim/pull/1439>`_


Version 3.1.5 (2023-12-15)
--------------------------
- Fixed a deprecation in ``pandas`` that prevented displaying the summary table of interventions and analyzers.
- *GitHub info*: PR `1433 <https://github.com/amath-idm/covasim/pull/1433>`_


Version 3.1.4 (2022-10-22)
--------------------------

Improvements
^^^^^^^^^^^^
- Added a new ``isolated`` state to explicitly track people who are in isolation (like quarantine).
- Switched default parallelizer from ``multiprocess`` to ``concurrent.futures``; the latter is faster but less robust, so will automatically try again using ``multiprocess`` if ``concurrent.futures`` fails.
- Added ``cv.Fit.summarize()`` to quickly get information about the mismatches.

Bugfixes
^^^^^^^^
- Fixed plotting bug that prevented ``MultiSim`` plots from appearing correctly in Jupyter notebooks with (ironically) ``cv.options(jupyter=True)``.
- Fixed bug with Matplotlib options (e.g. ``cv.options(dpi=150)``) not being set properly.
- ``cv.Calibration.remove_db()`` now deletes the Optuna study as well. 
- Fixed several issues with ``cv.git_info()`` pulling information from the wrong frame (i.e. file location).
- Fixed shrunken sims dropping interventions/analyzers that were functions rather than classes.

Other
^^^^^
- Updated Sciris dependency to 2.0.1.
- Updated license to MIT instead of Creative Commons.
- Added ``papers.rst`` and ``papers.bib`` that list additional Covasim publications.
- Added a style guide and draft code linting scripts.
- *GitHub info*: PR `1396 <https://github.com/amath-idm/covasim/pull/1396>`_


Version 3.1.3 (2022-07-19)
--------------------------
- Fixed a bug with using ``'seir'`` as a default plot option. (Thanks Rik Belew for finding and fixing.)
- Updated immunity calculations to address edge cases in historical vaccination and multiple sources of immunity. Immunity calculations were also updated to skip variants with no infections for a small performance gain.
- Previously, when a ``People`` object was passed into a sim, it was recreated using a new instance. Now, the existing instance is preserved. To restore the previous behavior, use the new ``recreate`` keyword argument to ``cv.make_people()``, e.g. ``sim = cv.Sim(people=people); sim.people = cv.make_people(sim, sim.people, recreate=True)``.
- ``People`` methods (e.g. ``check_infectious()`` now return the indices of the people whose state changed, rather than just the total number.
- *Regression information*: Immunity calculations have changed, so all simulations will produce stochastically different results. Simulations that use historical immunity (vaccination or waves) or multiple circulating variants may produce significantly different (not just stochastically different) results.
- *GitHub info*: PR `1374 <https://github.com/amath-idm/covasim/pull/1374>`_


Version 3.1.2 (2022-01-16)
--------------------------

Highlights
^^^^^^^^^^
- **New styles**: Plots have gotten a complete revamp.
- **More options**: There are now many more options to control plot styles, warnings, etc.
- **Populations**: It is now easier to work with pre-generated populations and contacts.

Plotting
^^^^^^^^
- Default plotting styles have been updated. Run e.g. ``cv.Sim(n_days=365).run().plot()`` to see most of the changes. Major changes include: a new date formatter, grid lines à la Seaborn, new bundled fonts (`Mulish <https://fonts.google.com/specimen/Mulish>`_ and `Rosario <https://fonts.google.com/specimen/Rosario>`_), finer control of style options, and better defaults for Jupyter.
- Changing styles is now much easier, e.g. ``sim.plot(style='seaborn-whitegrid')`` will use the named Matplotlib style. See ``cv.options.help()`` for more information.
- Covasim now uses Sciris' default date formatter, which is similar to Plotly's. You can change this using the ``dateformat`` argument, e.g. ``sim.plot(dateformat='sciris')`` vs. ``sim.plot(dateformat='concise')``.
- ``cv.savefig()`` can now save multiple figures simultaneously via the (new) ``fig`` argument.
- Figure window names can now be passed to plot. This is called ``num`` by Matplotlib, so ``sim.plot(num='My plot')`` or ``sim.plot(fig_args={'num':'My plot'})`` will work.
- You can plot every nth data point by passing the ``datastride`` argument to ``sim.plot()``. 
- You can do automatic figure layout using the ``tight`` and ``maximize`` arguments, e.g. ``sim.plot('overview', tight=True, maximize=True)``. (Note: maximize may not work on all systems.)

People and populations
^^^^^^^^^^^^^^^^^^^^^^
- The ``cv.Sim()`` now has arguments ``popfile`` and ``people``, rather than ``popfile``, ``loadpop``, and ``savepop``. Populations are now automatically loaded if ``popfile`` is provided. You can now also pass a ``People`` object directly in.
- ``People`` objects now have ``save()`` and ``load()`` methods to replace doing this from within the sim. It is now an error by default to save a partially-run ``People`` object.
- To create a population inside a sim for later use, create it with ``sim = cv.Sim().init_people()``, then save it with ``sim.people.save('people.ppl')``, then load with ``cv.Sim(popfile='people.ppl')``.
- Contacts can be added more easily and flexibly. For example, contacts created with ``cv.make_random_contacts()`` can now be added directly with ``people.add_contacts()``.
- Previously, infections were always initialized as part of ``sim.init_people()``, but now it is a separate method (``sim.init_infections()``).
- The methods ``people.to_people()`` and ``people.from_people()`` have been renamed ``people.to_list()`` and ``people.from_list()``.
- Fixed a bug preventing SynthPops populations from being loaded. (Thanks to Carter Merenstein for finding this bug.)

Other changes
^^^^^^^^^^^^^
- Added ``cv.help()``, which will search docstrings (or full source code) for matches to search phrases.
- Printed warning messages have been converted to actual warnings: use ``cv.options(warnings='print')`` to restore previous behavior.
- Parameters for individual variants can now be retrieved more easily, e.g. ``cv.get_variant_pars(variant='delta')``.
- MultiSim objects now automatically add labels to any sims that are missing labels.
- When old objects are loaded, their versions numbers are no longer changed, allowing for multiple migrations to occur.

Regression information
^^^^^^^^^^^^^^^^^^^^^^
- In ``cv.make_people()``, the arguments ``save_pop`` and ``popfile`` have been removed; call ``people.save()`` on the generated population instead.
- In ``cv.make_synthpop()``, the argument ``population`` has been renamed ``popdict``, and will accept any SynthPops output (``popdict``, ``Pop``, or ``People``).
- Calls to ``people.to_people()`` and ``people.from_people()`` should be replaced with ``people.to_list()`` and ``people.from_list()``, respectively.
- Arguments ``font_family``, ``font_size``, and ``mpl_args`` to plots should be replaced with ``font``, ``fontsize``, and ``style_args`` respectively.
- ``cv.date_formatter()`` has been removed; please use ``sc.dateformatter()`` (for a date x-axis) or ``sc.datenumformatter()`` (for a numeric axis that you want to format as dates) instead.
- The ``columns`` argument has been removed from ``cv.load_data()``. If needed, load the data as a dataframe, filter the columns, then pass it to the sim.
- *GitHub info*: PR `1295 <https://github.com/amath-idm/covasim/pull/1295>`_


Version 3.1.1 (2021-12-06)
--------------------------

Performance improvements
^^^^^^^^^^^^^^^^^^^^^^^^
Performance improvements have been implemented in all aspects of Covasim, including:

- By changing certain imports to be just-in-time rather than up-front, module import time (``import covasim as cv``) was roughly halved (from about 0.7 s to 0.4 s).
- Population generation methods have been refactored; initializing a sim is now twice as fast (from about 0.4 s to 0.2 s for 20,000 people).
- Immunity and infection methods have been refactored, leading to about a 30% improvement in run time (from about 0.7 to 0.5 s for 20,000 people 60 days).

Bugfixes
^^^^^^^^
- Fixed a bug in which ``sim.pars`` and ``sim.people.pars`` were not the same object. In almost all cases, the latter should now be a link to the former.
- Fixed a bug whereby interventions and analyzers were not being shrunk correctly.
- Fixed a bug with 2nd vaccine doses not being rescheduled if zero doses were given on the day they were scheduled.

Other updates
^^^^^^^^^^^^^
- A new convenience function has been added: ``cv.parallel(sim1, sim2)`` is equivalent to ``cv.MultiSim([sim1, sim2]).run()``.
- Calibrations now have multiple new analysis and plotting features, including ``calib.plot_trend()``, ``calib.plot_best()``, ``calib.to_json()``, etc.  ``calib.plot()`` has been renamed to ``calib.plot_sims()``.
- By default, calibrations now keep going if a sim encounters an exception; to restore the previous behavior, use ``sim.calibrate(die=True)``. A calibration run with a single worker now does not use ``multiprocess``, to simplify debugging.
- There is a new option for changing the thousands separator (e.g. to use European formatting), via e.g. ``cv.options.set(sep='.')``. This does not yet apply to plots, but will in a future version.
- A convenience method has been added for setting correct plot options for Jupyter: ``cv.options.set('jupyter')``.
- Population generation functions ``make_random_contacts()`` and ``make_microstructured_contacts()`` were updated to generate edgelists rather than lists-of-dicts.
- ``cv.poisson_test()`` was removed as as it was no longer being used.
- Tutorials, examples, and the FAQ have all been updated. In particular, all tutorials are now available to be run interactively with Binder via http://tutorials.covasim.org.
- *Regression information*: The new infection calculation method is mathematically identical but draws differently from the random number stream, giving stochastically different results than before. To revert to the previous (slower) calculation method, set ``sim._legacy_trans = True`` after initialization. This legacy option is automatically enabled if running with an earlier version of parameters, e.g. ``cv.Sim(version='3.1.0')``. Calls to ``calib.plot()`` should be replaced with ``calib.plot_sims()``. If you were using ``cv.poisson_test()``, you're on your own now, but are invited to copy it from an older version of Covasim.
- *GitHub info*: PR `1249 <https://github.com/amath-idm/covasim/pull/1249>`_


Version 3.1.0 (2021-12-03)
--------------------------
This version contains important updates to the parameters around immunity. It also introduces additional features designed to help with policy questions relevant at this stage of the pandemic, including support for boosters and the ability to initialize a population with pre-existing immunity. Although we will continue to update parameter values as new data come in, the immunity and vaccine features are now out of the beta stage and ready to use.

Highlights
^^^^^^^^^^
- **New immunity parameters**: Waning immunity and cross-immunity functions have been updated to match currently available empirical data.
- **Additional flexibility with vaccines**: Several new vaccines have been added (e.g. Sinopharm), and additional options have been provided to enable booster doses, simplify age targeting, etc.
- **Historical immunity**: To avoid the need to calibration to past epidemic waves and vaccine rollouts, new interventions have been added that let you control immunity levels from historical events.

Immunity-related parameter changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- By default, simulations now use full immune dynamics (i.e. ``use_waning=True``).
- When NAbs are primed, they are normalized to be equivalent to "vaccine NAbs". This is done so that when we check immunity, we can calculate immune protection using a single curve and account for multiple sources of immunity (vaccine and natural).
- Antibody kinetics were adjusted based on recent observational data suggesting a faster decay of NAbs and subsequent protection against infection. Source: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02183-8/fulltext
- A parameter ``trans_redux`` has been added to capture the reduction in transmission for breakthrough infections.
- Cross-immunity parameters have been updated.
- Default variant names now follow WHO conventions, e.g. ``'alpha'`` rather than ``'b117'``. (The other names can still be used, however.)
- ``'sinopharm'`` and ``'sinovac'`` have been added as built-in vaccines. Aliases have been added for other vaccines (e.g. ``'spikevax'`` for Moderna).
- Vaccine interventions now support booster doses.
- Age-targeting of vaccines can now be specified as e.g. ``sequence='age'``.

Changes to states and results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``people.vaccinations`` has been renamed to ``people.doses``, and keeps track of how many doses of any vaccine each agent has had. Likewise, ``new_vaccinations`` and ``cum_vaccinations`` have been renamed ``new_doses`` and ``cum_doses``.
- People have a new state, ``n_breakthroughs``, which tracks how many breakthrough infections they've had.
- NAb states have been updated: ``prior_symptoms`` has been removed and ``t_nab_event`` (the time when they were infected or vaccinated) has been added.
- A new result, ``n_imports``, has been added, which counts the number of imported infections (including from variants).

New functions, methods and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Added three new interventions designed to initiate a population with some prior immunity. The class ``cv.prior_immunity()`` is a wrapper for two options, ``cv.historical_vaccinate_prob()`` and  ``cv.historical_wave()``.
- ``cv.historical_vaccinate_prob()`` allocates vaccines parametrized by the daily probability of being vaccinated.  Unlike ``cv.vaccinate_prob()``, this function allows vaccination prior to ``t=0`` (and continuing into the simulation).
- ``cv.historical_wave()`` imprints a historical (pre ``t=0``) wave of infections in the population NAbs.
- A new analyzer, ``cv.nab_histogram()``, allows easy computation of statistics relating to NAbs.

Bugfixes
^^^^^^^^
- Keyword arguments to ``cv.Fit()`` are now correctly passed to ``cv.compute_gof()``. (Thanks to Zishu Liu for finding this bug.)
- The transmission tree can now be exported using the latest version of NetworkX. (Thanks to Alexander Zarebski for finding this bug.)
- The ``r_eff`` calculation method has been updated to avoid divide-by-zero issues.
- Rescaling now does *not* reset vaccination status; previously, dynamic rescaling erased it.
- Previously, ``cv.clip_edges()`` and ``cv.vaccinate_prob()`` used a lot of memory; these "memory leaks" have been fixed with new ``finalize()`` methods.
- Some results (e.g. number of tests) were being incorrectly rounded to integers prior to rescaling; this has been fixed.
- Imported infections are now sampled without replacement.
- Scenarios now re-initialize the sim object. The scenario label now matches the scenario name rather than key.

Other changes
^^^^^^^^^^^^^
- Result fields can now be accessed as keys as well as attributes, e.g. any combination of ``msim.results['r_eff']['high']`` and ``msim.results.r_eff.high`` works.
- Interventions and analyzers now have a ``shrink()`` method, for cleaning up memory-hungry intermediate results at the end of a simulation.
- By default, calibration now removes the database of individual trials. Set ``keep_db=True`` to keep it. There is also a ``remove_db()`` method to manually remove the database.
- Population creation methods have been updated to be more flexible, with keyword arguments being passed to helper functions.
- Simulation summaries now by default use comma-separated values. To change this to e.g. a dot, you can set a global option: ``cv.options.set(sep='.')``, or e.g. ``sim.summarize(sep='')``.
- ``cv.diff_sims()`` can now optionally skip specific results using the ``skip`` keyword.
- Vaccination is now included in the regression tests.

Regression information
^^^^^^^^^^^^^^^^^^^^^^
- Results for simulations with ``use_waning=True`` will be substantially different due to the update in parameters and functional form.
- ``r_eff`` results will not match previous versions due to the change in calculation method (but differences should be slight).
- Simulations that have been saved to disk which include variants may not work correctly. If this is an issue, please email us and we can help write a migration script.
- *GitHub info*: PR `1130 <https://github.com/amath-idm/covasim/pull/1130>`_



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 3.0.x (3.0.0 – 3.0.7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 3.0.7 (2021-06-29)
--------------------------
- Added parameters for the Delta variant.
- Refactored the NAb decay function to match the published version of `Khoury et al <https://www.nature.com/articles/s41591-021-01377-8>`_ (the previous implementation matched the preprint).
- Added optional ``capacity`` limit for ``cv.contact_tracing`` to cap the maximum number of people that can be traced each day.
- When loading a population from file, this is now done during sim initialization (``sim.initialize()``); previously this was done as part of sim creation (``cv.Sim()``). This fixed a bug with immunity characteristics not being initialized correctly. (Thanks to Paula Sanz-Leon for identifying and proposing a fix.)
- Fixed a log of 0 warning with NAbs.
- Fixed ``n_beds_hosp = 0`` and ``n_beds_icu = 0`` being ignored (for no limit, use ``n_beds_hosp = None`` or ``n_beds_hosp = np.inf``; thanks to Ankit Majhi for finding this bug).
- Added a more helpful error message if you try to export a MultiSim to JSON or Excel without reducing it first. (Thanks to Andrew Clark for finding this bug.)
- *Regression information*: Due to the change in NAb decay function, simulations run with ``use_waning = True`` will be slightly different than before. We are aiming to have a (relatively) stable version by Covasim v3.1; in the mean time, this aspect of the model may continue to receive frequent updates.
- *GitHub info*: PR `1102 <https://github.com/amath-idm/covasim/pull/1102>`_


Version 3.0.6 (2021-06-21)
--------------------------
- Added alpha, beta, and gamma as aliases for variants B117, B1351, and P1, respectively.
- Split vaccine implementation to separate the state changes associated with vaccinating a person from the allocation/prioritization of vaccine distribution. The base class ``cv.BaseVaccination`` implements vaccinating individuals, and derived classes define the ``cv.BaseVaccination.select_people()`` method which determines who to vaccinate each timestep.
- Added ``cv.vaccinate_num()`` as an alternate way to allocate vaccines. This intervention specifies the order in which to vaccinate people, and the number of doses to distribute each day.
- Renamed ``cv.vaccinate()`` to ``cv.vaccinate_prob()``, but added ``cv.vaccinate()`` as an alias that can be used (more or less) interchangeably with ``cv.vaccinate_prob()``.
- Updated NAb kinetics so that the NAb level no longer exceeds the peak NAb value after the second dose, and updated ``nab_growth_decay`` so that the NAb level no longer increases in the second decay phase (i.e. after 250 days by default). **Note**: we are in the process of changing the functional form for the NAb waning, so this will likely change again in version 3.0.7.
- Vaccine parameters for simulations with multiple different vaccines are now correctly handled. Previously only the first vaccine's parameters were used.
- Added a new ``fit_args`` argument to the ``Calibration`` class, allowing arguments to be passed to ``sim.compute_fit()``. Also added a ``par_samplers`` argument, allowing different Optuna samplers to be specified.
- *Regression information*: ``cv.vaccination`` has been renamed to ``cv.vaccinate_prob`` (however, ``cv.vaccinate()`` is retained as an alias to ``cv.vaccinate_prob()``, so user code should not break). The correction to the NAb decay implementation means results in simulations with vaccines and a long duration (e.g., >250 days) may differ -- vaccines are expected to be slightly less effective.
- *GitHub info*: PR `1088 <https://github.com/amath-idm/covasim/pull/1088>`_


Version 3.0.5 (2021-05-26)
--------------------------
- Changed all reference to variants from ``strain`` to ``variant``. For example, ``cv.strain()`` is now ``cv.variant()``, ``cv.Sim(strains=...)`` is now ``cv.Sim(variants=...)``, etc. See `this article <https://www.forbes.com/sites/jvchamary/2021/02/28/coronavirus-covid-variant-mutant-strain/?sh=4459cbc82241>`_ for the rationale behind the change.
- Changed the ``nab_to_efficacy`` function based on a joint estimation of the marginal vaccine efficacies and inferred conditional efficacies.
- Changed the parameters provided to ``nab_to_efficacy`` function.
- Updated some strain parameters to be based on studies and not modeled inferences.
- *Regression information*: All instances of ``strain`` should be renamed ``variant``. A find-and-replace should be sufficient for updating most scripts. Parameter values and functional forms have also been updated, so results using waning immunity will differ.
- *GitHub info*: PR `1069 <https://github.com/amath-idm/covasim/pull/1069>`__


Version 3.0.4 (2021-05-19)
--------------------------
- Fixed a bug that prevented simulations from being run *without* prognoses by age.
- Fixed an array length mismatch for single-dose vaccines.
- The default antibody kinetics are now a 3-part curve, with a 14-day growth, 250 day exp decay and then another exponential decay with a exponentially decaying decay parameter. This is captured in the new NAb functional form, ``nab_growth_decay``. To align with this change, NAbs are now initialized at the time of infection, so that individuals build immunity over the course of infection. 
- Some strain parameter changes based on https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348
- Added strain to the infection log
- Removed the ``rel_imm_strain`` parameter; self-immunity is now always 1.0.
- Updated vaccine and strain parameter values based on fits to empirical data.
- Merged multisims now use the labels from each multisim, rather than the sim labels, for plotting.
- *Regression information*: Parameter values have been updated, so results using waning immunity will differ.
- *GitHub info*: PR `1058 <https://github.com/amath-idm/covasim/pull/1058>`__


Version 3.0.3 (2021-05-17)
--------------------------
- Added a new class, ``cv.Calibration``, that can perform automatic calibration. Simplest usage is ``sim.calibrate(calib_pars)``. Note: this requires Optuna, which is not installed by default; please install separately via ``pip install optuna``. See the updated calibration tutorial for more information.
- Added a new result, ``known_deaths``, which counts only deaths among people who have been diagnosed.
- Updated several vaccine and variant parameters (e.g., B1.351 and B117 cross-immunity).
- ``sim.compute_fit()`` now returns the fit by default, and creates ``sim.fit`` (previously, this was stored in ``sim.results.fit``).
- *Regression information*: Calls to ``sim.results.fit`` should be replaced with ``sim.fit``. The ``output`` parameter for ``sim.compute_fit()`` has been removed since it now always outputs the ``Fit`` object.
- *GitHub info*: PR `1047 <https://github.com/amath-idm/covasim/pull/1047>`__


Version 3.0.2 (2021-04-26)
--------------------------
- Added Novavax as one of the default vaccines.
- If ``use_waning=True``, people will now become *undiagnosed* when they recover (so they are not incorrectly marked as diagnosed if they become reinfected).
- Added a new method, ``sim.to_df()``, that exports results to a pandas dataframe.
- Added ``people.lock()`` and ``people.unlock()`` methods, so you do not need to set ``people._lock`` manually.
- Added extra parameter checking to ``people.set_pars(pars)``, so ``pop_size`` is guaranteed to be an integer.
- Flattened ``sim['immunity']`` to no longer have separate axes for susceptible, symptomatic, and severe.
- Fixed a bug in ``cv.sequence()``, introduced in version 2.1.2, that meant it would only ever trigger the last intervention.
- Fixed a bug where if subtargeting was used with ``cv.vaccinate()``, it would trigger on every day.
- Fixed ``msim.compare()`` to be more careful about not converting all results to integers.
- *Regression information*: If you are using waning, ``sim.people.diagnosed`` no longer refers to everyone who has ever been diagnosed, only those still infectious. You can use ``sim.people.defined('date_diagnosed')`` in place of ``sim.people.true('diagnosed')`` (before these were identical).
- *GitHub info*: PR `1020 <https://github.com/amath-idm/covasim/pull/1020>`__


Version 3.0.1 (2021-04-16)
--------------------------
- Immunity and vaccine parameters have been updated.
- The ``People`` class has been updated to remove parameters that were copied into attributes; thus there is no longer both ``people.pars['pop_size']`` and ``people.pop_size``; only the former. Recommended practice is to use ``len(people)`` to get the number of people.
- Loaded population files can now be used with more than one strain; arrays will be resized automatically. If there is a mismatch in the number of people, this will *not* be automatically resized.
- A bug was fixed with the ``rescale`` argument to ``cv.strain()`` not having any effect.
- Dead people are no longer eligible to be vaccinated.
- *Regression information*: Any user scripts that call ``sim.people.pop_size`` should be updated to call ``len(sim.people)`` (preferred), or ``sim.n``, ``sim['pop_size']``, or ``sim.people.pars['pop_size']``.
- *GitHub info*: PR `999 <https://github.com/amath-idm/covasim/pull/999>`__


Version 3.0.0 (2021-04-13)
--------------------------
This version introduces fully featured vaccines, variants, and immunity. **Note:** These new features are still under development; please use with caution and email us at info@covasim.org if you have any questions or issues. We expect there to be several more releases over the next few weeks as we refine these new features.

Highlights
^^^^^^^^^^
- **Model structure**: The model now follows an "SEIS"-type structure, instead of the previous "SEIR" structure. This means that after recovering from an infection, agents return to the "susceptible" compartment. Each agent in the simulation has properties ``sus_imm``, ``trans_imm`` and ``prog_imm``, which respectively determine their immunity to acquiring an infection, transmitting an infection, or developing a more severe case of COVID-19. All these immunity levels are initially zero. They can be boosted by either natural infection or vaccination, and thereafter they can wane over time or remain permanently elevated. 
- **Multi-strain modeling**: Model functionality has been extended to allow for modeling of multiple different co-circulating strains with different properties. This means you can now do e.g. ``b117 = cv.strain('b117', days=1, n_imports=20)`` followed by ``sim = cv.Sim(strains=b117)`` to import strain B117. Further examples are contained in ``tests/test_immunity.py`` and in Tutorial 8.
- **New methods for vaccine modeling**: A new ``cv.vaccinate()`` intervention has been added, which allows more flexible modeling of vaccinations. Vaccines, like natural infections, are assumed to boost agents' immunity.
- **Consistency**: By default, results from Covasim 3.0.0 should exactly match Covasim 2.1.2. To use the new features, you will need to manually specify ``cv.Sim(use_waning=True)``.
- **Still TLDR?** Here's a quick showcase of the new features:

.. code-block:: python

    import covasim as cv

    pars = dict(
        use_waning    = True,  # Use the new immunity features
        n_days        = 180,   # Set the days, as before
        n_agents      = 50e3,  # New alias for pop_size
        scaled_pop    = 200e3, # New alternative to specifying pop_scale
        strains       = cv.strain('b117', days=20, n_imports=20), # Introduce B117
        interventions = cv.vaccinate('astrazeneca', days=80), # Create a vaccine
    )

    cv.Sim(pars).run().plot('strain') # Create, run, and plot strain results

Immunity-related parameter changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- A new control parameter, ``use_waning``, has been added that controls whether to use new waning immunity dynamics ("SEIS" structure) or the old dynamics where post-infection immunity was perfect and did not wane ("SEIR" structure). By default, ``use_waning=False``.
- A subset of existing parameters have been made strain-specific, meaning that they are allowed to differ by strain. These include: ``rel_beta``, which specifies the relative transmissibility of a new strain compared to the wild strain; ``rel_symp_prob``, ``rel_severe_prob``, ``rel_crit_prob``, and the newly-added immunity parameters ``rel_imm`` (see next point). The list of parameters that can vary by strain is specified in ``defaults.py``. 
- The parameter ``n_strains`` is an integer that specifies how many strains will be in circulation at some point during the course of the simulation. 
- Seven new parameters have been added to characterize agents' immunity levels:
   - The parameter ``nab_init`` specifies a distribution for the level of neutralizing antibodies that agents have following an infection. These values are on log2 scale, and by default they follow a normal distribution.
   - The parameter ``nab_decay`` is a dictionary specifying the kinetics of decay for neutralizing antibodies over time.
   - The parameter ``nab_kin``  is constructed during sim initialization, and contains pre-computed evaluations of the nab decay functions described above over time. 
   - The parameter ``nab_boost`` is a multiplicative factor applied to a person's nab levels if they get reinfected.
   - The parameter ``cross_immunity``. By default, infection with one strain of SARS-CoV-2 is assumed to grant 50% immunity to infection with a different strain. This default assumption of 50% cross-immunity can be modified via this parameter (which will then apply to all strains in the simulation), or it can be modified on a per-strain basis using the ``immunity`` parameter described below.
   - The parameter ``immunity`` is a matrix of size ``total_strains`` by ``total_strains``. Row ``i`` specifies the immunity levels that people who have been infected with strain ``i`` have to other strains.
   - The parameter ``rel_imm`` is a dictionary with keys ``asymp``, ``mild`` and ``severe``. These contain scalars specifying the relative immunity levels for someone who had an asymptomatic, mild, or severe infection. By default, values of 0.98, 0.99, and 1.0 are used.
- The parameter ``strains`` contains information about any circulating strains that have been specified as additional to the default strain. This is initialized as an empty list and then populated by the user. 

Other parameter changes
^^^^^^^^^^^^^^^^^^^^^^^
- The parameter ``frac_susceptible`` will initialize the simulation with less than 100% of the population to be susceptible to COVID (to represent, for example, a baseline level of population immunity). Note that this is intended for quick explorations only, since people are selected at random, whereas in reality higher-risk people will typically be infected first and preferentially be immune. This is primarily designed for use with ``use_waning=False``.
- The parameter ``scaled_pop``, if supplied, can be used in place of ``pop_scale`` or ``pop_size``. For example, if you specify ``cv.Sim(pop_size=100e3, scaled_pop=550e3)``, it will automatically calculate ``pop_scale=5.5``.
- Aliases have been added for several parameters: ``pop_size`` can also be supplied as ``n_agents``, and ``pop_infected`` can also be supplied as ``init_infected``. This only applies when creating a sim; otherwise, the default names will be used for these parameters.

Changes to states and results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Several new states have been added, such as ``people.naive``, which stores whether or not a person has ever been exposed to COVID before.
- New results have been added to store information by strain, as well as population immunity levels. In addition to new entries in ``sim.results``, such as ``pop_nabs`` (population level neutralizing antibodies) and ``new_reinfections``, there is a new set of results ``sim.results.strain``: ``cum_infections_by_strain``, ``cum_infectious_by_strain``, ``new_infections_by_strain``, ``new_infectious_by_strain``, ``prevalence_by_strain``, ``incidence_by_strain``. 

New functions, methods and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The newly-added file ``immunity.py`` contains functions, methods, and classes related to calculating immunity. This includes the ``strain`` class (which uses lowercase convention like Covasim interventions, which are also technically classes).
- A new ``cv.vaccinate()`` intervention has been added. Compared to the previous ``vaccine`` intervention (now renamed ``cv.simple_vaccine()``), this new intervention allows vaccination to boost agents' immunity against infection, transmission, and progression.
- There is a new ``sim.people.make_nonnaive()`` method, as the opposite of ``sim.people.make_naive()``.
- New functions ``cv.iundefined()`` and ``cv.iundefinedi()`` have been added for completeness.
- A new function ``cv.demo()`` has been added as a shortcut to ``cv.Sim().run().plot()``.
- There are now additional shortcut plotting methods, including ``sim.plot('strain')`` and ``sim.plot('all')``.

Renamed functions and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``cv.vaccine()`` is now called ``cv.simple_vaccine()``.
- ``cv.get_sim_plots()`` is now called ``cv.get_default_plots()``; ``cv.get_scen_plots()`` is now ``cv.get_default_plots(kind='scen')``.
- ``sim.people.make_susceptible()`` is now called ``sim.people.make_naive()``.

Bugfixes
^^^^^^^^
- ``n_imports`` now scales correctly with population scale (previously they were unscaled).
- ``cv.ifalse()`` and related functions now work correctly with non-boolean arrays (previously they used the ``~`` operator instead of ``np.logical_not()``, which gave incorrect results for int or float arrays).
- Interventions and analyzers are now deep-copied when supplied to a sim; this means that the same ones can be created and then used in multiple sims. Scenarios also now deep-copy their inputs.

Regression information
^^^^^^^^^^^^^^^^^^^^^^
- As noted above, with ``cv.Sim(use_waning=False)`` (the default), results should be the same as Covasim 2.1.2, except for new results keys mentioned above (which will mostly be zeros, since they are only populated with immunity turned on).
- Scripts using ``cv.vaccine()`` should be updated to use ``cv.simple_vaccine()``.
- Scripts calling ``sim.people.make_susceptible()`` should now call ``sim.people.make_naive()``.
- *GitHub info*: PR `927 <https://github.com/amath-idm/covasim/pull/927>`__



~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 2.x (2.0.0 – 2.1.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 2.1.2 (2021-03-31)
--------------------------

- Interventions and analyzers now accept a function as an argument to ``days`` or e.g. ``start_day``. For example, instead of defining ``start_day=30``, you can define a function (with the intervention and the sim object as arguments) that calculates and returns a start day. This allows interventions to be dynamically triggered based on the state of the sim. See [Tutorial 5](https://docs.idmod.org/projects/covasim/en/latest/tutorials/t05.html) for a new section on how to use this feature.
- Added a ``finalize()`` method to interventions and analyzers, to replace the ``if sim.t == sim.npts-1:`` blocks in ``apply()`` that had been being used to finalize.
- Changed setup instructions from ``python setup.py develop`` to ``pip install -e .``, and unpinned ``line_profiler``.
- *Regression information*: If you have any scripts/workflows that have been using ``python setup.py develop``, please update them to ``pip install -e .``. Likewise, ``python setup.py develop`` is now ``pip install -e .[full]``.
- *GitHub info*: PR `897 <https://github.com/amath-idm/covasim/pull/897>`__


Version 2.1.1 (2021-03-29)
--------------------------

- **Duration updates:** All duration parameters have been updated from the literature. While most are similar to what they were before, there are some differences: in particular, durations of severe and critical disease (either to recovery or death) have increased; for example, duration from symptom onset to death has increased from 15.8±3.8 days to 18.8±7.2 days. 
- **Performance updates:** The innermost loop of Covasim, ``cv.compute_infections()``, has been refactored to make more efficient use of array indexing. The observed difference will depend on the nature of the simulation (e.g., network type, interventions), but runs may be up to 1.5x faster now.
- **Graphs:** People, contacts, and contacts layers now have a new method, ``to_graph()``, that will return a ``networkx`` graph (requires ``networkx`` to be installed, of course). For example, ``nx.draw(cv.Sim(pop_size=100).run().people.to_graph())`` will draw all connections between 100 default people. See ``cv.Sim.people.to_graph()`` for full documentation.
- A bug was fixed with ``cv.TransTree.animate()`` failing in some cases.
- ``cv.date_formatter()`` now takes ``interval``, ``start``, and ``end`` arguments.
- Temporarily pinned ``line_profiler`` to version 3.1 due to `this issue <https://github.com/pyutils/line_profiler/issues/49>`__.
- *Regression information*: Parameters can be restored by using the ``version`` argument when creating a sim. Specifically, the parameters for the following distributions (all lognormal) have been changed as follows::

    exp2inf:  μ =  4.6 →  4.5, σ = 4.8 → 1.5
    inf2sym:  μ =  1.0 →  1.1, σ = 0.9 → 0.9
    sev2crit: μ =  3.0 →  1.5, σ = 7.4 → 2.0
    sev2rec:  μ = 14.0 → 18.1, σ = 2.4 → 6.3
    crit2rec: μ = 14.0 → 18.1, σ = 2.4 → 6.3
    crit2die: μ =  6.2 → 10.7, σ = 1.7 → 4.8

- *GitHub info*: PR `887 <https://github.com/amath-idm/covasim/pull/887>`__


Version 2.1.0 (2021-03-23)
--------------------------

Highlights
^^^^^^^^^^
- **Updated lognormal distributions**: Lognormal distributions had been inadvertently using the variance instead of the standard deviation as the second parameter, resulting in too small variance. This has been fixed. This has a small but nonzero impact on the results (e.g. with default parameters, the time to peak infections is about 5-10% sooner now).
- **Expanded plotting features**: You now have much more flexibility with passing arguments to ``sim.plot()`` and other plotting functions, such as to temporarily set global Matplotlib options (such as DPI), modify axis styles and limits, etc. For example, you can now do things like this: ``cv.Sim().run().plot(dpi=150, rotation=30, start_day='2020-03-01', end_day=55, interval=7)``.
- **Improved analyzers**: Transmission trees can be computed 20 times faster, Fit objects are more forgiving for data problems, and analyzers can now be exported to JSON.

Bugfixes
^^^^^^^^
- Previously, the lognormal distributions were unintentionally using the variance of the distribution, instead of the standard deviation, as the second parameter. This makes a small difference to the results (slightly higher transmission due to the increased variance). Old simulations that are loaded will automatically have their parameters updated so they give the same results; however, new simulations will now give slightly different results than they did previously. (Thanks to Ace Thompson for identifying this.)
- If a results object has low and high values, these are now exported to JSON (and also to Excel).
- MultiSim and Scenarios ``.run()`` methods now return themselves, as Sim does. This means that just as you can do ``sim.run().plot()``, you can also now do ``msim.run().plot()``.

Plotting and options
^^^^^^^^^^^^^^^^^^^^
- Standard plots now accept keyword arguments that will be passed around to all available subfunctions. For example, if you specify ``dpi=150``, Covasim knows that this is a Matplotlib setting and will configure it accordingly; likewise things like ``bottom`` (only for axes), ``frameon`` (only for legends), etc. If you pass an ambiguous keyword (e.g. ``alpha``, which is used for line and scatter plots), it will only be used for the *first* one.
- There is a new keyword argument, ``date_args``, that will format the x-axis: options include ``dateformat`` (e.g. ``%Y-%m-%d``), ``rotation`` (to avoid label collisions), and ``start_day`` and ``end_day``.
- Default plotting styles have updated, including less intrusive lines for interventions.

Other changes
^^^^^^^^^^^^^
- MultiSims now have ``to_json()`` and ``to_excel()`` methods, which are shortcuts for calling these methods on the base sim.
- If no label is supplied to an analyzer or intervention, it will use its class name (e.g. the default label for ``cv.change_beta`` is ``'change_beta'``).
- Analyzers now have a ``to_json()`` method.
- The ``cv.Fit`` and ``cv.TransTree`` classes now derive from ``Analyzer``, giving them some new methods and attributes.
- ``cv.sim.compute_fit()`` has a new keyword argument, ``die``, that will print warnings rather than raise exceptions if no matching data is found. Exceptions are now caught and helpful error messages are provided (e.g., if dates don't match).
- The algorithm for ``cv.TransTree`` has been rewritten, and now runs 20x as fast. The detailed transmission tree, in ``tt.detailed``, is now a pandas dataframe rather than a list of dictionaries. To restore something close to the previous version, use ``tt.detailed.to_dict('records')``.
- A data file with an integer rather than date "date" index can now be loaded; these will be counted relative to the simulation's start day.
- ``cv.load()`` has two new keyword arguments, ``update`` and ``verbose``, than are passed to ``cv.migrate()``.
- ``cv.options`` has new a ``get_default()`` method which returns the value of that parameter when Covasim was first loaded.

Documentation and testing
^^^^^^^^^^^^^^^^^^^^^^^^^
- An extra tutorial has been added on "Deployment", covering how to use it with `Dask <https://dask.org/>`__ and for using Covasim with interactive notebooks and websites. 
- Tutorials 7 and 10 have been updated so they work on Windows machines.
- Additional unit tests have been written to check the statistical properties of the sampling algorithms.

Regression information
^^^^^^^^^^^^^^^^^^^^^^
- To restore previous behavior for a simulation (i.e. using variance instead of standard deviation for lognormal distributions), call ``cv.misc.migrate_lognormal(sim)``. This is done automatically when loading a saved sim from disk. To undo a migration, type ``cv.misc.migrate_lognormal(sim, revert=True)``. What this function does is loop over the duration parameters and replace ``par2`` with its square root. If you have used lognormal distributions elsewhere, you will need to update them manually.
- Code that was designed to parse transmission trees will likely need to be revised. The object ``tt.detailed`` is now a dataframe; calling ``tt.detailed.to_dict('records')`` will bring it very close to what it used to be, with the exception that for a given row, ``'t'`` and ``'s'`` used to be nested dictionaries, whereas now they are prefixes. For example, whereas before the 45th person's source's "is quarantined" state would have been ``tt.detailed[45]['s']['is_quarantined']``, it is now ``tt.detailed.iloc[45]['src_is_quarantined']``.
- *GitHub info*: PR `859 <https://github.com/amath-idm/covasim/pull/859>`__


Version 2.0.4 (2021-03-19)
--------------------------
- Added a new analyzer, ``cv.daily_age_stats()``, which will compute statistics by age for each day of the simulation (compared to ``cv.age_histogram()``, which only looks at particular points in time).
- Added a new function, ``cv.date_formatter()``, which may be useful in quickly formatting axes using dates.
- Removed the need for ``self._store_args()`` in interventions; now custom interventions only need to implement ``super().__init__(**kwargs)`` rather than both.
- Changed how custom interventions print out by default (a short representation rather than the jsonified version used by built-in interventions).
- Added an ``update()`` method to ``Layer``, to allow greater flexibility for dynamic updating.
- *GitHub info*: PR `854 <https://github.com/amath-idm/covasim/pull/854>`__


Version 2.0.3 (2021-03-11)
--------------------------
- Previously, the way a sim was printed (e.g. ``print(sim)``) depended on what the global ``verbose`` parameter was set to (e.g. ``cv.options.set(verbose=0.1)``), which used ``sim.brief()`` if verbosity was 0, or ``sim.disp()`` otherwise. This has been changed to always use the ``sim.brief()`` representation regardless of verbosity. To restore the previous behavior, use ``sim.disp()`` instead of ``print(sim)``.
- ``sim.run()`` now returns a pointer to the sim object rather than either nothing (the current default) or the ``sim.results`` object. This means you can now do e.g. ``sim.run().plot()`` or ``sim.run().results`` rather than ``sim.run(do_plot=True)`` or ``sim.run(output=True)``.
- ``sim.get_interventions()`` and ``sim.get_analyzers()`` have been changed to return all interventions/analyzers if no arguments are supplied. Previously, they would return only the last intervention. To restore the previous behavior, call ``sim.get_intervention()`` or ``sim.get_analyzer()`` instead.
- The ``Fit`` object (and ``cv.compute_gof()``) have been updated to allow a custom goodness-of-fit estimator to be supplied.
- Two new results have been added, ``n_preinfectious`` and ``n_removed``, corresponding to the E and R compartments of the SEIR model, respectively.
- A new shortcut plotting option has been introduced, ``sim.plot(to_plot='seir')``.
- Plotting colors have been revised to have greater contrast.
- The ``numba_parallel`` option has been updated to include a "safe" option, which parallelizes as much as it can without disrupting the random number stream. For large sims (>100,000 people), this increases performance by about 10%. The previous ``numba_parallel=True`` option now corresponds to ``numba_parallel='full'``, which is about 20% faster but means results are non-reproducible. Note that for sims smaller than 100,000 people, Numba parallelization has almost no effect on performance.
- A new option has been added, ``numba_cache``, which controls whether or not Numba functions are cached. They are by default to save compilation time, but if you change Numba options (especially ``numba_parallel``), with caching you may also need to delete the ``__pycache__`` folder for changes to take effect.
- A frozen list of ``pip`` requirements, as well as test requirements, has been added to the ``tests`` folder.
- The testing suite has been revamped, with defensive code skipped, bringing code coverage to 90%.
- *Regression information*: Calls to ``sim.run(do_plot=True, **kwargs)`` should be changed to ``sim.run().plot(**kwargs)``. Calls to ``sim.get_interventions()``/``sim.get_analyzers()`` (with no arguments) should be changed to ``sim.get_intervention()``/``sim.get_analyzer()``. Calls to ``results = sim.run(output=True)`` should be replaced with ``results = sim.run().results``.
- *GitHub info*: PR `788 <https://github.com/amath-idm/covasim/pull/788>`__


Version 2.0.2 (2021-02-01)
--------------------------
- Added a new option to easily turn on/off interactive plotting: e.g., simply set ``cv.options.set(interactive=False)`` to turn off interactive plotting. This meta-option sets the other options ``show``, ``close``, and ``backend``.
- Changed the logic of ``do_show``, such that ``do_show=False`` will never show a plot, even if ``cv.options.show`` is ``True``.
- Added a new method, ``cv.diff_sims()``, that allows the differences in results between two simulations to be quickly calculated.
- Removed the ``keys`` argument from ``cv.daily_stats()``, since non-default keys are had to validate.
- Fixed a bug that prevented prognoses parameters from being correctly set to those from an earlier version.
- Added an R usage example to the ``examples`` folder (matching the one in the FAQ).
- Added additional tests, increasing test coverage from 72% to 88%.
- *GitHub info*: PR `779 <https://github.com/amath-idm/covasim/pull/779>`__


Version 2.0.1 (2021-01-31)
--------------------------
- Pinned ``xlrd`` version to 1.2.0 due to a bug introduced in the ``2.0.1`` version of ``xlrd`` (see `here <https://stackoverflow.com/questions/65250207/pandas-cannot-open-an-excel-xlsx-file>`__ for details).
- Fixed a bug that prevented a function from being supplied as ``subtarget`` for ``cv.test_prob()``.
- Fixed a bug that prevented regression parameters (e.g. ``cv.Sim(version='1.7.5')``) from working when Covasim was installed via ``pip``.
- Fixed typos in docstrings and tutorials.
- *GitHub info*: PR `775 <https://github.com/amath-idm/covasim/pull/775>`__


Version 2.0.0 (2020-12-05)
--------------------------

This version contains a number of major updates. Note: this version requires Sciris 1.0, so when upgrading to this version, you may also need to upgrade Sciris (``pip install sciris --upgrade``).

Highlights
^^^^^^^^^^
- **Parameters**: Default infection fatality ratio estimates have been updated in line with the latest literature.
- **Plotting**: Plotting defaults have been updated to support a wider range of systems, and users now have greater control over plotting and options.
- **New functions**: New methods have been added to display objects in different levels of detail; new methods have also been added for working with data, adding contacts, and analyzing multisims.
- **Webapp**: The webapp has been moved to a separate Python package, ``covasim_webapp`` (available `here <https://github.com/institutefordiseasemodeling/covasim_webapp>`__).
- **Documentation**: A comprehensive set of tutorials has been added, along with a glossary and FAQ; see https://docs.covasim.org or look in the ``docs/tutorials`` folder.

Parameter updates
^^^^^^^^^^^^^^^^^
- The infection fatality rate rate has been updated to use O'Driscoll et al. (https://www.nature.com/articles/s41586-020-2918-0). We also validated against other estimates, most notably Brazeau et al. (https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-34-ifr). The new estimates have similar estimates for older ages, but tend to be lower for younger ages, especially the 60–70 age category.
- While we have not made any updates to the hospitalization rate, viral load distribution, or infectious durations at this time, we are currently reviewing the literature on these parameters and may be making updates relatively soon.
- A new ``version`` option has been added to sims, to use an earlier version of parameters if desired. For example, you can use Covasim version 2.0 but with default parameters from the previous version (1.7.6) via ``sim = cv.Sim(version='1.7.6')``. If you wish to load and inspect parameters without making a sim, you can use e.g. ``cv.get_version_pars('1.7.6')``.
- A ``cv.migration()`` function has also been added. Covasim sims and multisims are "migrated" (updated to have the right structure) automatically if loading old versions. However, you may wish to call this function explicitly if you're migrating a custom saved object (e.g., a list of sims).

Plotting and options
^^^^^^^^^^^^^^^^^^^^
- Plotting defaults have been updated to work better on a wider variety of systems.
- Almost all plotting functions now take both ``fig`` and ``ax`` keywords, which let you pass in existing figures/axes to be used by the plot.
- A new ``options`` module has been added that lets the user specify plotting and run options; see ``cv.options.help()`` for a list of the options.
- Plot options that were previously set on a per-figure basis (e.g. font size, font family) are now set globally via the ``options`` module, e.g. ``cv.options.set(font_size=18)``.
- If plots are too small, you can increase the DPI (default 100), e.g. ``cv.options.set(dpi=200)``. If they are too large, you can decrease it, e.g. ``cv.options.set(dpi=50)``.
- In addition, you can also change whether Covasim uses 32-bit or 64-bit arithmetic. To use 64-bit (which is about 20% slower and uses about 40% more memory), use ``cv.options.set(precision=64)``.
- Options can also now be set via environment variables. For example, you can set ``COVASIM_DPI`` to change the default DPI, and ``COVASIM_VERBOSE`` to set the default verbosity. For example, ``export COVASIM_VERBOSE=0`` is equivalent to ``cv.options.set(verbose=0)``. See ``cv.options.help()`` for the full list.
- The built-in intervention plotting method was renamed from ``plot()`` to ``plot_intervention()``, allowing the user to define custom plotting functions that do something different.

Webapp
^^^^^^
- The webapp has been moved to a separate repository and ``pip`` package, in order to improve installation and load times of Covasim.
- The ``docker`` and ``.platform`` folders have been moved to ``covasim_webapp``.
- Since web dependencies are no longer included, installing and importing Covasim both take half as much time as they did previously.

Bugfixes
^^^^^^^^
- The ``quar_period`` argument is now correctly passed to the ``cv.contact_tracing()`` intervention. (Thanks to Scott McCrae for finding this bug.)
- If the user supplies an incorrect type to ``cv.Layer.find_contacts()``, this is now caught and corrected. (Thanks to user sba5827 for finding this bug.)
- Non-string ``Layer`` keys no longer raise an exception.
- The ``sim.compute_r_eff()`` error message now gives correct instructions (contributed by `Andrea Cattaneo <https://github.com/InstituteforDiseaseModeling/covasim/pull/295>`__).
- Parallelization in Jupyter notebooks (e.g. ``msim.run()``) should now work without crashing.
- If parallelization (e.g. ``msim.run()``) is called outside a ``main`` block on Windows, this leads to a cryptic error. This error is now caught more elegantly.
- Interventions now print out with their actual name (previously they all printed out as ``InterventionDict``).
- The keyword argument ``test_sensitivity`` for ``cv.test_prob()`` has been renamed ``sensitivity``, for consistency with ``cv.test_num()``.

New functions and methods
^^^^^^^^^^^^^^^^^^^^^^^^^
- Sims, multisims, scenarios, and people objects now have ``disp()``, ``summarize()``, and ``brief()`` methods, which display full detail, moderate detail, and very little detail about each. If ``cv.options.verbose`` is 0, then ``brief()`` will be used to display objects; otherwise, ``disp()`` will be used.
- Two new functions have been added, ``sim.get_intervention()`` and ``sim.get_analyzer()``. These act very similarly to e.g. ``sim.get_interventions()``, except they return the last matching intervention/analyzer, rather than returning a list of interventions/analyzers.
- MultiSims now have a ``shrink()`` method, which shrinks both the base sim and the other sims they contain.
- MultiSims also provide options to compute statistics using either the mean or the median; this can be done via the ``msim.reduce(use_mean=True)`` method. Two convenience methods, ``msim.mean()`` and ``msim.median()``, have also been added as shortcuts.
- Scenarios now have a ``scens.compare()`` method, which (like the multisim equivalent) creates a dataframe comparing results across scenarios.
- Contacts now have new methods for handling layers, ``sim.people.contacts.add_layer()`` and ``sim.people.contacts.pop_layer()``. Additional validation on layers is also performed.
- There is a new function, ``cv.data.show_locations()``, that lists locations for which demographic data are available. You can also now edit the data dictionaries directly, by modifying e.g. ``cv.data.country_age_data.data`` (suggested by `Andrea Cattaneo <https://github.com/InstituteforDiseaseModeling/covasim/issues/273>`__).

Other changes
^^^^^^^^^^^^^
- There is a new verbose option for sims: ``cv.Sim(verbose='brief').run()`` will print a single line of output when the sim finishes (namely, ``sim.brief()``).
- The argument ``n_cpus`` can now be supplied directly to ``cv.multirun()`` and ``msim.run()``.
- The types ``cv.default_float`` and ``cv.default_int`` are now available at the top level (previously they had to be accessed by e.g. ``cv.defaults.default_float``).
- Transmission trees now contain additional output; after ``tt = sim.make_transtree()``, a dataframe of key results is contained in ``tt.df``.
- The default number of seed infections has been changed from 10 to 20 for greater numerical stability. (Note that this placeholder value should be overridden for all actual applications.) 
- ``sim.run()`` no longer returns the results object by default (if you want it, set ``output=True``).
- A migrations module has been added (in ``misc.py``). Objects are  now automatically migrated to the current version of Covasim whene loaded The function ``cv.migrate()`` can also be called explicitly on objects if needed.

Documentation
^^^^^^^^^^^^^
- A glossary, FAQ, and tutorials have been added. All are available from https://docs.covasim.org.

Regression information
^^^^^^^^^^^^^^^^^^^^^^
- To restore previous default parameters for simulations, use e.g. ``sim = cv.Sim(version='1.7.6')``. Note that this does not affect saved sims (which store their own parameters).
- Any scripts that specify the ``test_sensitivity`` keyword for the ``test_prob`` intervention will need to rename that variable to ``sensitivity``.
- Any scripts that used ``results = sim.run()`` will need to be updated to ``results = sim.run(output=True)``.
- Any scripts that passed formatting options directly to plots should set these as options instead; e.g. ``sim.plot(font_size=18)`` should now be ``cv.options.set(font_size=18); sim.plot()``.
- Any custom interventions that defined a custom ``plot()`` method should use ``plot_interventions()`` instead.
- *GitHub info*: PRs `738 <https://github.com/amath-idm/covasim/pull/738>`__, `740 <https://github.com/amath-idm/covasim/pull/740>`__



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.7.x (1.7.0 – 1.7.6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.7.6 (2020-10-23)
--------------------------
- Added additional flexibility to ``cv.People``, ``cv.make_people()``, and ``cv.make_synthpop()`` to allow easier modification of different types of people (e.g. the raw output of SynthPops, the popdict, and the ``People`` object).
- *GitHub info*: PR `712 <https://github.com/amath-idm/covasim/pull/712>`__


Version 1.7.5 (2020-10-13)
--------------------------
- Added extra convenience methods to ``Layer`` objects:
   - ``Layer.members`` returns an array of all people with interactions in the layer
   - ``__contains__`` is implemented so ``uid in layer`` can be used
- ``cv.sequence.apply()`` passes on the underlying intervention's return value rather than always returning ``None``
- *GitHub info*: PR `709 <https://github.com/amath-idm/covasim/pull/709>`__


Version 1.7.4 (2020-10-02)
--------------------------
- Refactored `cv.contact_tracing()` so that derived classes can extend individual parts of contact tracing without having to re-implement the entire intervention
- Moved `people.trace` to `contact_tracing` so that the tracing step can be extended via custom interventions
- *Regression information*: Custom interventions calling `people.trace` should inherit from `cv.contact_tracing` instead and use `contact_tracing.identify_contacts` and `contact_tracing.notify_contacts` to replace `people.trace`. In most cases however, it would be possible to overload one of the contact tracing steps rather than `contact_tracing.apply`, which thus eliminates the need to call `people.trace` entirely.
- *GitHub info*: PR `702 <https://github.com/amath-idm/covasim/pull/702>`__


Version 1.7.3 (2020-09-30)
--------------------------
- Changed ``test_prob.apply()`` and ``test_num.apply()`` to return the indices of people that were tested
- ``cvm.date(None)`` returns ``None`` instead of an empty list. Both ``cvm.date()`` and ``cvm.day()`` no longer raise errors if the list of inputs includes ``None`` entries.
- *GitHub info*: PR `699 <https://github.com/amath-idm/covasim/pull/699>`__


Version 1.7.2 (2020-09-24)
--------------------------
- Changed the intervention validation introduced in version 1.7.1 from an exception to a printed warning, to accommodate for custom-defined interventions.
- Docstrings were clarified to indicate that usage guidance is a recommendation, not a requirement.
- *GitHub info*: PR `693 <https://github.com/amath-idm/covasim/pull/693>`__


Version 1.7.1 (2020-09-23)
--------------------------
- Added two new methods, ``sim.get_interventions()`` and ``sim.get_analyzers()``, which return interventions or analyzers based on the index, label, or type.
- Added a new analyzer, ``cv.daily_stats()``, which can print out and plot detailed information about the state of the simulation on each day.
- MultiSims can now be run without parallelization; use ``msim.run(parallel=False)``. This can be useful for debugging, or for parallelizing across rather than within MultiSims (since ``multiprocessing`` calls cannot be nested).
- ``sim.people.not_defined()`` has been renamed ``sim.people.undefined()``, and ``sim.people.quarantine()`` has been renamed ``sim.people.schedule_quarantine()``, since it does not actually place people in quarantine.
- New helper functions have been added: ``cv.maximize()`` maximizes the current figure, and ``cv.get_rows_cols()`` converts a number (usually a number of plots) into the required number of rows and columns. Both will eventually be moved to Sciris.
- The transmission tree plot has been corrected to account for people who have left quarantine. The definition of "quarantine end" for the sake of testing (``quar_policy='end'`` for ``cv.test_num()`` and ``cv.test_prob()``) has also been shifted up by a day (since by ``date_end_quarantine``, people are no longer in quarantine by the end of the day, so tests were not being counted as happening in quarantine).
- Additional validation is done on intervention order to ensure that testing interventions are defined before tracing interventions.
- Code has been moved between ``sim.py``, ``people.py``, and ``base.py`` to better reflect the division between "the simulation" (the first two files) and "the housekeeping" (the last file).
- *Regression information*: Scripts that used ``quar_policy='end'`` may now provide stochastically different results. User scripts that explicitly call ``sim.people.not_defined()`` or ``sim.people.quarantine()`` should be updated to call ``sim.people.undefined()`` and ``sim.people.schedule_quarantine()`` instead.
- *GitHub info*: PR `690 <https://github.com/amath-idm/covasim/pull/690>`__


Version 1.7.0 (2020-09-20)
--------------------------
- The way in which ``test_num`` handles rescaling has changed, taking into account the non-modeled population. It now behaves more consistently throughout the dynamic rescaling period. In addition, it previously used sampling with replacement, whereas now it uses sampling without replacement. While this does not affect results in most cases, it can make a difference if certain subgroups (e.g. people with severe disease) have very high testing rates.
- Two new results have been added: ``n_alive`` (total number of people minus deaths) and ``rel_test_yield`` (the proportion of tests that are positive relative to a random sample from the population). In addition, the ``n_susceptible`` calculation has been updated for simulations with dynamic rescaling to reflect the number of people rather than the number of agents.
- There are additional options for the quarantine policy in the ``test_prob`` intervention. For example, you can now test people on entry and 5 days into quarantine by specifing ``quar_policy=[0,5]``.
- A new method ``cv.randround()`` has been introduced which will probabilistically round a float to an integer -- for example, 3.2 will be rounded up 20% of the time and rounded down 80% of the time. This is used to ensure accurate mean values for small numbers.
- ``cv.check_version()`` can now take a comparison, e.g. ``cv.check_version('>=1.7.0')``.
- A ``People`` object can now be created with a single number, representing the number of people. However, to be fully initialized, it still needs the other model parameters. This change lets the people and their connections be created first, and then inserted into a sim later.
- Additional checking is performed on interventions to ensure they are in the correct order (i.e., testing before tracing).
- The ``Result`` object used to have several scaling options, but now it simply has ``True`` (corresponding to the previous ``'dynamic'``) and ``False``. The ``static`` scaling option has been removed since it is no longer used by any result types.
- *Regression information*: sims that used ``test_num`` may now produce different results, given the changes for sample-without-replacement and dynamic rescaling. Previous behavior had the effect of artificially inflating the effectiveness of ``test_num`` before and during dynamic rescaling, since all tests were assigned to the modeled subpopulation. As a result, to get comparable results as before, test efficacy (loosely parameterized by ``symp_test``) should increase. Although there is not an exact relationship, to give an example, a simulation with ``symp_test=7`` and ``pop_scale=10`` previously may correspond to ``symp_test=25`` now. This change means that ``symp_test`` behaves consistently across the simulation period, so whereas previously this parameter may have needed to change over time, it should now be possible to use a single value (typically the last one used).
- *GitHub info*: PR `684 <https://github.com/amath-idm/covasim/pull/684>`__, head ``bfb9f66``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.6.x (1.6.0 – 1.6.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.6.1 (2020-09-13)
--------------------------
- Unpinned ``numba`` from version 0.48. Version 0.49 `changed the seed <https://numba.pydata.org/numba-doc/latest/release-notes.html#version-0-49-0-apr-16-2020>`__ used for ``np.random.choice()``, meaning that results from versions >=0.49 will differ from versions <=0.48. Version 0.49 was also significantly slower for some operations, which is why the switch was not made at the time, but this no longer appears to impact Covasim.
- ``People.person()`` now populates the contacts dictionary when returning a person, so that e.g. ``sim.people[0].contacts`` is no longer ``None``.
- There is a new ``story()`` method for ``People`` that prints a history of an individual person, e.g. ``sim.people.story(35)``.
- The baseline test in ``test_baseline.py`` has been updated to include contact tracing, giving greater code coverage for regression changes.
- *Regression information*: No changes to the Covasim codebase were made; however, new installations of Covasim (or if you update Numba manually) will have a different random number stream. To return previous results, use the previous version of Numba: ``pip install numba==0.48.0``.
- *GitHub info*: PRs `669 <https://github.com/amath-idm/covasim/pull/669>`__, `677 <https://github.com/amath-idm/covasim/pull/677>`__, head ``756e8eab``


Version 1.6.0 (2020-09-08)
--------------------------
- There is a new ``cv.vaccine()`` intervention, which can be used to implement vaccination for subgroups of people. Vaccination can affect susceptibility, symptomaticity, or both. Multiple doses (optionally with diminishing efficacy) can be delivered.
- ``cv.Layer`` objects have a new highly optimized ``find_contacts()`` method, which reduces time required for the contact tracing by a factor of roughly 2. This method can also be used directly to find the matching contacts for a set of indices, e.g. ``sim.people.contacts['h'].find_contacts([12, 144, 2048])`` will find all contacts of the three people listed.
- The method ``sim.compute_summary()`` has been removed; ``sim.summarize()`` now serves both purposes. This function previously always took the last time point in the results arrays, but now can take any time point.
- A new ``reset`` keyword has been added to ``sim.initialize()``, which will overwrite ``sim.people`` even if it already exists. Similarly, both interventions and analyzers are preserved after a sim run, unless ``sim.initialize()`` is called again (previously, analyzers were preserved but interventions were reset). This is to support storing data in interventions, as used by ``cv.vaccine()``.
- ``sim.date()`` can now handle strings or date objects (previously, it could only handle integers).
- Data files in formats ``.json`` and ``.xls`` can now be loaded, in addition to the ``.csv`` and ``.xlsx`` formats supported previously.
- Additional flexibility has been added to plotting, including user-specified colors for data; custom sim labels; and reusing existing axes for plots.
- Metadata now saves correctly to PDF and SVG images via ``cv.savefig()``. An issue with ``cv.check_save_version()`` using the wrong calling frame was also fixed.
- The field ``date_exposed`` has been added to transmission trees.
- The result "Effective reproductive number" has been renamed "Effective reproduction number".
- Analyzers now have additional validation to avoid out-of-bounds dates, as well as additional test coverage.
- *Regression information*: No major backwards incompatibilities are introduced by this version. Instances of ``sim.compute_summary()`` should be replaced by ``sim.summarize()``, and results dependent on the original state of an intervention post-simulation should use ``sim._orig_pars['interventions']`` (or perform ``sim.initialize()`` prior to using them) instead of ``sim['interventions']``.
- *GitHub info*: PR `664 <https://github.com/amath-idm/covasim/pull/664>`__, head ``e902cdff``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.5.x (1.5.0 – 1.5.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.5.3 (2020-09-01)
--------------------------

- An ``AlreadyRunError`` is now raised if ``sim.run()`` is called in such a way that no timesteps will be taken. This error is a distinct type so that it can be safely caught and ignored if required, but it is anticipated that most of the time, calling ``run()`` and not taking any timesteps, would be an inadvertent error.
- If the simulation has reached the end, ``sim.run()`` (and ``sim.step()``) will now raise an ``AlreadyRunError``.
- ``sim.run()`` now only validates parameters as part of initialization. Parameters will always be validated in the normal workflow where ``sim.initialize()`` is called via ``sim.run()``. However, the use case for modifying parameters during a split run or otherwise modifying parameters after initialization suggests that the user should have maximum control over the parameters at this point, so in this specialist workflow, the user is responsible for setting the parameter values correctly and in return, ``sim.run()`` is guaranteed not to change them.
- Added a ``sim.complete`` attribute, which is ``True`` if all timesteps have been executed. This is independent of finalizing results, since if ``sim.step()`` is being called externally, then finalizing the results may happen separately.
- *GitHub info*: : PR `654 <https://github.com/amath-idm/covasim/pull/654>`__, head ``d84b5f97``


Version 1.5.2 (2020-08-18)
--------------------------

- Modify ``cv.People.quarantine()`` to allow it schedule future quarantines, and allow quarantines of varying duration.
- Update the quarantine pipeline so that ``date_known_contact`` is not removed when someone goes into quarantine.
- Fixed bug where people identified as known contacts while on quarantine would be re-quarantined at the end of their quarantine for the entire quarantine duration. Now if a quarantine is requested while someone is already on quarantine, their existing quarantine will be correctly extended where required. For example, if someone is quarantined for 14 days on day 0 so they are scheduled to leave quarantine on day 14, and they are then subsequently identified as a known contact of a separate person on day 6 requiring 14 days quarantine, in previous versions of Covasim they would be released from quarantine on day 15, and then immediately quarantined on day 16 until day 30. With this update, their original quarantine would now be extended, so they would be released from quarantine on day 20.
- Quarantine duration via ``cv.People.trace()`` is now based on time since tracing, not time since notification, as people are typically instructed to isolate for a period after their last contact with the confirmed case, whenever that was. This results in an overall decrease in time spent in quarantine when the ``trace_time`` is greater than 0.
- *Regression information*:
    - Scripts that called ``cv.People.quarantine()`` directly would have also had to manually update ``sim.results['new_quarantined']``. This is no longer required, and those commands should now be removed as they will otherwise be double counted
    - Results are expected to differ slightly because the handling of quarantines being extended has been improved, and because quarantine duration is now reduced by the ``trace_time``.
- *GitHub info*: PR `624 <https://github.com/amath-idm/covasim/pull/624>`__, head ``9041157f``


Version 1.5.1 (2020-08-17)
--------------------------
- Modify ``cv.BasePeople.__getitem__()`` to retrieve a person if the item is an integer, so that ``sim.people[5]`` will return a ``cv.Person`` instance
- Modify ``cv.BasePeople.__iter__`` so that iterating over people e.g. ``for person in sim.people:`` iterates over ``cv.Person`` instances
- *Regression information*: To restore previous behavior of ``for idx in sim.people:`` use ``for idx in range(len(sim.people)):`` instead
- *GitHub info*: PR `623 <https://github.com/amath-idm/covasim/pull/623>`__, head ``aaa4d7c1``


Version 1.5.0 (2020-07-01)
--------------------------
- Based on calibrations to Seattle-King County data, default parameter values have been updated to have higher dispersion and smaller differences between layers.
- Keywords for computing goodness-of-fit (e.g. ``use_frac``) can now be passed to the ``Fit()`` object.
- The overview plot (``to_plot='overview'``) has been updated with more plots.
- Subtargeting of testing interventions is now more flexible: values can now be specified per person.
- Issues with specifying DPI and for saving calling function information via ``cv.savefig()`` have been addressed.
- Several minor plotting bugs were fixed.
- A new function, ``cv.undefined()``, can be used to find indices for which a quantity is *not* defined (e.g., ``cv.undefined(sim.people.date_diagnosed)`` returns the indices of everyone who has never been diagnosed).
- *Regression information*: To restore previous behavior, use the following parameter changes::

    pars['beta_dist'] = {'dist':'lognormal','par1':0.84, 'par2':0.3}
    pars['beta_layer'] = dict(h=7.0, s=0.7, w=0.7, c=0.14)
    pars['iso_factor']  = dict(h=0.3, s=0.0, w=0.0, c=0.1)
    pars['quar_factor'] = dict(h=0.8, s=0.0, w=0.0, c=0.3)

- *GitHub info*: PR `596 <https://github.com/amath-idm/covasim/pull/596>`__, head ``775cf358``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.4.x (1.4.0 – 1.4.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.4.8 (2020-06-11)
--------------------------
- Prerelease version of 1.5.0, including the layer and beta distribution changes.
- *GitHub info*: head ``2cb21846``


Version 1.4.7 (2020-06-02)
--------------------------
- Added ``quar_policy`` argument to ``cv.test_num()`` and ``cv.test_prob()``; by default, people are only tested upon entering quarantine (``'start'``); other options are to test people as they leave quarantine, both as they enter and leave, and every day they are in quarantine (which was the previous default behavior).
- Requirements have been tidied up; ``python setup.py develop nowebapp`` now only installs minimal packages. In a future version, this may become the default.
- Fixed intervention export and import from JSON.
- *Regression information*: To restore previous behavior (not recommended) with using contact tracing, add ``quar_policy='daily'`` to ``cv.test_num()`` and ``cv.test_prob()`` interventions.
- *GitHub info*: PR `593 <https://github.com/amath-idm/covasim/pull/593>`__, head ``4d8016fa``


Version 1.4.6 (2020-06-01)
--------------------------
- Implemented continuous rescaling: dynamic rescaling can now be used with an arbitrarily small ``rescale_factor``. The amount of rescaling on a given timestep is now either ``rescale_factor`` or the factor that would be required to bring the population below the threshold, whichever is larger.
- *Regression information*: Results should not be affected unless a simulation was run with too small of a rescaling factor. This change corrects this issue.
- *GitHub info*: PR `588 <https://github.com/amath-idm/covasim/pull/588>`__, head ``f7ef0fa5``


Version 1.4.5 (2020-05-31)
--------------------------
- Added ``cv.date_range()``.
- Changed ``cv.day()`` and ``cv.date()`` to assume a start day of 2020-01-01 if not supplied.
- Added the option to add custom data to a ``Fit`` object, e.g. age histogram data.
- *GitHub info*: PR `585 <https://github.com/amath-idm/covasim/pull/585>`__, head ``4cabddc3``


Version 1.4.4 (2020-05-31)
--------------------------
- Improved transmission tree histogram plotting, including allowing start and end days, and renamed ``plot_histograms()``.
- Added functions for negative binomial distributions, allowing easier exploration of overdispersion effects: see ``cv.make_random_contacts()``, and, most importantly, ``pars['beta_dist']``.
- Renamed ``cv.multinomial()`` to ``cv.n_multinomial()``.
- Added a ``build_docs`` script.
- *GitHub info*: PR `582 <https://github.com/amath-idm/covasim/pull/582>`__, head ``8bb8b82e``


Version 1.4.3 (2020-05-30)
--------------------------
- Added ``swab_delay`` to ``cv.test_prob()``, which behaves the same way as for ``cv.test_num()`` (to set the delay between experiencing symptoms and receiving a test).
- Allowed weights for a ``Fit`` to be specified as a time series.
- *GitHub info*: PR `573 <https://github.com/amath-idm/covasim/pull/573>`__, head ``d84ffeff``


Version 1.4.2 (2020-05-30)
--------------------------
- Renamed ``cv.check_save_info()`` to ``cv.check_save_version()``, and allowed the ``die`` argument to be passed.
- Allowed ``verbose`` to be a float instead of an int; if between 0 and 1, during a model run, it will print out once every ``1/verbose`` days, e.g. ``verbose = 0.2`` will print an update once every 5 days.
- Updated the default number of household contacts from 2.7 to 2.0 for ``hybrid``, and changed ``cv.poisson()`` to no longer cast to an integer. These two changes cancel out, so default behavior has not changed.
- Updated the calculation of contacts from household sizes (now uses household size - 1, to remove self-connections).
- Added ``cv.MultiSim.load()``.
- Added Numba caching to ``compute_viral_load()``, reducing overall Covasim load time by roughly 50%.
- Added an option for parallel execution of Numba functions (see ``utils.py``); although this significantly improves performance (20-30%), it results in non-deterministic results, so is disabled by default.
- Changed ``People`` to use its own contact layer keys rather than those taken from the parameters.
- Improved plotting and corrected minor bugs in age histogram and model fit analyzers.
- *Regression information*:

  - Replace ``cv.check_save_info()`` with ``cv.check_save_version()``.
  - If you used a non-integer number of contacts, round down to the nearest integer (e.g., change 2.7 to 2.0).
  - If you loaded a household size distribution (e.g. ``cv.Sim(location='nigeria')``), add one to the number of household contacts (but then round down).

- *GitHub info*: PR `577 <https://github.com/amath-idm/covasim/pull/577>`__, head ``5569b88a``


Version 1.4.1 (2020-05-29)
--------------------------
- Added ``sim.people.plot()``, which shows the age distribution, and distribution of contacts by age and layer.
- Added ``sim.make_age_histogram()``, as well as the ability to call ``cv.age_histogram(sim)``, as an alternative to adding these as analyzers to a sim.
- Updated ``cv.make_synthpop()`` to pass a random seed to SynthPops (note: requires SynthPops version 0.7.1 or later).
- ``cv.set_seed()`` now also resets ``random.seed()``, to ensure reproducibility among functions that use this (e.g., NetworkX).
- Corrected ``sim.run()`` so ``sim.t`` is left at the last timestep (instead of one more).
- *GitHub info*: PR `574 <https://github.com/amath-idm/covasim/pull/574>`__, head ``a828d29b``


Version 1.4.0 (2020-05-28)
--------------------------

This version contains a large number of changes, including two new classes, ``Analyzer`` and ``Fit``, for performing simulation analyses and fitting the model to data, respectively. These changes are described below.

Analysis
^^^^^^^^
- Added a new class, ``Analyzer``, to perform analyses on a simulation.
- Added a new parameter, ``sim['analyzers']``, that operates like ``interventions``: it accepts a list of functions or ``Analyzer`` objects.
- Added two analyzers: ``cv.age_hist`` records age histograms of infections, diagnoses, and deaths; ``cv.snapshot`` makes copies of the ``People`` object at specified points in time.


Fitting
^^^^^^^
- Added a new class, ``cv.Fit()``, that stores information about the fit between the model and the data. "Likelihood" is no longer automatically calculated, but instead "mismatch" can be calculated via ``fit = sim.compute_fit()``.
- The Poisson test that was previously used for the "likelihood" calculation has been deprecated; the new default mismatch is based on normalized absolute error.
- For a plot of how the mismatch is being calculated, use ``fit.plot()``.

MultiSims
^^^^^^^^^
- Added ``multisim.init_sims()``, which is not usually necessary, but can be helpful if you want to create the ``Sim`` objects without running them straight away.
- Added ``multisim.split()``, easily allowing a merged multisim to be split back into its constituent parts (non-merged multisims can also be split). This can be used for example to create several multisims, merge them together, run them all at the same time in parallel, and then split the back for analysis.

Display functions
^^^^^^^^^^^^^^^^^
- Added ``sim.summarize()``, which shows a short review of key sim results (cumulative counts).
- Added ``sim.brief()``, which shows a one-line summary of the sim.
- Added ``multisim.summarize()``, which prints a brief summary of all the constituent sims.

Parameter changes
^^^^^^^^^^^^^^^^^
- Removed the parameter ``interv_func``; instead, intervention functions can now be appended to ``sim['interventions']``.
- Changed the default for the ``rescale`` parameter from ``False`` to ``True``. To return to previous behavior, define ``sim['rescale'] = False`` explicitly.

Other changes
^^^^^^^^^^^^^
- Added ``cv.day()`` convenience function to convert a date to an integer number of days (similar to ``cv.daydiff()``); also modified ``cv.date()`` to be able to handle input more flexibly. While ``sim.day()`` and ``sim.date()`` are still the recommended functions, the same functionality is now also available without a ``Sim`` object available.
- Allowed `cv.load_data()`` to accept non-time-series inputs.
- Added cumulative diagnoses to default plots.
- Moved ``sweeps`` (Weights & Biases) to ``examples/wandb``.
- Refactored cruise ship example to work again.
- Various bugfixes (e.g. to plotting arguments, data scrapers, etc.).
- *Regression information*: To migrate an old parameter set ``pars`` to this version and to restore previous behavior, use:

.. code-block:: python

    pars['analyzers'] = None # Add the new parameter key
    interv_func = pars.pop('interv_func', None) # Remove the deprecated key
    if interv_func:
        pars['interventions'] = interv_func # If no interventions
        pars['interventions'].append(interv_func) # If other interventions are present
    pars['rescale'] = pars.pop('rescale', False) # Change default to False

- *GitHub info*: PR `569 <https://github.com/amath-idm/covasim/pull/569>`__, head ``2dcf6ad8``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.3.x (1.3.0 – 1.3.5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.3.5 (2020-05-28)
--------------------------
- Added ``swab_delay`` argument to ``cv.test_num()``, allowing a distribution of times between when a person develops symptoms and when they go to be tested (i.e., receive a swab) to be specified.
- *GitHub info*: PR `566 <https://github.com/amath-idm/covasim/pull/566>`__, head ``19dcfdd7``


Version 1.3.4 (2020-05-26)
--------------------------
- Allowed data to be loaded from a dataframe instead of from file.
- Fixed data scrapers to use correct column labels.
- *GitHub info*: PR `568 <https://github.com/amath-idm/covasim/pull/568>`__, head ``8b157a26``


Version 1.3.3 (2020-05-26)
--------------------------
- Fixed issue with a loaded population being reloaded when a simulation is re-initialized.
- Fixed issue with the argument ``dateformat`` not being passed to the right plotting routine.
- Fixed issue with MultiSim plotting appearing in separate panels when run in a Jupyter notebook.
- Fixed issue with ``cv.git_info()`` failing to write to file when the calling function could not be found.
- *GitHub info*: PR `567 <https://github.com/amath-idm/covasim/pull/567>`__, head ``d1b2bc40``


Version 1.3.2 (2020-05-25)
--------------------------
- ``People`` and ``popdict`` objects can now be supplied directly to the sim instead of a file name.
- ``git_info()`` and ``check_save_info()`` now include information from the calling script (not just Covasim). They also now include a ``comments`` field to optionally store additional information.
- *GitHub info*: PR `562 <https://github.com/amath-idm/covasim/pull/562>`__, head ``a943bb9e``


Version 1.3.1 (2020-05-25)
--------------------------
- Modified calculation of ``R_eff`` to include a longer integration period at the beginning, and restored previous method of creating seed infections. 
- Updated default plots to include number of active infections, and removed recoveries.
- *GitHub info*: PR `561 <https://github.com/amath-idm/covasim/pull/561>`__, head ``6c91a32c``


Version 1.3.0 (2020-05-24)
--------------------------
- Changed the default number of work contacts in hybrid from 8 to 16, and halved beta from 1.4 to 0.7, to better capture superspreading events. *Regression information*: To restore previous behavior, set ``sim['beta_layer']['w'] = 0.14`` and ``sim['contacts']['w'] = 8``.
- Initial infections now occur at a distribution of dates instead of all at once; this fixes the artificial spike in ``R_eff`` that occurred at the very beginning of a simulation. *Regression information*: This change affects results, but was reverted in the next version (1.3.1).
- Changed the definition of age bins in prognoses to be lower limits rather than upper limits. Added an extra set of age bins for 90+.
- Changed population loading and saving to be based on People objects, not popdicts (syntax is exactly the same, although it is recommended to use ``.ppl`` instead of ``.pop`` for these files).
- Added additional random seed resets to population initialization and just before the run so that populations loaded from disk produce identical results to newly created ones. *Regression information*: This affects results by changing the random number stream. In most cases, previous behavior can typically be restored by setting ``sim.run(reset_seed=False)``.
- Added a new convenience method, ``cv.check_save_info()``, which can be put at the top of a script to check the Covasim version and automatically save the Git info to file.
- Added additional methods to ``People`` to retrieve different types of keys: e.g., ``sim.people.state_keys()`` returns all the different states a person can be in (e.g., ``symptomatic``).
- *GitHub info*: PR `557 <https://github.com/amath-idm/covasim/pull/557>`__, head ``32c5e1e3``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.2.x (1.2.0 – 1.2.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.2.3 (2020-05-23)
--------------------------
- Added ``cv.savefig()``, which is an alias to Matplotlib's ``savefig()`` function, but which saves additional metadata in the figure file. This metadata can be loaded with the new ``cv.get_png_metdata()`` function.
- Major changes to ``MultiSim`` plotting, incorporating all the flexibility of both simulation and scenario plotting. By default, with a small number of runs (<= 5), it defaults to scenario-style plotting; else, it defaults to simulation-style plotting.
- Default scenario plotting options were updated (e.g., showing deaths instead of hospitalizations).
- You may merge multiple multisims more merrily now, with e.g. ``msim = cv.MultiSim.merge(msim1, msim2)``.
- Test scripts (e.g. ``tests/run_tests``) have been updated to use ``pytest-parallel``, reducing wall-clock time by a factor of 5.
- *GitHub info*: PR `552 <https://github.com/amath-idm/covasim/pull/552>`__, head ``3c1ca8b3``


Version 1.2.2 (2020-05-22)
--------------------------
- Changed the syntax of ``cv.clip_edges()`` to match ``cv.change_beta()``. The old format of intervention ``cv.clip_edges(start_day=d1, end_day=d2, change=c)`` should now be written as ``cv.clip_edges(days=[d1, d2], changes=[c, 1.0])``.
- Changed the syntax for the transmission tree: it now takes the ``Sim`` object rather than the ``People`` object, and typical usage is now ``tt = sim.make_transtree()``.
- Plots now default to a maximum of 4 rows; this can be overridden using the ``n_cols`` argument, e.g. ``sim.plot(to_plot='overview', n_cols=2)``.
- Various bugs with ``MultiSim`` plotting were fixed.
- *GitHub info*: PR `551 <https://github.com/amath-idm/covasim/pull/551>`__, head ``28bf02b5``


Version 1.2.1 (2020-05-21)
--------------------------
- Added influenza-like illness (ILI) symptoms to testing interventions. If nonzero, this reduces the effectiveness of symptomatic testing, because you cannot distinguish between people who are symptomatic with COVID and people with other ILI symptoms.
- Removed an unneeded ``copy()`` in ``single_run()`` because multiprocessing always produces copies of objects via the pickling process.
- *GitHub info*: PR `541 <https://github.com/amath-idm/covasim/pull/541>`__, head ``07009eb9``


Version 1.2.0 (2020-05-20)
--------------------------
- Since parameters can be modified during the run, previously, the sim could not be rerun with the guarantee that the results would be the same. ``sim.run()`` now has a ``restore_pars`` argument (default true), which makes a copy of the parameters just prior to the run to ensure reproducibility.
- In plotting, by default, data points are now slightly transparent and behind the lines to improve visibility of the model curve.
- Interventions now have a ``label`` attribute, which can be helpful for finding them if many are used, e.g. ``[interv if interv.label=='Close schools' for interv in sim['interventions']``. There is also a new method, ``intervention.disp()``, which prints out detailed information about an intervention object.
- Subtargeting of particular people in testing interventions can now be done via a function that gets called dynamically, avoiding the need to initialize the sim prior to creating the intervention.
- Layer keys are now stored inside the ``popdict``, for greater consistency handling loaded populations. Layer key handling has been simplified and made more robust.
- Loading and saving a population is now controlled by the ``Sim`` object, not by the ``sim.initialize()`` method. Instead of ``sim = cv.Sim(); sim.initialize(save_pop=True)``, you can now simply do ``sim = cv.Sim(save_pop=True``, and it will save when the sim is initialized.
- Added prevalence and incidence as results.
- Added ``sim.scaled_pop_size``, which is the population size (the number of agents) times the population scale factor. This corresponds to the "actual" population size being modeled.
- Removed the numerical artifact at the beginning and end of the ``R_eff`` calculation due to the smoothing kernel, and confirmed that the spike in ``R_eff`` often seen at the beginning is due to the way the seed infectious progress from exposed to infectious, and not from a bug.
- Added more flexibility to plotting, including a new ``show_args`` keyword, allowing particular aspects of plotting (e.g., the data or interventions) to be turned on or off.
- Moved the cruise ship code from the core folder into the examples folder.
- *GitHub info*: PR `538 <https://github.com/amath-idm/covasim/pull/538>`__, head ``9b2dbfba``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.1.x (1.1.0 – 1.1.7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.1.7 (2020-05-19)
--------------------------
- Diagnoses are now reported on the day the test was conducted, not the day the person gets their diagnosis. This is to better align with data (which is reported this way), and to avoid a bug in which test yield could be >100%. A new attribute, ``date_pos_test``, was added to the ``sim.people`` object in order to track the date on which a person is given the test which will (after ``test_delay`` days) come back positive.
- An "overview" plotting feature has been added for sims and scenarios: simply use ``sim.plot(to_plot='overview')`` to use. This plots almost all of the simulation outputs on one screen.
- It is now possible to set ``pop_type = None`` if you are supplying a custom population.
- Population creation functions (including the ``People`` class) have been tidied up with additional docstrings added.
- Duplication between pre- and post-step state checking has been removed.
- *GitHub info*: PR `537 <https://github.com/amath-idm/covasim/pull/537>`__, head ``451f4100``


Version 1.1.6 (2020-05-19)
--------------------------
- Created an ``analysis.py`` file to support different types of analysis.
- Moved ``transtree`` from ``sim.people`` into its own class: thus instead of ``sim.people.make_detailed_transtree()``, the new syntax is ``tt = cv.TransTree(sim.people)``.
- *GitHub info*: PR `531 <https://github.com/amath-idm/covasim/pull/531>`__, head ``2d55c380``


Version 1.1.5 (2020-05-18)
--------------------------
- Added extra flexibility for targeting interventions by index of a person, for example, by age.
- *GitHub info*: head ``fda4cc17``


Version 1.1.4 (2020-05-18)
--------------------------
- Added a new hospital bed capacity constraint and renamed health system capacity parameters. To migrate an older set of parameters to this version, set:

.. code-block:: python

    pars['no_icu_factor']  = pars.pop('OR_no_treat')
    pars['n_beds_icu']     = pars.pop('n_beds')
    pars['no_hosp_factor'] = 1.0
    pars['n_beds_hosp']    = None

- Removed the ``bed_capacity`` result.
- *GitHub info*: PR `510 <https://github.com/amath-idm/covasim/pull/510>`__, head ``81261f90``


Version 1.1.3 (2020-05-18)
--------------------------
- Improved the how "layer parameters" (e.g., ``beta_layer``) are initialized.
- Allowed arbitrary arguments to be passed to SynthPops via ``cv.make_synthpop``.
- *GitHub info*: head ``0f6d48c0``


Version 1.1.2 (2020-05-18)
--------------------------
- Added a new result, ``test_yield``, which is the number of diagnoses divided by the number of cases each day.
- Minor improvements to date handling and plotting.
- *GitHub info*: head ``6f2f0455``


Version 1.1.1 (2020-05-13)
--------------------------
- Refactored the contact tracing and quarantining functions, to fixed a bug (introduced in v1.1.0) in which some people who went into quarantine never came out of quarantine.
- Changed initialization so seed infections are now sampled randomly from the population, rather than the first ``pop_infected`` agents. Since ``hybrid`` also uses consecutive indices for constructing households, this was causing some households to be fully infected on initialization, while all other households had no infections.
- Updated the default ``rescale_factor`` from 2.0 to 1.2, since large amounts of rescaling cause noticeable "blips" in inhomogeneous networks (e.g., a population where some households are 100% infected and most are 0% infected).
- Added ability to pass plotting arguments to ``intervention.plot()``.
- Removed default noise in scenarios (restore previous behavior by setting ``metapars = dict(noise=0.1)``).
- Refactored and renamed computed results (e.g., summary stats) in the Sim class.
- *GitHub info*: PR `513 <https://github.com/amath-idm/covasim/pull/513>`__, head ``2332c319``


Version 1.1.0 (2020-05-12)
--------------------------
- Renamed the parameter ``diag_factor`` to ``iso_factor``, and converted it to a dictionary by layer.
- Renamed the parameter ``quar_eff`` to ``quar_factor`` (but otherwise left it unchanged).
- Added the option for presumptive isolation and quarantine in testing interventions.
- Fixed a bug whereby people who had been in quarantine and were then diagnosed had both diagnosis and quarantine factors applied.
- *GitHub info*: PR `502 <https://github.com/amath-idm/covasim/pull/502>`__, head ``973801a6``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.0.x (1.0.0 – 1.0.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.0.3 (2020-05-11)
--------------------------
- Added an extra output of ``make_microstructured_contacts()`` to store each person's cluster identifier. Currently, this is only supported for the ``hybrid`` population type, but in future versions, ``synthpops`` will also be supported.
- Removed the ``directed`` argument from population creation functions since it is no longer supported in the model.
- *GitHub info*: head ``57f58480``


Version 1.0.2 (2020-05-10)
--------------------------
- Added uncertainty to the ``plot_result()`` method of MultiSims.
- Added documentation and webapp links to the paper.
- *GitHub info*: head ``6811bc59``


Version 1.0.1 (2020-05-09)
--------------------------
- Added argument ``as_date`` for ``sim.date()`` to return a ``datetime`` object instead of a string.
- Fixed plotting of interventions in the webapp.
- Removed default 1-hour time limit for simulations.
- *GitHub info*: PR `490 <https://github.com/amath-idm/covasim/pull/490>`__, head ``1e08cc9a``


Version 1.0.0 (2020-05-08)
--------------------------
- Official release of Covasim.
- Made scenario and simulation plotting more flexible: ``to_plot`` can now simply be a list of results keys, e.g. ``cum_deaths``.
- Added additional tests, increasing test coverage from 67% to 92%.
- Fixed bug in ``cv.save()``.
- Added ``reset()`` to MultiSim that undoes a ``reduce()`` or ``combine()`` call.
- General code cleaning: made exceptions raised more consistent, removed unused functions, etc.
- *GitHub info*: PR `487 <https://github.com/amath-idm/covasim/pull/487>`__, head ``9a6c23b``



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prerelease versions (0.27.0 – 0.32.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 0.32.1 (2020-05-06)
---------------------------
- Allow ``until`` to be a date, e.g. ``sim.run(until='2020-05-06')``.
- Added ``ipywidgets`` dependency since otherwise the webapp breaks due to a `bug <https://github.com/plotly/plotly.py/issues/2443>`__ with the latest Plotly version (4.7).
- *GitHub info*: head ``c8ca32d``


Version 0.32.0 (2020-05-05)
---------------------------
- Changed the edges of the contact network from being directed to undirected, halving the amount of memory required and making contact tracing and edge clipping more realistic.
- Added comorbidities to the prognoses parameters.
- *GitHub info*: PR `482 <https://github.com/amath-idm/covasim/pull/482>`__ 


Version 0.31.0 (2020-05-05)
---------------------------
- Added age-susceptible odds ratios, and modified severe and critical progression probabilities. To compensate, default ``beta`` has been increased from 0.015 to 0.016. To restore previous behavior (which was based on the `Imperial paper <https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf>`__), set ``beta=0.015`` and set the following values in ``sim.pars['prognoses']``::

    sus_ORs[:]   = 1.0
    severe_probs = np.array([0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400])
    crit_probs   = np.array([0.00004, 0.00011, 0.00050, 0.00123, 0.00214, 0.00800, 0.02750, 0.06000, 0.10333])

- Relative susceptibility and transmissibility (i.e., ``sim.people.rel_sus``) are now set when the population is initialized (before, they were modified dynamically when a person became infected or recovered). This means that modifying them before a simulation starts, or during a simulation, should be more robust.
- Reordered results dictionary to start with cumulative counts.
- ``sim.export_pars()`` now accepts a filename to save to.
- Added a ``tests/regression`` folder with previous versions of default parameter values.
- Changed ``pars['n_beds']`` to interpret 0 or ``None`` as no bed constraint.
- *GitHub info*: PR `480 <https://github.com/amath-idm/covasim/pull/480>`__, head ``029585f``, previous head ``c7171f8``


Version 0.30.4 (2020-05-04)
---------------------------
- Changed the detailed transmission tree (``sim.people.transtree.detailed``) to include much more information.
- Added animation method to transmission tree: ``sim.people.transtree.animate()``.
- Added support to generate populations on the fly in SynthPops.
- Adjusted the default arguments for ``test_prob`` and fixed a bug with ``test_num`` not accepting date input.
- Added ``tests/devtests/intervention_showcase.py``, using and comparing all available interventions.


Version 0.30.3 (2020-05-03)
---------------------------
- Fixed bugs in dynamic scaling; see ``tests/devtests/dev_test_rescaling.py``. When using ``pop_scale>1``, the recommendation is now to use ``rescale=True``.
- In ``cv.test_num()``, renamed argument from ``sympt_test`` to ``symp_test`` for consistency.
- Added ``plot_compare()`` method to ``MultiSim``.
- Added ``labels`` arguments to plotting methods, to allow custom labels to be used.


Version 0.30.2 (2020-05-02)
---------------------------
- Updated ``r_eff`` to use a new method based on daily new infections. The previous version, where infections were counted from when someone recovered or died, is available as ``sim.compute_r_eff(method='outcome')``, while the traditional method, where infections are counted from the day someone becomes infectious, is available via ``sim.compute_r_eff(method='infectious')``.


Version 0.30.1 (2020-05-02)
---------------------------
- Added ``end_day`` as a parameter, allowing an end date to be specified instead of a number of days.
- ``Sim.run()`` now displays the date being simulated.
- Added a ``par_args`` argument to ``multi_run()``, allowing arguments (e.g. ``ncpus``) to be passed to ``sc.parallelize()``.
- Added a ``compare()`` method to multisims and stopped people from being saved by default.
- Fixed bug whereby intervention were not getting initialized if they were added to a sim after it was initialized.


Version 0.30.0 (2020-05-02)
---------------------------
- Added new ``MultiSim`` class for plotting a single simulation with uncertainty.
- Added ``low`` and ``high`` attributes to the ``Result`` object.
- Refactored plotting to increase consistency between ``sim.plot()``, ``sim.plot_result()``, ``scens.plot()``, and ``multisim.plot()``.
- Doubling time calculation defaults have been updated to use a window of 3 days and a maximum of 30 days.
- Added an ``until`` argument to ``sim.run()``, to make it easier to run a partially completed sim and then resume. See ``tests/devtests/test_run_until.py``.
- Fixed a bug whereby ``cv.clip_edges()`` with no end day specified resulted in large sim files when saved.


Version 0.29.9 (2020-04-28)
---------------------------
- Fixed bug in which people who had been tested and since recovered were not being diagnosed.
- Updated definition of "Time to die" parameter in the webapp.


Version 0.29.8 (2020-04-28)
---------------------------
- Updated webapp UI with more detail on and control over interventions.


Version 0.29.7 (2020-04-27)
---------------------------
- New functions ``cv.date()`` and ``cv.daydiff()`` have been added, to ease handling of dates of different formats.
- Defaults are now functions rather than dictionaries, specifically: ``cv.default_sim_plots`` is now ``cv.get_sim_plots()``; ``cv.default_scen_plots`` is now ``cv.get_scen_plots()``; and ``cv.default_colors`` is now ``cv.get_colors()``.
- Interventions now have a ``do_plot`` kwarg, which if ``False`` will disable their plotting.
- The example scenario (``examples/run_scenario.py``) has been rewritten to include a test-trace-quarantine example.


Version 0.29.6 (2020-04-27)
---------------------------
- Updated to use Sciris v0.17.0, to fix JSON export issues and improve ``KeyError`` messages.


Version 0.29.5 (2020-04-26)
---------------------------
- Fixed bug whereby layer betas were applied twice, and updated default values.
- Includes individual-level viral load (to use previous results, set ``pars['beta_dist'] = {'dist':'lognormal','par1':1.0, 'par2':0.0}`` and ``pars['viral_dist']  = {'frac_time':0.0, 'load_ratio':1, 'high_cap':0}``).
- Updated parameter values (mostly durations) based on revised literature review.
- Added ``sim.export_pars()`` and ``sim.export_results()`` methods.
- Interventions can now be converted to/from JSON -- automatically when loading a parameters dictionary into a sim, or manually using ``cv.InterventionDict()``.
- Improvements to transmission trees: can now make a detailed tree with ``sim.people.make_detailed_transtree()`` (replacing ``sim.people.transtree.make_detailed(sim.people)``), and can plot via ``sim.people.transtree.plot()``.
- Improved date handling, so most functions are now agnostic as to whether a date string, datetime object, or number of days is provided; new functions: ``sim.day()`` converts dates to days, ``sim.date()`` (formerly ``sim.inds2dates()``) converts days to dates, and ``sim.daydiff()`` computes the number of days between two dates.


Version 0.28.8 (2020-04-24)
---------------------------
- Includes data on household sizes from various countries.
- Includes age data on US states.
- Changes to interventions to include end as well as start days, and plotting as a default option.
- Adds version checks to loading and introduces a new function ``cv.load()`` to replace e.g. ``cv.Sim.load()``.
- Major layout and functionality changes to the webapp, including country selection (disabled by default).
- Provided access to Plotly graphs via the backend.
- Moved relative probabilities (e.g. ``rel_death_prob``) from population creation to loop so can be modified dynamically.
- Introduced ``cv.clip_edges()`` intervention, similar to ``cv.change_beta()`` but removes contacts entirely.


Version 0.28.1 (2020-04-19)
---------------------------
- Major refactor of transmission trees, including additional detail via ``sim.people.transtree.make_detailed()``.
- Counting of diagnoses before and after interventions on each timestep (allowing people to go into quarantine on the same day).
- Improved saving of people in scenarios, and updated keyword for sims (``sim.save(keep_people=True)``).


Version 0.28.0 (2020-04-19)
---------------------------
- Includes dynamic per-person viral load.
- Refactored data types.
- Changed how populations are handled, including adding a ``dynam_layer`` parameter to specify which layers are dynamic.
- Disease progression duration parameters were updated to be longer.
- Fixed bugs with quarantine.
- Fixed bug with hybrid school and work contacts.
- Changed contact tracing to be only for contacts with nonzero transmission.


Version 0.27.12 (2020-04-17)
----------------------------
- Caches Numba functions, reducing load time from 2.5 to 0.5 seconds.
- Pins Numba to 0.48, which is 10x faster than 0.49.
- Fixed issue with saving populations in scenarios.
- Refactored how populations are handled, removing ``use_layers`` parameter (use ``pop_type`` instead).
- Removed layer key from layer object, reducing total sim memory footprint by 3x.
- Improved handling of mismatches between loaded population layers and simulation parameters.
- Added custom key errors to handle multiline error messages.
- Fix several issues with probability-based testing.
- Changed how layer betas are applied (inside the sim rather than statically).
- Added more detail to the transmission tree.
- Refactored random population calculation, speeding up large populations (>100k) by a factor of 10.
- Added `documentation <https://institutefordiseasemodeling.github.io/covasim-docs/>`__.


Version 0.27.0 (2020-04-16)
---------------------------
-  Refactor calculations to be vector-based rather than object based.
-  Include factors for per-person viral load (transmissibility) and
   susceptibility.
-  Started a changelog (needless to say).
