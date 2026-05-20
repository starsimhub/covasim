===
FAQ
===

This document contains answers to frequently (and some not so frequently) asked questions. If there are others you'd like to see included, please email us at info@covasim.org.

.. contents:: **Contents**
   :local:
   :depth: 2


Usage questions
^^^^^^^^^^^^^^^

What are the system requirements for Covasim?
---------------------------------------------------------------------------------

If your system can run scientific Python (Numpy, SciPy, and Matplotlib), then you can probably run Covasim. Covasim requires 1 GB of RAM per 1 million people, and can simulate roughly 5-10 million person-days per second. A typical use case, such as a population of 100,000 agents running for 500 days, would require 100 MB of memory and take about 5-10 seconds to run.


Can Covasim be run on HPC clusters?
---------------------------------------------------------------------------------

Yes. On a single-node setup, it is quite easy: in fact, ``MultiSim`` objects will automatically scale to the number of cores available. This can also be specified explicitly with e.g. ``msim.run(n_cpus=24)``.

For more complex use cases (e.g. running across multiple virtual machines), we recommend using `Celery <https://docs.celeryproject.org>`__; please `email us <mailto:info@covasim.org>`__ for more information.


What method is best for saving simulation objects?
---------------------------------------------------------------------------------

The recommended way to save a simulation is simply via ``sim.save(filename)``. By default, this does *not* save the people (``sim.people``), since they are very large (i.e., 7 KB without people vs. 7 MB with people for 100,000 agents). However, if you really want to save the people, pass ``keep_people=True``.

To load, you can use ``cv.load(filename)`` or ``cv.Sim.load(filename)`` (they are identical except for type checking). Under the hood, Covasim uses ``sc.saveobj()`` from `Sciris <http://sciris.org>`__, which in turn is a gzipped pickle. If you need a more "portable" format, you can use ``sim.to_json()``, but note that this will only save the results and the parameters, not the full sim object.


Does Covasim support time-dependent parameters?
---------------------------------------------------------------------------------

Typically, parameters are held constant for the duration of the simulation. However, it is possible to modify them dynamically – see the ``cv.dynamic_pars()`` "intervention".


How can you introduce new infections into a simulation?
---------------------------------------------------------------------------------

These are referred to as *importations*. You can set the ``n_imports`` parameter for a fixed number of importations each day (or make it time-varying with ``cv.dynamic_pars()``, as described above). Alternatively, you can infect people directly using ``sim.people.infect()``. Since version 3.0, you can also import specific strains on a given day: e.g., ``cv.Sim(strains=cv.strain('b117', days=50, n_imports=10)``.


How can I find out what happened to a particular individual during the simulation?
----------------------------------------------------------------------------------

This can be done using the ``story`` method after the simulation has been run:

.. code-block:: python

    import covasim as cv
    sim = cv.Sim()
    sim.people.story(12)



How do you set custom prognoses parameters (mortality rate, susceptibility etc.)?
---------------------------------------------------------------------------------

Most parameters can be set quite simply, e.g.:

.. code-block:: python

    import covasim as cv
    sim = cv.Sim(beta=0.008)

or:

.. code-block:: python

    import covasim as cv
    pars = dict(beta=0.008, verbose=0)
    sim = cv.Sim(pars)

However, prognoses parameters are a bit different since they're a dictionary of dictionaries of arrays. Usually the easiest solution is to create the simulation first, and then modify these parameters before initializing the sim:

.. code-block:: python

    import covasim as cv
    sim = cv.Sim()
    sim['prognoses']['death_probs'][-1] *= 2 # Double the risk of death in the oldest age group

Another option is to create the parameters first, then modify them and provide them to the sim:

.. code-block:: python

    import covasim as cv
    prognoses = cv.get_prognoses()
    prognoses['death_probs'][-1] *= 2
    sim = cv.Sim(prognoses=prognoses)

One thing to be careful of is that since the prognoses are used when the population properties are set, you must make any changes to them *before* you initialize the sim (i.e. ``sim.initialize()``). If you want to change prognoses for an already-created simulation, it is best to call ``sim.init_people()`` to ensure the sim parameters (``sim.pars``) are synchronized with the people parameters (``sim.people.pars``).


I want to generate a contact network for <insert location here>. How do I do this?
----------------------------------------------------------------------------------

There are a few options. For many cases, the default options work reasonably well, i.e. ``sim = cv.Sim(pop_type='hybrid', location='eswatini')``. If you want to use location that is not currently supported, there is generally a lot of data required (census data, school enrolment rates, workplace size and participation rates, etc.). Detailed contact networks are generally created using the `SynthPops <http://synthpops.org>`__ library.

Another option is to adapt the functions in ``population.py`` for your purposes. Covasim can also read in fairly generic representations of populations; for example you could create a random network and then modify the edge list (i.e. ``sim.people.contacts``) to reflect the network you want. Please `email us <mailto:info@covasim.org>`__ for more information.


Is it possible to model interacting geographical regions?
---------------------------------------------------------------------------------

Possible, but not easy. Your best option is to create a single simulation where the contact network structure reflects the different regions. Please `email us <mailto:info@covasim.org>`__ for more information.


Is it possible to model births, non-COVID deaths, or migration?
---------------------------------------------------------------------------------

Not currently, but what you can do is create the final population size, and then remove all contacts for the people you want to be "absent". In terms of *transmission*, a susceptible person who is unconnected to anyone else is effectively identical to a person who doesn't exist. You can then "create" these people by adding or restoring their contacts to other people. However, be mindful that results that are population averages (e.g. prevalence) will be wrong since they will include these inactive people, and interventions (such as ``cv.test_num()``) will also apply to these people.


I really don't like Python, can I run Covasim via R?
---------------------------------------------------------------------------------

Actually, you can! R's `reticulate <https://rstudio.github.io/reticulate/>`__ package lets you easily interface between Python and R. For example:

.. code-block:: S

    library(reticulate)
    cv <- import('covasim')
    sim <- cv$Sim()
    sim$run()
    sim$plot()

(NB: if the above doesn't bring up a figure, try adding ``plt <- import('matplotlib.pyplot')`` and ``plt$show()``.)



Conceptual questions
^^^^^^^^^^^^^^^^^^^^

What are the relationships between population size, number of agents, population scaling, and total population?
---------------------------------------------------------------------------------------------------------------

The terms are a bit confusing and may be refactored in a future version of Covasim. The ``pop_size`` parameter actually controls the number of *agents* in the simulation (note: you can also use the parameter ``n_agents``, it's just an alias for ``pop_size``). In many cases the number of agents is the same as the "total population size" or "scaled population size" being simulated, i.e., the actual number of people. The "actual number of people" (not agents) is available in the simulation as ``sim.scaled_pop_size``. If ``pop_scale`` is greater than 1, the total population size will be greater than the number of agents. You can also set the ``scaled_pop`` parameter (which is the total population size), in which case ``pop_scale`` will be calculated automatically. Some examples might help make this clearer:

*Example 1*. You want to simulate a population of 100,000 people. This will only take a few seconds to run, so you set ``pop_size = 100e3`` and ``pop_scale = 1``. In this example the population size is 100,000, the scaled population size is 100,000, the number of agents is 100,000, and the number of people being represented is also 100,000. Life is simple and you are happy.

*Example 2*. You want to simulate a population of 1,000,000 people. This would take too long to run easily (several minutes per run), so you set ``pop_size = 200e3`` and ``pop_scale = 5`` with dynamic rescaling on (``rescale = True``). (Note: this is exactly equivalent to setting ``n_agents = 200e3`` and ``scaled_pop = 1e6``, in which case ``pop_scale`` will be automaticlaly set to 5.) In this example the (simulated) population size is 200,000, the (final) scaled population size is 1,000,000, the number of agents is always 200,000, and the (final) number of people being represented is 1,000,000. Since dynamic rescaling is on, when the simulation starts, one agent represents one person, but only 200,000 people are included in the simulation (the other 800,000 are not infected and are not exposed to anyone who is infected, so are not represented in the sim). As more and more people become infected – say, 10,000 infections – 200,000 people is no longer enough to accurately represent the epidemic, since 10,000 infections out of 200,000 people is prevalence of 5%, whereas the real prevalence is 1% (10,000 infections out of 1,000,000 people). Dynamic rescaling kicks in (``rescale_threshold = 0.05``, the current prevalence level), and half of the infected people are converted back to susceptibles (``rescale_factor = 2``). There are now 5,000 infected *agents* in the model, corresponding to 10,000 infected *people*, i.e. one agent now counts as (represents) two people. This is equivalent to saying that for any given agent in the model (e.g., an infected 57-year-old woman who has 2 household contacts and 8 workplace contacts), there is another identical person somewhere else in the population.

*Example 3*. As in example 2, but you turn dynamic rescaling off (``rescale = False``). In this case, from the very beginning of the simulation, one agent represents 5 people (since ``pop_scale = 5``). This is basically the same as running a simulation of 200,000 agents with ``pop_scale = 1`` and then multiplying the results (e.g., cumulative number of infections) by a factor of 5 after the simulation finishes running: each infection counts as 5 infections, each death counts as 5 deaths, etc. Note that with dynamic rescaling off, the number of seed infections should be divided by ``pop_scale`` in order to give the same results

**TLDR?** Except for a few corner cases (e.g., calculating transmission trees), you should get nearly identical results with and without dynamic rescaling, so feel free to use it (it's turned on by default). That said, it's always best to use as small of a population scale factor as you can, although once you reach roughly 200,000 agents, using more agents shouldn't make much difference.

This example illustrates the three different ways to simulation a population of 100,000 people:

.. code-block:: python

    import covasim as cv

    s1 = cv.Sim(n_days=120, pop_size=200e3, pop_infected=50, pop_scale=1,  rescale=True,  label='Full population')
    s2 = cv.Sim(n_days=120, pop_size=20e3,  pop_infected=50, pop_scale=10, rescale=True,  label='Dynamic rescaling')
    s3 = cv.Sim(n_days=120, pop_size=20e3,  pop_infected=5,  pop_scale=10, rescale=False, label='Static rescaling')

    msim = cv.MultiSim([s1, s2, s3])
    msim.run(verbose=-1)
    msim.plot()

Note that using the full population and using dynamic rescaling give virtually identical results, whereas static scaling gives slightly different results.


Are test results counted from swab date or result date?
---------------------------------------------------------------------------------

The results are reported for the date of the test which came back positive, not the the date of diagnosis. This reason for this is that in most places, this is how the data are reported – if they do 100 tests on August 1st, say, and there is a 2-4 day test delay so 5 of these tests come back positive on each of August 2nd, 3rd, 4th, then in most places, this would be reported as 100 tests on August 1st, 15 diagnoses on August 1st (even though the lab work was done over August 2-4), and 85 negative tests on August 1st. The reason for doing it this way – both in real world reporting and in the model – is because otherwise you have a situation where if there is a big change in the number of tests from day to day, you could have more diagnoses on that day than tests. However, in terms of the model, the test delay is still being correctly taken into account. Specifically, ``sim.people.date_pos_test`` is used to (temporarily) store the date of the positive test, which is what's shown in the plots, but sim.people.date_diagnosed has the correct (true) diagnosis date for each person. 
For example:

.. code-block:: python

    import covasim as cv
    tn = cv.test_num(daily_tests=100, start_day=10, test_delay=10)
    sim = cv.Sim(interventions=tn)
    sim.run()
    sim.plot(to_plot=['new_infections', 'new_tests', 'new_diagnoses'])

shows that positive tests start coming back on day 10 (the start day of the intervention), but:

.. code-block:: python

    >>> np.nanmin(sim.people.date_diagnosed)
    20.0

shows that the earliest date a person is actually diagnosed is on day 20 (the start day of the intervention plus the test delay).


Is the underlying model capable of generating oscillations?
---------------------------------------------------------------------------------

Yes, although oscillatory modes are not a natural state of the system – you can get them with a combination of high infection rates, low testing rates, and high contact tracing rates with significant delays. This will create little clusters that grow stochastically until someone gets tested, then most of the cluster gets traced and shut down, but a few people usually escape to start the next cluster.


Why doesn't anyone start off as infectious?
-------------------------------------------
If you run a simple simulation (e.g. ``sim = cv.Sim().run()``), you might notice that no one starts off as infectious (``sim.results.n_infectious[0]`` is 0). This is because they all start in the just-infected or "exposed" state, when they're not infectious yet (it takes a few days for someone to become infectious, determined by the parameter ``exp2inf``, exposed to infectious duration). This corresponds to the "E" compartment of an SEIR model.


What are the valid states an agent can be in?
----------------------------------------------
This table shows what combination of the states an agent can be in. Blue indicates that if an agent is in state A, they must also be in state B; orange shows that if an agent is in state A, they cannot be in state B; green shows that if an agent is in state A, they may or may not be in state B.

.. raw:: html

    <embed>

        <meta http-equiv="Content-Type" content="text/html; charset=utf-8"><style type="text/css">.ritz .waffle a { color: inherit; }.ritz .waffle .s4{background-color:#ff9900;text-align:center;color:#000000;font-family:'Arial';font-size:8pt;vertical-align:bottom;white-space:nowrap;direction:ltr;padding:2px 3px 2px 3px;}.ritz .waffle .s1{background-color:#ffffff;text-align:left;color:#000000;font-family:'Arial';font-size:8pt;vertical-align:bottom;white-space:nowrap;direction:ltr;padding:2px 3px 2px 3px;}.ritz .waffle .s0{background-color:#ffffff;text-align:left;font-weight:bold;color:#000000;font-family:'Arial';font-size:8pt;vertical-align:bottom;white-space:nowrap;direction:ltr;padding:2px 3px 2px 3px;}.ritz .waffle .s2{background-color:#0000ff;text-align:center;color:#ffffff;font-family:'Arial';font-size:8pt;vertical-align:bottom;white-space:nowrap;direction:ltr;padding:2px 3px 2px 3px;}.ritz .waffle .s3{background-color:#b7e1cd;text-align:center;color:#000000;font-family:'Arial';font-size:8pt;vertical-align:bottom;white-space:nowrap;direction:ltr;padding:2px 3px 2px 3px;}</style><div class="ritz grid-container" dir="ltr" style="overflow-x:auto"><table class="waffle" cellspacing="0" cellpadding="0"><thead><tr><th class="row-header freezebar-origin-ltr"></th><th id="1733083472C0" style="width:100px;" class="column-headers-background"></th><th id="1733083472C1" style="width:80px;" class="column-headers-background"></th><th id="1733083472C2" style="width:80px;" class="column-headers-background"></th><th id="1733083472C3" style="width:80px;" class="column-headers-background"></th><th id="1733083472C4" style="width:80px;" class="column-headers-background"></th><th id="1733083472C5" style="width:80px;" class="column-headers-background"></th><th id="1733083472C6" style="width:80px;" class="column-headers-background"></th><th id="1733083472C7" style="width:80px;" class="column-headers-background"></th><th id="1733083472C8" style="width:80px;" class="column-headers-background"></th><th id="1733083472C9" style="width:80px;" class="column-headers-background"></th><th id="1733083472C10" style="width:80px;" class="column-headers-background"></th><th id="1733083472C11" style="width:80px;" class="column-headers-background"></th><th id="1733083472C12" style="width:80px;" class="column-headers-background"></th><th id="1733083472C13" style="width:80px;" class="column-headers-background"></th><th id="1733083472C14" style="width:80px;" class="column-headers-background"></th></tr></thead><tbody><tr style="height: 20px"><th id="1733083472R0" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s0 softmerge" dir="ltr"><div class="softmerge-inner" style="width:97px;left:-1px">In ↓ you can be →</div></td><td class="s1" dir="ltr">susceptible</td><td class="s1" dir="ltr">naive</td><td class="s1" dir="ltr">exposed</td><td class="s1" dir="ltr">infectious</td><td class="s1" dir="ltr">symptomatic</td><td class="s1" dir="ltr">severe</td><td class="s1" dir="ltr">critical</td><td class="s1" dir="ltr">tested</td><td class="s1" dir="ltr">diagnosed</td><td class="s1" dir="ltr">recovered</td><td class="s1" dir="ltr">dead</td><td class="s1 softmerge" dir="ltr"><div class="softmerge-inner" style="width:77px;left:-1px">known_contact</div></td><td class="s1" dir="ltr">quarantined</td><td class="s1" dir="ltr">vaccinated</td></tr><tr style="height: 20px"><th id="1733083472R1" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">susceptible</td><td class="s2" dir="ltr"></td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R2" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">naive</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R3" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">exposed</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R4" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">infectious</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R5" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">symptomatic</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R6" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">severe</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R7" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">critical</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R8" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">tested</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R9" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">diagnosed</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R10" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">recovered</td><td class="s2" dir="ltr">1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s2" dir="ltr">1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R11" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">dead</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s4" dir="ltr">-1</td><td class="s4" dir="ltr">-1</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R12" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">known_contact</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R13" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">quarantined</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s4" dir="ltr">-1</td><td class="s2" dir="ltr">1</td><td class="s2" dir="ltr">1</td><td class="s3" dir="ltr">0</td></tr><tr style="height: 20px"><th id="1733083472R14" style="height: 20px;" class="row-headers-background"><div class="row-header-wrapper" style="line-height: 20px"></div></th><td class="s1" dir="ltr">vaccinated</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s3" dir="ltr">0</td><td class="s2" dir="ltr">1</td></tr></tbody></table></div><br><br>

    </embed>




Common problems
^^^^^^^^^^^^^^^

I'm getting different results to someone else, or to what I got previously, with the same parameters. Why?
---------------------------------------------------------------------------------------------------------------

One of the trickiest aspects of working with agent-based models is getting the random number stream right. Covasim uses both ``numpy`` and ``numba`` random number streams. These are usually initialized automatically when a simulation is created/run (via ``cv.set_seed(seed)``, which you can call directly as well), but anything that disrupts the random number stream will result in differences between two simulation runs. This is also why seemingly trivial changes (e.g., adding an intervention that doesn't actually do anything) can cause simulation trajectories to diverge.

In addition, random number streams sometimes change with different library versions. For example, due to a bugfix, random number streams changed between ``numba`` 0.48 and 0.49. Therefore, simulation run with ``numba`` 0.48 or earlier won't (exactly) match simulations run with  ``numba`` 0.49 or later.

If you're having trouble reproducing results between simulations that should be the same, check: (a) the Covasim version, (b) the ``numpy`` version, (c) the ``numba`` version, and (d) the SynthPops version (if using). If all these match but results still differ, then a useful debugging strategy can be to insert ``print(np.random.rand())`` at various points throughout the code to see at what point the two versions diverge.


Why doesn't the webapp accept long durations or large population sizes?
---------------------------------------------------------------------------------

The webapp is limited by the results needing to be returned before the request times out. However, when running directly via Python, you are limited only by your computer's RAM (and your patience) in terms of simulation duration or population size.


Why do parallel simulations fail on Windows or in Jupyter notebooks? 
---------------------------------------------------------------------------------

If you are running on Windows, because of the way Python's ``multiprocessing`` library is implemented, you must start the run from inside a ``__main__`` block (see discussion `here <https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main>`__).
For example, instead of this:

.. code-block:: python

    import covasim as cv
    sims = [cv.Sim(pop_infected=100, beta=0.005*i, label=f'Beta factor {i}') for i in range(5)]
    msim = cv.MultiSim(sims)
    msim.run()
    msim.plot()

do this:

.. code-block:: python

    import covasim as cv
    sims = [cv.Sim(pop_infected=100, beta=0.005*i, label=f'Beta factor {i}') for i in range(5)]
    msim = cv.MultiSim(sims)

    if __name__ == '__main__':
        msim.run()
        msim.plot()

When parallelizing inside Jupyter notebooks, sometimes a "Duplicate signature" error will be encountered. This is because of how multiprocessing conflicts with Jupyter's internal threading (see discussion `here <https://stackoverflow.com/a/23641560/4613606>`__). One solution is to move ``msim.run()`` (or other parallel command) to a separate ``.py`` file, and not have it be part of the notebook itself. This problem should be fixed in version 2.0 though, so if you're using an older version, consider upgrading.