====================
Running calibrations
====================

Optuna
======

The file ``optuna_example.py`` in the parent ``examples`` folder includes a complete example, taken from Tutorial 7, for running a simple calibration.


Weights and Biases
==================

The remaining files are an example of calibration using `Weights and Biases`_. These instructions are a minimal subset of `these docs`_.

.. _Weights and Biases: https://www.wandb.com/
.. _these docs: https://docs.wandb.com/sweeps

To begin a sweep follow these steps:

1.  Make sure wandb is installed::

        > pip install wandb

2.  Login to wandb::

        > wandb login

3.  Initialize wandb (optional)::

        > wandb init

4.  When asked to choose a project make sure you  select **covasim**.  If you don't see a project with this name, instead select **Create New** and name your project **covasim**.
5.  From the **root of this repo**, initialize a sweep::

        # Choose the yaml file that corresponds to the search strategy.
        wandb sweep sweep/sweep-random.yaml

    This command will print out a **sweep ID**. Copy that to use in the next step!

6.  Launch agent(s)::

        wandb agent your-sweep-id


    From `the docs`_:

    > You can run wandb agent on multiple machines or in multiple processes on the same machine, and each agent will poll the central W&B Sweep server for the next set of hyperparameters to run.

.. _the docs: https://docs.wandb.com/sweeps/quickstart