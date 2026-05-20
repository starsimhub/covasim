import sciris as sc
import covasim as cv
import fire
import wandb

def train(beta:float=0.015, 
          pop_infected: int=10, 
          rel_death_prob: float=1.0,
          rel_severe_prob: float=1.0,
          rel_crit_prob: float=1.0,
          start_day: str='2019-12-25',
          datafile='tests/example_data.csv') -> None:
    """
    Perform hyperparameter sweep with Weights and Biases
    https://docs.wandb.com/sweeps
    """
    sc.makefilepath(datafile, checkexists=True)

    pars = dict(
        beta = beta,
        pop_infected = pop_infected,
        rel_death_prob = rel_death_prob,
        rel_crit_prob = rel_crit_prob,
        start_day = start_day,
        )

    # instantiate wandb run
    wb_handle = wandb.init(config=pars, project="covasim")
    run_id = wandb.run.id

    # Create and run the simulation
    sc.heading('Hyperparmeter Sweep')
    sim = cv.Sim(pars=pars, datafile=datafile)
    sim.run(verbose=False)
    likelihood = sim.likelihood()

    # log relevant metrics and artifacts
    wandb.log({'likelihood': likelihood})
    sim.plot(do_show=False, 
             do_save=True, 
             fig_path=sc.makefilepath(folder=wandb.run.dir, filename=f'{run_id}.png'))
    #wandb.save(datafile)
    sc.saveobj(sim.pars, folder=wandb.run.dir, filename=f'pars_{run_id}.pkl')
    
if __name__ == '__main__':
    fire.Fire(train)
