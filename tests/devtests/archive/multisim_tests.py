import covasim as cv

sim = cv.Sim() # Create the sim
msim1 = cv.MultiSim(sim, n_runs=5) # Create the multisim
msim1.run() # Run them in parallel
msim1.combine() # Combine into one sim
msim1.plot() # Plot results

sim = cv.Sim() # Create the sim
msim2 = cv.MultiSim(sim, n_runs=11, noise=0.1) # Set up a multisim with noise
msim2.run() # Run
msim2.reduce() # Compute statistics
msim2.plot() # Plot

# Run multiple sims
n_sims = 5
betas = [0.015*(1+0.02*i) for i in range(n_sims)]
labels = [f'beta={beta}' for beta in betas]
sims = [cv.Sim(beta=betas[i], label=labels[i]) for i in range(n_sims)] # Create sims
for sim in sims:
    sim.run()
msim3 = cv.MultiSim(sims) # Convert to multisim
msim3.plot() # Plot as single sim