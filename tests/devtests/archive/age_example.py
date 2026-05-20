import covasim as cv; cv

sim = cv.Sim()
sim.run()

dead_under70 = sum(sim.people.dead[sim.people.age<70])
dead_over70 = sum(sim.people.dead[sim.people.age>=70])
print(f'Under: {dead_under70}, over: {dead_over70}')

dead_age = sim.people.age[sim.people.true('dead')]
print(f'Mean age at death: {dead_age.mean()}')