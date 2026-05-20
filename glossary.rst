========
Glossary
========

This document contains a list of commonly used terms, both in Covasim and for COVID-19 more generally. If there are others you'd like to see included, please email us at info@covasim.org. Cross-references are listed in *italics*.

- **agent**: The basic unit of the model. Typically, one agent in the simulation corresponds to one person in the population, but the population may also be subsampled, such that a larger population is represented by a smaller number of agents. This depends on the assumption that the full population behaves the same way that a subset of the population behaves. See also *dynamic rescaling*.
- **agent-based model**: The modeling approach that tracks individual *agents* in the population. Compared to *compartmental model*, agent-based models are typically slower, but allow much greater detail in simulations. Covasim is an agent-based model.
- **back-tracking**: The process of finding the person who infected a person who has just been diagnosed.
- **basic reproduction number**: See *reproduction number*.
- **beta**: The probability of transmission from an *infectious* person to a *susceptible* person, also known as infectiousness or transmissibility. The overall transmission probability depends on the network *layer* two people are connected by, as well as the infected person's *viral load*, the susceptible person's age, and other factors.
- **case**: A person who has tested positive (i.e., been diagnosed) with COVID-19.
- **case detection ratio** (CDR): The proportion of all infections that have been diagnosed. For example, if 10,000 people have been infected and 1,500 have been diagnosed, the CDR will be 15%.
- **case fatality ratio** (CFR): The proportion of people who have been diagnosed who eventually die. This is typically much higher than the *infection fatality ratio* since more severe cases are more likely to be diagnosed. Typical CFRs are 5-10%.
- **compartmental model**: The modeling approach that does not track individual agents, but instead considers compartments with rates of transfer between them (e.g., the total number of susceptible people in the compartment S, and their rate of transfer into the exposed compartment E). Compared to *agent-based models*, compartmental models are typically faster but rely more heavily on assumptions and approximations.
- **contact tracing**: The program for getting in touch with people who have been exposed to contacts of a known positive *case*. 
- **COVID-19**: Technically defined as the disease caused by infection with the virus *SARS-CoV-2*, but informally used to refer to both the infection and the disease.
- **dynamic rescaling**: A method by which a smaller number of *agents* is used to represent a larger number of people. For example, 10,000 agents may be used to represent 100,000 people, with an initial 1:1 ratio at the beginning of the simulation, and as the number of infections increases during the simulation, scaling up to a 1:10 ratio.
- **effective reproduction number**: See *reproduction number*.
- **exposed**: The state of a person who has been infected with *SARS-CoV-2*, but before they become infectious *infectious*. One of the four states of the *SEIR* model. Also known as *infected*, but the term "exposed" is used to avoid confusion in the acronym SEIR.
- **generation time**: The time between the infection of the *primary infection* and one or more *secondary infections*. Compare *serial interval*.
- **index case**: See *primary case*.
- **infection fatality ratio** (IFR): The proportion of people who become infected who eventually die. Typical IFRs are 0.3-1.5%, depending on the age distribution and other factors.
- **infectious**: The state of a person who is capable of passing infection on to others. One of the four states in the *SEIR* model.
- **infectiousness**: See *beta*.
- **intervention**: Any program or policy implemented to limit the spread of COVID-19; these might including testing or tracing programs, physical distancing, mobility restrictions, vaccination, and other programs. See also *non-pharmaceutical intervention*.
- **isolation**: The behavior change that occurs after a person has received a positive diagnosis. See also *quarantine*.
- **layer**: The way in which two people are connected. Also called a contact layer or network layer. Examples include households, schools, workplaces, communities, and *LTCFs*.
- **long-term care facilities** (LTCFs): Facilities to care for people over long periods of time, also known as assisted living facilities, aged-care facilities, or nursing homes.
- **neutralizing antibodies** (NAbs): The antibodies that offer protection against SARS-CoV-2 infection and COVID-19 disease severity, which may be due to prior infection (natural immunity) or from vaccination.
- **non-pharmaceutical intervention**: Any *intervention* to reduce the spread or severity of COVID-19 other than therapeutics or vaccination. Examples including distancing, hand washing, mask wearing, etc.
- **parameter**: One or more values (usually numbers) that define how the simulation runs. For example, "probability of developing severe disease" could be considered a parameter (i.e., list of values by age), as could "probability of developing severe disease for people aged 60-70" (i.e., a single number). Parameters can also be qualitative values (e.g., population type) or true/false values (e.g., whether or not to dynamically rescale the population).
- **primary case**: The earliest-infected person diagnosed in a cluster of infections, i.e. the person who is the source of other infections in that cluster. Usually, but not always, this person is also the earliest person in the cluster to be diagnosed. (If *back-tracing* is used, the primary case may be diagnosed after one or more of the secondary cases.) See also *secondary infections*.
- **primary infection**: Same as *primary case*, but not necessarily diagnosed.
- **program**: See *intervention*.
- **quarantine**: The behavior change that occurs when a person has been notified that they have been in contact with a person who has tested positive. See also *isolation*.
- **R0, R_e, R_eff**: See *reproduction number*.
- **recovered**: A person who has been infected with *SARS-CoV-2* and since recovered. They are usually considered to be immune (reinfection is not considered), and thus **removed** from the model.
- **removed**: In the context of *SEIR* modeling, refers to someone who has either recovered from infection or has died; i.e. they are no longer susceptible or infected, so act as if they have been "removed" from the simulation.
- **reproduction number**: The average number of *secondary infections* caused by each each *primary infection*. In the absence of *interventions*, this is called the basic reproduction number, or R0. Otherwise, it is usually called the effective reproduction number, abbreviated R_e or R_eff. If R_e > 1, then the epidemic is (usually) growing; if R_e < 1, the epidemic is (usually) shrinking.
- **random seed**: The starting point for a given simulation used to convert probabilities (e.g., 10% probability of infection) to events (e.g., a person actually being infected). Two simulations will produce identical results if (and usually only if) they have identical *parameters* as well as the same random seed.
- **rescaling**: When one agent does not necessarily correspond to one person in the population; for example, 10,000 agents may be used to represent 100,000 people in the population. See also *dynamic rescaling*.
- **SARS-CoV-2**: The virus responsible for causing *COVID-19*. Informally, the two terms are used interchangeably.
- **secondary case**: The people who are infected by the *index case* who have been diagnosed.
- **secondary infection**: Same as *secondary case*, but not necessarily diagnosed.
- **SEIR**: A common type of epidemic model, of which Covasim is an example. It stands for *Susceptible* - *Exposed* - *Infectious* - *Removed* (or "recovered"), referring to the four different states that *agents* may have. Both *agent-based models* and *compartmental models* may have SEIR structure.
- **serial interval**: The time between when the *primary case* develops symptoms and when *secondary cases* develop infections. Usually used in reference to symptomatic and diagnosed infections (which is a subset of all infections). While the *generation time* is of more interest, the serial interval is easier to measure, so is often used as a proxy for it.
- **simulation**: A single realization of the model, consisting of: the model itself (i.e., Covasim), along with the *random seed* and other *parameters*. Running the same simulation is expected to produce the same results every time.
- **susceptible**: The state of a person who has not been infected with *SARS-CoV-2*, and can become infected. One of the four states of the *SEIR* model. Also sometimes referred to simply as uninfected.
- **susceptibility**: The probability of a person becoming infected after being exposed to an infectious person. This typically depends on age, as well as whether or not a person has been vaccinated.
- **testing**: The program for diagnosing people with COVID-19. Most typically reverse to polymerase chain reaction (PCR) tests, but can also refer to antigen tests.
- **transmissibility**: See *beta*.
- **transmission tree**: The network of infections in the model; so named because when plotted, it resembles a tree, with the "trunk" consisting of the initial infections, and the "twigs" consisting of the most recent infections.
- **viral load**: The amount of virus in an infected person's body. Typically it is assumed that infectiousness is proportional to viral load, although the two are not necessarily linearly proportional.