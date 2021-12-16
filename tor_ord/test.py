#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas
from math import log10 
import random

# %%
def initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 
    Returns:
        An Individual with conductance parameters 
    """
    # Builds a list of parameters using random upper and lower bounds.
    params_lower_bound = 0.1
    params_upper_bound = 10
    iks_upper_bound = 0.01
    iks_lower_bound = 100

    tunable_parameters = ['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','jup_multiplier']

    lower_exp = log10(params_lower_bound)
    upper_exp = log10(params_upper_bound)

    iks_lower_exp = log10(iks_lower_bound)
    iks_upper_exp = log10(iks_upper_bound)
    #initial_params = [10**random.uniform(lower_exp, upper_exp)
    #                  for i in range(0, len(
    #                      GA_CONFIG.tunable_parameters))]

    initial_params = []
    for i in range(0, len(tunable_parameters)):
        if i == 1: #increase bounds for iks to try to allow for better convergence 
            initial_params.append(10**random.uniform(iks_lower_exp, iks_upper_exp))
        else:
            initial_params.append(10**random.uniform(lower_exp, upper_exp))

    keys = [val for val in tunable_parameters]
    return dict(zip(keys, initial_params))

ind0 = initialize_individuals()
print(ind0)

#%% 

def mutate(individual):
    """Performs a mutation on an individual in the population.
    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.
    Args:
        individual: An individual to be mutated.
    """
    gene_mutation_probability = 0.2
    params_lower_bound = 0.1
    params_upper_bound = 10
    iks_upper_bound = 100
    iks_lower_bound = 0.01

    keys = [k for k, v in individual.items()]

    for key in keys:
        print(key)
        if key == 'i_ks_multiplier':
            rand_val = random.random()
            print(rand_val)
            if rand_val < gene_mutation_probability:
                new_param = -1

                while ((new_param < iks_lower_bound) or
                    (new_param > iks_upper_bound)):
                    new_param = np.random.normal(
                            individual[key],
                            individual[key] * .1)

                individual[key] = new_param
        else:
            rand_val = random.random()
            print(rand_val)
            if rand_val < gene_mutation_probability:
                new_param = -1

                while ((new_param < params_lower_bound) or
                    (new_param > params_upper_bound)):
                    new_param = np.random.normal(
                            individual[key],
                            individual[key] * .1)

                individual[key] = new_param
    return(individual)

mut_ind0 = mutate(ind0)
print(mut_ind0)

# %%
