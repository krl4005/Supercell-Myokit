"""Runs a genetic algorithm for parameter tuning on specified target objective.

Example usage:
    config = <GENERATE CONFIG OBJECT>
    ga_instance = ParameterTuningGeneticAlgorithm(config)
    ga_instance.run()
"""

import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool

from deap import base, creator, tools # pip install deap
import myokit
import numpy as np


def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
    final_population : List[model obects]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals and returns initial population
    population = toolbox.population(GA_CONFIG.population_size)

    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)
    # Note: visualize individual fitnesses with: population[0].fitness

    # Store initial population details for result processing.
    final_population = [population]

    for generation in range(1, GA_CONFIG.max_generations):
        print('Generation {}'.format(generation))
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.

        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            if random.random() < GA_CONFIG.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.

        # 8. Evaluating the offspring of the current generation
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit,)

        population = offspring

        final_population.append(population)

    return final_population


def _initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 

    Returns:
        A model instance with a new set of parameters
    """
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp)
                      for i in range(0, len(
                          GA_CONFIG.tunable_parameters))]

    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))


def _evaluate_fitness(ind):
    ead_error = get_ead_error(ind)
    feature_error = get_feature_errors(ind)

    error = feature_error #+ ead_error

    return error


def _mate(i_one, i_two):
    """Performs crossover between two individuals.

    There may be a possibility no parameters are swapped. This probability
    is controlled by `self.config.gene_swap_probability`. Modifies
    both individuals in-place.

    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].items():
        if random.random() < GA_CONFIG.gene_swap_probability:
            i_one[0][key],\
                i_two[0][key] = (
                    i_two[0][key],
                    i_one[0][key])


def _mutate(individual):
    """Performs a mutation on an individual in the population.

    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.

    Args:
        individual: An individual to be mutated.
    """
    keys = [k for k, v in individual[0].items()]

    for key in keys:
        if random.random() < GA_CONFIG.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_CONFIG.params_lower_bound) or
                   (new_param > GA_CONFIG.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0][key],
                        individual[0][key] * .1)

            individual[0][key] = new_param


def get_is_viable_cell(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    for k, v in ind[0].items():
        mod['multipliers'][k].set_rhs(v)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(10000)

    is_viable_ap = get_is_viable_cell(ind)

    t = dat['engine.time']
    v = dat['membrane.v']
    cai = dat['intracellular_ions.cai']
    i_stim = dat['stimulus.i_stim']

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3]
    end_ap = peaks[-2]

    if ((min(v[start_ap:end_ap]) >-60) or (max(v[start_ap:end_ap]) <0)):
        t = np.array(dat['engine.time'][start_ap:end_ap])
        plt.plot(t, dat['membrane.v'][start_ap:end_ap])
        plt.show()
        import pdb
        pdb.set_trace()
        return False
    else:
        return True


def get_feature_errors(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    for k, v in ind[0].items():
        mod['multipliers'][k].set_rhs(v)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(50000) # set time in ms

    ap_features = {}

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']
    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-2]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-900))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [10, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]

        ap_features[f'apd{apd_pct}'] = apd_val

    # Calcium/CaT features######################## 
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp

    for cat_pct in [10, 50, 90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

        ap_features[f'cat{cat_pct}'] = catd_val 

    error = 0

    if GA_CONFIG.cost == 'function_1':
        for k, v in ap_features.items():
            error += (GA_CONFIG.feature_targets[k][1] - v)**2
    else:
        for k, v in ap_features.items():
            if ((v < GA_CONFIG.feature_targets[k][0]) or
                    (v > GA_CONFIG.feature_targets[k][2])):
                error += 1000 

    return error


def get_ead_error(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    for k, v in ind[0].items():
        mod['multipliers'][k].set_rhs(v)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(50000)

    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(50000)



#def plot_population(pop, generation=None):
#    #if None, then plot final gen 
#    if generation is None:
#        gen_to_plot = pop[-1]
#    else:
#        gen_to_plot = pop[generation]
#
#    keys = [k for k, v in gen_to_plot[0][0].items()]
#
#    all_cond_vals = {'i_cal_pca_multiplier': [],
#                     'i_ks_multiplier': [],
#                     'i_kr_multiplier': [],
#                     'i_nal_multiplier': [],
#                     'jup_multiplier': []} 
#
#    for ind in gen_to_plot:


class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             params_lower_bound,
             params_upper_bound,
             tunable_parameters,
             mate_probability,
             mutate_probability,
             gene_swap_probability,
             gene_mutation_probability,
             tournament_size,
             cost,
             feature_targets):
        self.population_size = population_size
        self.max_generations = max_generations
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.tunable_parameters = tunable_parameters
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.cost = cost
        self.feature_targets = feature_targets

feature_targets = {'dvdt_max': [80, 86, 92],
                   'apd10': [2, 8, 20],
                   'apd50': [200, 220, 250],
                   'apd90': [250, 270, 300],
                   'cat_amp': [3E-4, 3.5E-4, 4E-4],
                   'cat10': [80, 100, 120],
                   'cat50': [200, 220, 240],
                   'cat90': [450, 470, 490]}

# 1. Initializing GA hyperparameters
global GA_CONFIG
GA_CONFIG = Ga_Config(population_size=3, 
                      max_generations=2,
                      params_lower_bound=0.5,
                      params_upper_bound=2,
                      tunable_parameters=['i_cal_pca_multiplier',
                                          'i_ks_multiplier',
                                          'i_kr_multiplier',
                                          'i_nal_multiplier',
                                          'jup_multiplier'],
                      mate_probability=0.9,
                      mutate_probability=0.9,
                      gene_swap_probability=0.2,
                      gene_mutation_probability=0.2,
                      tournament_size=4,
                      cost='function_2',
                      feature_targets=feature_targets)


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

creator.create('Individual', list, fitness=creator.FitnessMin)

def start_ga():
    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individuals)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    # To speed things up
    #p = Pool()
    #toolbox.register("map", p.map)
    toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population

# Final population includes list of individuals from each generation
# To access an individual from last gen:
# final_population[-1][0].fitness.values[0] Gives you fitness/error
# final_population[-1][0][0] Gives you dictionary with conductance values
final_population = start_ga()

#plot_population(final_population, generation=None)
