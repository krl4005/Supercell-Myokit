"""Runs a genetic algorithm for parameter tuning on specified target objective.

Example usage:
    config = <GENERATE CONFIG OBJECT>
    ga_instance = ParameterTuningGeneticAlgorithm(config)
    ga_instance.run()
"""

import random
#from typing import List
from math import log10

from deap import base, creator, tools
import myokit
#import numpy as np
#from scipy.interpolate import interp1d
#from pickle import dump, HIGHEST_PROTOCOL, load
#import time
#import copy
#from multiprocessing import Pool
#from os import environ
#import pkg_resources
#import matplotlib.pyplot as plt


def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
    final_population : List[model obects]
    """
    print('Evaluating initial population.')

    population = toolbox.population(GA_CONFIG.population_size)

    fitnesses = toolbox.map(toolbox.evaluate, population)
   
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Store initial population details for result processing.
    initial_population = []
    for i in range(len(population)):
        initial_population.append(
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=population[i][0].default_parameters,
                fitness=population[i].fitness.values))

    final_population = [initial_population]

    avg_fitness = []

    for generation in range(1, ga_params.max_generations):
        print('Generation {}'.format(generation))
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.
        ga_params.previous_population = population

        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        for i in offspring:
            if random.random() < ga_params.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.
        updated_individuals = [i for i in offspring if not i.fitness.values]

        targets = [copy.deepcopy(ga_params.targets) for i in
                     range(0, len(updated_individuals))]

        eval_input = np.transpose([updated_individuals, targets])

        fitnesses = toolbox.map(toolbox.evaluate, eval_input)

        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = fit

        population = offspring

        # Store intermediate population details for result processing.
        intermediate_population = []
        for i in range(len(population)):
            intermediate_population.append(
                genetic_algorithm_results.ParameterTuningIndividual(
                    parameters=population[i][0].default_parameters,
                    fitness=population[i].fitness.values))

        final_population.append(intermediate_population)

        generate_statistics(population)

        fitness_values = [i.fitness.values for i in population]

        #TODO: create exit condition for all multi-objective
        #if len(avg_fitness) > 3:
        #    if len(fitness_values) > 1:
        #        print('multiobjective')
        #    if np.mean(fitness_values) >= max(avg_fitness[-3:]):
        #        break

        #avg_fitness.append(np.mean(fitness_values))

    
    final_ga_results = genetic_algorithm_results.GAResultParameterTuning(
            'kernik', TARGETS, 
            final_population, GA_PARAMS,
            )

    return final_ga_results


def get_model_response(model, command, prestep=5000.0, is_command_prestep=True):
    """
    Parameters
    ----------
    model : CellModel
        This can be a Kernik, Paci, or OR model instance
    protocol : VoltageClampProtocol
        This can be any VoltageClampProtocol

    Returns
    -------
    trace : Trace
        Trace object with the current and voltage data during the protocol

    Accepts a model object, applies  a -80mV holding prestep, and then 
    applies the protocol. The function returns a trace object with the 
    recording during the input protocol.
    """
    if is_command_prestep:
        prestep_protocol = protocols.VoltageClampProtocol(
            [protocols.VoltageClampStep(voltage=-80.0,
                                        duration=prestep)])
    else:
        prestep_protocol = command

    if isinstance(command, TargetObjective):
        if command.protocol_type == 'Dynamic Clamp':
            #TODO: Aperiodic Pacing Protocol

            prestep_protocol = protocols.AperiodicPacingProtocol(
                GA_PARAMS.model_name)
            command = prestep_protocol
            model.generate_response(prestep_protocol,
                        is_no_ion_selective=True)
            response_trace = model.generate_response(command,
                    is_no_ion_selective=True)
        else:
            model.generate_response(prestep_protocol,
                        is_no_ion_selective=False)
            model.y_ss = model.y[:, -1]
            response_trace = model.generate_response(command.protocol,
                    is_no_ion_selective=False)
    else:
        model.generate_response(prestep_protocol,
                    is_no_ion_selective=False)

        model.y_ss = model.y[:, -1]

        response_trace = model.generate_response(command,
                is_no_ion_selective=False)

    return response_trace


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
    """
    Evaluates performance of an individual compared to the target obj.

        Returns
        -------
            error: Number
                The error between the trace generated by the individual's
                parameter set and the baseline target objective.
    """
    mod, proto, x = myokit.load('./run_tor_ord.py')
    

    return errors


def _mate(i_one, i_two):
    """Performs crossover between two individuals.

    There may be a possibility no parameters are swapped. This probability
    is controlled by `self.config.gene_swap_probability`. Modifies
    both individuals in-place.

    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].default_parameters.items():
        if random.random() < GA_PARAMS.gene_swap_probability:
            i_one[0].default_parameters[key],\
                i_two[0].default_parameters[key] = (
                    i_two[0].default_parameters[key],
                    i_one[0].default_parameters[key])


def _mutate(individual):
    """Performs a mutation on an individual in the population.

    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.

    Args:
        individual: An individual to be mutated.
    """
    keys = [p.name for p in GA_PARAMS.tunable_parameters]

    for key in keys:
        if random.random() < GA_PARAMS.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_PARAMS.params_lower_bound) or
                   (new_param > GA_PARAMS.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0].default_parameters[key],
                        individual[0].default_parameters[key] * .1)

            individual[0].default_parameters[key] = new_param


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
             tournament_size):
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


global GA_CONFIG
GA_CONFIG = Ga_Config(population_size=10, 
                      max_generations=2,
                      params_lower_bound=0.1,
                      params_upper_bound=10,
                      tunable_parameters=['GKr', 'GKs', 'GCaL', 'GNaL', 'GJup'],
                      mate_probability=0.9,
                      mutate_probability=0.9,
                      gene_swap_probability=0.2,
                      gene_mutation_probability=0.2,
                      tournament_size=4)

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

    #p = Pool()
    toolbox.register("map", map)

    final_population = run_ga(toolbox)

    return final_population

start_ga()
