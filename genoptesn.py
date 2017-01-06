#!/usr/bin/env python2
import argparse
import json
import logging
import numpy as np
import os
import random
import sys

from deap import base, creator, tools, algorithms
from functools import partial
from scoop import futures

import esnet
import parameterhelper

# Check python version (for str/basestring)
if sys.version_info[0] == 3:
    str_type = str,
else:
    str_type = basestring,

# Initialize logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

###############################################################################################
# The next part needs to be in the global scope, since all workers
# need access to these variables (pickling problems).
############################################################################
# Parse input arguments
############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data file", type=str)
parser.add_argument("optconfig", help="path to optimization config file", type=str)
parser.add_argument("esnconfig", help="path to where the ESN config file should be saved", type=str)
parser.add_argument("--percent_dim", help="use dimensionality as a percentage of the reservoir size. DEFAULT: False.", type=bool, default=False, const=True, nargs='?')
args = parser.parse_args()

############################################################################
# Read config file
############################################################################
paramhelper = parameterhelper.ParameterHelper(args.optconfig, args.percent_dim)
optconfig = paramhelper._optimization

############################################################################
# Load data
############################################################################
logger.info("Loading data (%s)"%args.data)
# If the data is stored in a directory, load the data from there. Otherwise,
# load from the single file and split it.
if os.path.isdir(args.data):
    Xtr, Ytr, Xval, Yval, _, _ = esnet.load_from_dir(args.data)

else:
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, Xval, Yval, _, _ = esnet.generate_datasets(X, Y)

############################################################################
# Initialization of the genetic algorithm
############################################################################
# Fitness and individual. Different formats, depending on dimensionality reduction.
if paramhelper._fixed_values['embedding'] == 'identity':
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # -1.0 => minimize function
else:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-0.1)) # -1.0 => minimize function

# The individuals are dicts of numbers (parameters)
# The length and the type of number varies across ESN configurations.
creator.create("Individual", dict, fitness=creator.FitnessMin)
###############################################################################################

def get_minmax(prototype, key, individual = None):
    """
    Returns the minimum and maximum value for a specific parameter.
    If the parameter is defined as a string (reference to another parameter),
    it is pulled from that parameter in the individual/prototype.
    """
    if individual is not None:
        if isinstance(prototype[key][1], str_type):
            # Reference to another parameter
            minval = individual[prototype[key][1]]
        else:
            # Number
            minval = prototype[key][1]

        if isinstance(prototype[key][2], str_type):
            # Reference to another parameter
            maxval = individual[prototype[key][2]]
        else:
            # Number
            maxval = prototype[key][2]
    else:
        if isinstance(prototype[key][1], str_type):
            # Reference to another parameter
            minval = prototype[prototype[key][1]][1]
        else:
            # Number
            minval = prototype[key][1]

        if isinstance(prototype[key][2], str_type):
            # Reference to another parameter
            maxval = prototype[prototype[key][2]][2]
        else:
            # Number
            maxval = prototype[key][2]

    return minval, maxval

def gen_individual(prototype):
    """
    Generate individuals for the population based on the recipe in 'prototype'.
    """
    def gen_val(dtype, minval, maxval):
        if dtype == 'f':
            value = random.uniform(minval, maxval)
        elif dtype == 'i':
            value = random.randint(minval, maxval)

        return value

    individual = dict()

    for key in prototype:
        dtype = prototype[key][0]

        # Get the minimum and maximum value for this parameter
        minval, maxval = get_minmax(prototype, key, None)

        individual[key] = gen_val(dtype, minval, maxval)

    # Ensure that all restrictions are satisfied.
    individual = validate_individual(individual, prototype)

    return individual

def validate_individual(individual, prototype):
    """
    Checks the parameters for the individual against the prototype and adjusts
    type/values accordingly.
    """
    for key in individual:
        # Get the minimum and maximum value for this parameter
        minval, maxval = get_minmax(prototype, key, individual)

        # Check bounds
        if individual[key] < minval:
            individual[key] = minval
        elif individual[key] > maxval:
            individual[key] = maxval

        # Cast to int if applicable
        if prototype[key][0] == 'i':
            individual[key] = int(round(individual[key]))

    return individual

def check_individuals(prototype):
    """
    Decorator function to ensure that certain attributes will be integers and
    that each value is within its bounds.
    """
    def decorator(func):
        def wrapper(*args, **kargs):

            offspring = func(*args, **kargs)
            for child in offspring:
                child = validate_individual(child, prototype)

            return offspring
        return wrapper
    return decorator

def cxTwoDictWeave(ind1, ind2):
    """
    Weave the two dictionary individuals.
    """
    # We assume both have the same number of elements
    size = len(ind1)

    # Create a list of zeros and ones. The elements with ones are swapped.
    crossover = [random.randint(0, 1) for x in range(size)]

    for key,idx in zip(ind1, range(size)):
        if crossover[idx] == 1:
            # Swap values
            ind1[key], ind2[key] = ind2[key], ind1[key]

    return ind1, ind2

def mutGaussianDict(individual, mu, sigma, indpb):
    """
    Gaussian mutation. Supports both
        - sigma ~ dict => Keys must have same name as in the individual dict
        - sigma ~ float => Same variance on all parameters.
    """
    size = len(individual)

    for key in individual:
        if random.random() < indpb:
            if isinstance(sigma, dict):
                individual[key] += random.gauss(mu, sigma[key])
            else:
                individual[key] += random.gauss(mu, sigma)

    return individual,

def evaluate_ind(individual):
    """
    Fitness function.
    Trains a randomly initiated ESN using the parameters in 'individual' and
    the config file.

    Returns touple with error metric (touple required by DEAP)
    """

    parameters = paramhelper.get_parameters(individual)


    # Run a few times to get the average error over several networks.
    n_eval = optconfig['n_eval']
    errors = np.empty((n_eval,), dtype=float)

    for i in range(n_eval):
        _, errors[i] = esnet.run_from_config(Xtr, Ytr, Xval, Yval, parameters)

    error = np.mean(errors)

    # Do we have dimensionality reduction?
    if parameters['n_dim'] is None:
        return error,
    else:
        return error,float(parameters['n_dim'])/parameters['n_internal_units']

def save_parameters(halloffame, filename):
    """
    Saves the parameters from the best individual in 'halloffame' to 'filename'
    in the JSON format.

    Parameters that were not tuned by the genetic algorithm is retrieved from
    the genopt config file.
    """
    best_individual = halloffame[-1]

    best_parameters = paramhelper.get_parameters(best_individual)

    # Save
    json.dump(best_parameters, open(filename +'.json', 'w'), indent=4)

    return

def init_toolbox(prototype, sigma):
    """
    Initialize and return the DEAP toolbox object.
    """
    toolbox = base.Toolbox()

    gen_ind = partial(gen_individual, prototype)
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_ind)

    # Generate population as a list of individuals (list of dicts)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.population(n=optconfig['population_size'])

    # Crossover and mutation
    toolbox.register("mate", cxTwoDictWeave)
    toolbox.decorate("mate", check_individuals(prototype))

    # Check if a GLOBAL sigma is defined. If so, use that.
    if optconfig['sigma'] is not None:
        sigma = optconfig['sigma']

    toolbox.register("mutate", mutGaussianDict, mu=optconfig['mu'], sigma=sigma, indpb=0.2)

    # Ensure that min/max values are respected and that the datatype is correct
    toolbox.decorate("mutate", check_individuals(prototype))

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=4)

    # Fitness function
    toolbox.register("evaluate", evaluate_ind)

    # Enable multithreading
    if optconfig['parallel'] == True:
        toolbox.register("map", futures.map)

    return toolbox

def init_stats():
    """
    Initiate statistical functions.
    Returns DEAP stats object.
    """
    # Register statistical functions
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return stats

def main():
    logger.info("Initializing genetic algorithm")

    # Generate prototype (recipe) for the individuals in the population
    prototype, sigma = paramhelper.get_prototype()

    # Initiate DEAP toolbox
    toolbox = init_toolbox(prototype, sigma)

    # Initiate statistical functions
    stats = init_stats()

    # Generate initial population
    pop = toolbox.population(n = optconfig['population_size'])

    # Run optimization
    cxpb = optconfig['cxpb']
    mutpb = optconfig['mutpb']
    ngen = optconfig['n_generations']
    n_offsprings = optconfig['n_offsprings']
    halloffame = tools.HallOfFame(maxsize=1)

    logger.info("Running GA optimization")

    final_population, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=optconfig['population_size'],
            lambda_ = n_offsprings, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=True, halloffame=halloffame)

    ############################################################################
    # Save ESN config
    ############################################################################"""
    logger.info("Saving the best parameters")
    save_parameters(halloffame, args.esnconfig)

    logger.info("Done")

if __name__ == "__main__":
    main()
