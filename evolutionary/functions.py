import os
import sys
import random as rand
import numpy as np
from torch import tensor

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import proect modules
from evolutionary.individual import Individual

#Normalize the fitness of a generation
def norm_fitness_of_generation(generation: list[Individual]):
    cost_of_generation = [individual.fitness for individual in generation]
    sum_of_generation_cost = sum(cost_of_generation)
    fitness_of_generation = [individual_cost/sum_of_generation_cost for individual_cost in cost_of_generation]
    return fitness_of_generation

#Returns the index of the selected parent through roulette wheel selection
def roulette_wheel_selection(fitness_of_generation: list[float]):
    selected_fitness = rand.uniform(0,1)
    curr_fitness_sum = 0
    selected_idx = 0
    for idx, fitness in enumerate(fitness_of_generation):
        curr_fitness_sum += fitness
        if curr_fitness_sum > selected_fitness:
            selected_idx = idx
            break

    return selected_idx

#Select up to beam width best solutions from pool
def select_solutions_from_gen(generation: list[Individual], gen_size: int):

    #Store index and cost of each solution from input solution pool
    cost_solution_pool = [(idx, individual.fitness) for idx, individual in enumerate(generation)]

    #Sort solution pool by the cost
    cost_solution_pool.sort(key=lambda tup: tup[1], reverse=True)

    #Store the best solutions up to beam width
    selected_individuals = []
    for i in range(0, gen_size):
        selected_individuals.append(generation[cost_solution_pool[i][0]])

    return selected_individuals

#Resets the fitness and steps an Agent has for every new generation
def resetFitness(generation: list[Individual]):
    for i in generation:
        i.fitness = 0

#Mutation operator
def mutate(params: tensor, mutations: int = 1):
    for _ in range(mutations):
        params[rand.randrange(len(params))] += np.random.normal(0.0, 0.02)
    return params