import os
import sys
import random as rand
import gym
import numpy as np
from torch import save, load
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evolutionary.classes import Individual
from evolutionary.functions import mutate, norm_fitness_of_generation, roulette_wheel_selection, select_solutions_from_gen, resetFitness
from controller.cpg_rbfn import CPG_RBFN

#Get current directory
CWD = Path.cwd()

#Folder to save models
MODELS_PATH = f"{CWD}/models"

# ENV_TYPE = 'MountainCar-v0'
ENV_TYPE = "BipedalWalker-v3"
# ENV_TYPE = "Walker2d-v4"

ENV = gym.make(ENV_TYPE)
# ENV = gym.make(ENV_TYPE, healthy_reward=0.1, forward_reward_weight=2.0)

#CPG-RBFN Parameters
RBFN_UNITS = 20
OUTPUT_UNITS = 4

##########-- NEUROEVOLUTION --##########

#Run individuals in generation through environment
def run_gen(generation, rewards_goal, min_equal_steps):
    equal_steps = 0
    #Takes each individual and makes it play the game
    for individual in generation:

        #Reset the environment, get initial state
        state, _ = ENV.reset()

        #This is the goal you have set for the individual.
        for _ in range(rewards_goal):

            #Choose action
            action = individual.choose_action()

            next_state, reward, terminated, _, _ = ENV.step(action)

            individual.fitness += reward

            #Count how many times we are stuck on the same step
            if np.allclose(state, next_state):
                equal_steps += 1
            else:
                equal_steps = 0
            
            state = next_state

            if (equal_steps>=min_equal_steps):
                individual.fitness -= 50
                break

            if terminated:
                break

#Train through neuro evolution
def neuro_evolution(gen_size: int, generations: int, rewards_goal: int, min_equal_steps: int, elite_size: int, elite: list[Individual]=[]):

    best_per_gen = []
    best_indv = Individual(RBFN_UNITS, OUTPUT_UNITS)
    
    #Initialize first gen
    generation = []
    try:
        # Add elite if any
        for i in range(len(elite)):
            generation.append(elite[i])

        for _ in range(gen_size-len(elite)):
            new_individual = Individual(RBFN_UNITS, OUTPUT_UNITS)
            generation.append(new_individual)

        #Iterate generations
        for gen_count in range(generations):

            #Runs each individual through the sim
            run_gen(generation, rewards_goal, min_equal_steps)

            #Get fitness of current generation
            fitness_of_generation = norm_fitness_of_generation(generation)

            #Breed gen_size children
            children = []
            for _ in range(0,gen_size):

                # Select parents for breeding through roulette wheel selection
                parent = generation[roulette_wheel_selection(fitness_of_generation)]

                #Mutation
                mutate_percent = 0.1
                mutations = int(parent.model.dim * mutate_percent)

                child = Individual(RBFN_UNITS, OUTPUT_UNITS)
                child.model.set_params(mutate(parent.model.get_params(), mutations))

                children.append(child)

            #Runs each child through the biped walker sim
            run_gen(children, rewards_goal, min_equal_steps)

            #Add the breeded children to current generation
            generation.extend(children)

            #From select the best solutions up to gen_size
            generation = select_solutions_from_gen(generation, gen_size)

            #Print results
            print(f'Generation: {gen_count} Best Fitness: {generation[0].fitness}')

            best_per_gen.append(generation[0].fitness)
            best_indv = generation[0]
            elite = generation[0:elite_size]

            #Reset generation's fitness
            resetFitness(generation)
        
    except KeyboardInterrupt:
        for i in range(len(elite)):
            save(elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")
        print("Saved Models")
        sys.exit()

    ENV.close()

    return best_indv, elite, best_per_gen
 
#Run the algorithms with learned models
def test_algorithm(best_nn:Individual, episodes:int=1000, min_equal_steps:int=5):
    #Set test environment
    test_env = gym.make(ENV_TYPE, render_mode="human")

    #Reset the environment, get initial state
    state, _ = test_env.reset()

    total_rewards = 0

    for _ in range(episodes):

        #Choose action
        action = best_nn.choose_action()
        print(action)

        next_state, reward, terminated, _, _ = test_env.step(action)

        total_rewards += reward

        if np.allclose(state, next_state):
                equal_steps += 1
        else:
            equal_steps = 0
        
        state = next_state

        if (equal_steps>=min_equal_steps):
            total_rewards -= 50

        print(f"Rewards: {total_rewards}")

        if terminated:
            break

        test_env.render()

    test_env.close()

### NEUROEVOLUTION PARAMS ###
elite_size = 20
min_equal_steps = 5
rewards_goal = 500
generations = 100
gen_size = 20

### FIRST NEUROEVOLUTION RUN ###
# best_indv, elite, best_per_gen = neuro_evolution(gen_size=gen_size, generations=generations, rewards_goal=rewards_goal, min_equal_steps = min_equal_steps, elite_size=elite_size)
# for i in range(len(elite)):
#             save(elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")

### CONTINUE NEUROEVOLUTION RUN ###
# Load elite
elite = []
for i in range(elite_size):
    model = CPG_RBFN(RBFN_UNITS, OUTPUT_UNITS)
    model.load_state_dict(load(f"{MODELS_PATH}/model{i}.pth"))
    best_indv = Individual(RBFN_UNITS, OUTPUT_UNITS)
    best_indv.model = model
    elite.append(best_indv)

best_indv, new_elite, best_per_gen = neuro_evolution(gen_size=gen_size, generations=generations, rewards_goal=rewards_goal, min_equal_steps = min_equal_steps, elite_size=elite_size, elite=elite)
for i in range(len(new_elite)):
    save(new_elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")

## LOAD BEST SAVED MODEL ###
# model = CPG_RBFN(RBFN_UNITS, OUTPUT_UNITS)
# model.load_state_dict(load(f"{MODELS_PATH}/model0.pth"))
# best_indv = Individual(RBFN_UNITS, OUTPUT_UNITS)
# best_indv.model = model
# test_algorithm(best_nn=best_indv)
