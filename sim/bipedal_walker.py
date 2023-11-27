import os
import sys
import types
import gym
import numpy as np
from torch import save, load, float32, tensor
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from controller.fc import FC
from controller.cpg_rbfn import CPG_RBFN
from controller.cpg_fc import CPG_FC
from evolutionary.classes import Individual
from evolutionary.functions import mutate, norm_fitness_of_generation, roulette_wheel_selection, select_solutions_from_gen, resetFitness


#Get current directory
CWD = Path.cwd()


#Gym environment
ENV_TYPE = "HalfCheetah-v4"
ENV = gym.make(ENV_TYPE)

#MODEL TYPE
models = types.SimpleNamespace()
models.CPG_RBFN_MODEL = "CPG-RBFN"
models.CPG_FC_MODEL = "CPG-FC"
models.FC_MODEL = 'FC'
MODEL_TYPE = models.CPG_RBFN_MODEL
#MODEL_TYPE = models.CPG_FC_MODEL
#MODEL_TYPE = models.FC_MODEL

#Folder to save models
MODELS_PATH = f"{CWD}/models/{MODEL_TYPE}"

#CPG-RBFN Parameters
RBFN_UNITS = 25

#FC Network
FC_INPUT_UNITS = 17
FC_HID1_UNITS = 30
FC_HID2_UNITS = 30

OUTPUT_UNITS = 6

### NEUROEVOLUTION PARAMS ###
REWARDS_GOAL = 1000
GENERATIONS = 500
GEN_SIZE = 10
ELITE_SIZE = 5

##########-- NEUROEVOLUTION --##########

#Run individuals in generation through environment
def run_gen(generation, rewards_goal):
    #Takes each individual and makes it play the game
    for individual in generation:

        #Reset the environment, get initial state
        state, _ = ENV.reset()

        #This is the goal you have set for the individual.
        for _ in range(rewards_goal):

            #Choose action
            action = None
            match(MODEL_TYPE):
                case models.CPG_RBFN_MODEL | models.CPG_FC_MODEL:
                    action = individual.choose_action()
                case models.FC_MODEL:
                    x = np.array(state, dtype=np.float32)
                    x = tensor(x, dtype=float32)
                    action = individual.choose_action(x)

            state, reward, terminated, _, _ = ENV.step(action)

            individual.fitness += reward

            if terminated:
                break

#Train through neuro evolution
def neuro_evolution(gen_size: int, generations: int, rewards_goal: int, elite_size: int, elite: list[Individual]=[]):

    best_per_gen = []
    best_indv = None
    
    #Initialize first gen
    generation = []

    try:
        # Add elite if any
        for i in range(len(elite)):
            generation.append(elite[i])

        for _ in range(gen_size-len(elite)):

            model = None
            match(MODEL_TYPE):
                case models.CPG_RBFN_MODEL:
                    model = CPG_RBFN(RBFN_UNITS, OUTPUT_UNITS)
                case models.CPG_FC_MODEL:
                    model = CPG_FC(FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)
                case models.FC_MODEL:
                    model = FC(FC_INPUT_UNITS, FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)

            new_individual = Individual(model)
            generation.append(new_individual)

        #Iterate generations
        for gen_count in range(generations):

            #Runs each individual through the sim
            run_gen(generation, rewards_goal)

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

                model = None
                match(MODEL_TYPE):
                    case models.CPG_RBFN_MODEL:
                        model = CPG_RBFN(RBFN_UNITS, OUTPUT_UNITS)
                    case models.CPG_FC_MODEL:
                        model = CPG_FC(FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)
                    case models.FC_MODEL:
                        model = FC(FC_INPUT_UNITS, FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)

                child = Individual(model)
                child.model.set_params(mutate(parent.model.get_params(), mutations))

                children.append(child)

            #Runs each child through the biped walker sim
            run_gen(children, rewards_goal)

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
            if MODEL_TYPE == models.CPG_RBFN_MODEL:
                for i in generation:
                    i.model.cpg.reset()

    except KeyboardInterrupt:
        for i in range(len(elite)):
            save(elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")
        print("Saved Models")
        print(best_per_gen)
        sys.exit()

    ENV.close()

    return best_indv, elite, best_per_gen

#Run the algorithms with learned models
def test_algorithm(best_nn:Individual, episodes:int=1000):
    #Set test environment
    test_env = gym.make(ENV_TYPE, render_mode="human")

    #Reset the environment, get initial state
    state, _ = test_env.reset()

    total_rewards = 0

    for _ in range(episodes):

        #Choose action
        action = None
        match(MODEL_TYPE):
            case models.CPG_RBFN_MODEL | models.CPG_FC_MODEL:
                action = best_nn.choose_action()
            case models.FC_MODEL:
                x = np.array(state, dtype=np.float32)
                x = tensor(x, dtype=float32)
                action = best_nn.choose_action(x)
        print(action)

        state, reward, terminated, _, _ = test_env.step(action)

        total_rewards += reward

        print(f"Rewards: {total_rewards}")

        if terminated:
            break

        test_env.render()

    test_env.close()

model = None
match(MODEL_TYPE):
    case models.CPG_RBFN_MODEL:
        model = CPG_RBFN(RBFN_UNITS, OUTPUT_UNITS)
    case models.CPG_FC_MODEL:
        model = CPG_FC(FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)
    case models.FC_MODEL:
        model = FC(FC_INPUT_UNITS, FC_HID1_UNITS, FC_HID2_UNITS, OUTPUT_UNITS)

### CONTINUE NEUROEVOLUTION RUN ###

# elite = []
# for i in range(ELITE_SIZE):
#     model.load_state_dict(load(f"{MODELS_PATH}/model{i}.pth"))
#     best_indv = Individual(model)
#     best_indv.model = model
#     elite.append(best_indv)
# best_indv, new_elite, best_per_gen = neuro_evolution(gen_size=GEN_SIZE, generations=GENERATIONS, rewards_goal=REWARDS_GOAL, elite_size=ELITE_SIZE, elite=elite)
# for i in range(len(new_elite)):
#     save(new_elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")

## LOAD BEST SAVED MODEL ###

model.load_state_dict(load(f"{MODELS_PATH}/model0.pth"))
print(model.state_dict())
best_indv = Individual(model)
best_indv.model = model
test_algorithm(best_nn=best_indv)

### FIRST NEUROEVOLUTION RUN ###

# best_indv, elite, best_per_gen = neuro_evolution(gen_size=GEN_SIZE, generations=GENERATIONS, rewards_goal=REWARDS_GOAL, elite_size=ELITE_SIZE)
# for i in range(len(elite)):
#             save(elite[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")
# print(best_per_gen)