import os
import sys
import random as rand
import gym
import numpy as np
from torch import nn, tanh, relu, tensor, float32, zeros, cat, save, from_numpy, flatten, load
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import controller.torch_rbf as rbf
from controller.cpg import CPG

#Get current directory
CWD = Path.cwd()

#Folder to save models
MODELS_PATH = f"{CWD}/models"

#ENV_TYPE = 'MountainCar-v0'
ENV_TYPE = "BipedalWalker-v3"

env = gym.make(ENV_TYPE)

##########-- NEUROEVOLUTION --##########

class Network(nn.Module):
  def __init__(self, in_size, hid1_size, hid2_size, out_size):
    super(Network, self).__init__()
    self.in_size = in_size
    self.hid1_size = hid1_size
    self.hid2_size = hid2_size
    self.out_size = out_size

    self.hid1 = nn.Linear(self.in_size, self.hid1_size)
    self.hid2 = nn.Linear(self.hid1_size, self.hid2_size)
    self.out = nn.Linear(self.hid2_size, self.out_size)

    self.dim = (self.in_size * self.hid1_size  * self.hid2_size * self.out_size) + (self.hid1_size  + self.hid2_size + self.out_size)
    self.params = zeros(self.dim)

    #Initialize weights and biases
    nn.init.xavier_uniform_(self.hid1.weight)
    nn.init.zeros_(self.hid1.bias)
    nn.init.xavier_uniform_(self.hid2.weight)
    nn.init.zeros_(self.hid2.bias)
    nn.init.xavier_uniform_(self.out.weight)
    nn.init.zeros_(self.out.bias)


  def forward(self, x):
    z = relu(self.hid1(x))
    z = relu(self.hid2(z))
    z = tanh(self.out(z))
    return z

  def get_params(self):
    hid1_weights = nn.utils.parameters_to_vector(self.hid1.weight)
    hid1_bias = nn.utils.parameters_to_vector(self.hid1.bias)
    hid2_weights = nn.utils.parameters_to_vector(self.hid2.weight)
    hid2_bias = nn.utils.parameters_to_vector(self.hid2.bias)
    out_weights = nn.utils.parameters_to_vector(self.out.weight)
    out_bias = nn.utils.parameters_to_vector(self.out.bias)
    return cat((hid1_weights, hid1_bias, hid2_weights, hid2_bias, out_weights, out_bias), dim=0)

  def set_params(self, params):
    in_to_hid1 =self.in_size * self.hid1_size
    in_to_hid1_bias = in_to_hid1 + self.hid1_size

    hid1_to_hid2 = self.hid1_size * self.hid2_size
    hid1_to_hid2_bias = hid1_to_hid2 + self.hid2_size

    hid2_to_out = self.hid2_size * self.out_size
    hid2_to_out_bias = hid2_to_out + self.out_size

    self.hid1.weight.data = params[0 : in_to_hid1].reshape((self.hid1_size, self.in_size))
    self.hid1.bias.data = params[in_to_hid1 : in_to_hid1_bias]

    self.hid2.weight.data = params[in_to_hid1_bias : in_to_hid1_bias + hid1_to_hid2].reshape((self.hid2_size, self.hid1_size))
    self.hid2.bias.data = params[in_to_hid1_bias + hid1_to_hid2 : in_to_hid1_bias + hid1_to_hid2_bias]

    self.out.weight.data = params[in_to_hid1_bias + hid1_to_hid2_bias : in_to_hid1_bias + hid1_to_hid2_bias + hid2_to_out].reshape((self.out_size, self.hid2_size))
    self.out.bias.data = params[in_to_hid1_bias + hid1_to_hid2_bias + hid2_to_out : in_to_hid1_bias + hid1_to_hid2_bias + hid2_to_out_bias]

class CPG_RBFN(nn.Module):
  def __init__(self, rbf_size, out_size):
    super(CPG_RBFN, self).__init__()
    self.in_size = 2
    self.rbf_kernels = rbf_size
    self.out_size = out_size

    self.cpg = CPG()
    self.rbfn = rbf.RBF(self.in_size, self.rbf_kernels, rbf.gaussian)

    self.out = nn.Linear(self.rbf_kernels, self.out_size)

    self.dim = ((2 * self.rbf_kernels) + (self.rbf_kernels * self.out_size)) + (self.rbf_kernels + self.out_size)
    self.params = zeros(self.dim)

    #Initialize weights and biases
    nn.init.xavier_uniform_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def forward(self):
    z = from_numpy(self.cpg.get_output()).float()
    z = self.rbfn(z)
    z = tanh(self.out(z))
    return z

  def get_params(self):
    flatten_centers = flatten(self.rbfn.centres, start_dim=0)
    rbf_centers = nn.utils.parameters_to_vector(flatten_centers)
    rbf_log_sigmas = nn.utils.parameters_to_vector(self.rbfn.log_sigmas)
    out_weights = nn.utils.parameters_to_vector(self.out.weight)
    out_bias = nn.utils.parameters_to_vector(self.out.bias)
    return cat((rbf_centers, rbf_log_sigmas, out_weights, out_bias), dim=0)

  def set_params(self, params):
    in_to_rbf = self.in_size * self.rbf_kernels
    rbf_to_out = self.out_size * self.rbf_kernels

    self.rbfn.centres.data = params[0 : in_to_rbf].reshape((self.rbf_kernels, self.in_size))
    self.rbfn.log_sigmas.data = params[in_to_rbf : in_to_rbf + self.rbf_kernels]

    self.out.weight.data = params[in_to_rbf + self.rbf_kernels : in_to_rbf + self.rbf_kernels + rbf_to_out].reshape((self.out_size, self.rbf_kernels))
    self.out.bias.data = params[in_to_rbf + self.rbf_kernels + rbf_to_out : in_to_rbf + self.rbf_kernels + rbf_to_out + self.out_size]

class Individual():
    def __init__(self):
        # self.in_size = 24
        # self.hid1_size = 40
        # self.hid2_size = 40
        # self.out_size = 4

        # self.model   = Network(self.in_size, self.hid1_size, self.hid2_size, self.out_size)

        self.rbf_size = 40
        self.out_size = 4

        self.model   = CPG_RBFN(self.rbf_size, self.out_size)

        self.fitness = 0 #Total fitness the model gets in a game

    def choose_action(self, x):
        #output = self.model.forward(x)
        output = self.model.forward()
        return output.detach().numpy()[0]
        #return np.argmax(output.detach().numpy())

#Run individuals in generation through environment
def run_gen(generation, rewards_goal, min_equal_steps):
    equal_steps = 0
    #Takes each individual and makes it play the game
    for individual in generation:

        #Reset the environment, get initial state
        state, _ = env.reset()

        #This is the goal you have set for the individual.
        for _ in range(rewards_goal):

            #Choose action
            x = np.array(state, dtype=np.float32)
            x = tensor(x, dtype=float32)
            action = individual.choose_action(x)

            next_state, reward, terminated, _, _ = env.step(action)

            individual.fitness += reward

            #Count how many times we are stuck on the same step
            if np.allclose(state, next_state):
                equal_steps += 1
            else:
                equal_steps = 0

            if (equal_steps>=min_equal_steps):
                individual.fitness -= 50
                break

            if terminated:
                break

#Normalize the fitness of a generation
def norm_fitness_of_generation(generation):
    cost_of_generation = [individual.fitness for individual in generation]
    sum_of_generation_cost = sum(cost_of_generation)
    fitness_of_generation = [individual_cost/sum_of_generation_cost for individual_cost in cost_of_generation]
    return fitness_of_generation

#Returns the index of the selected parent through roulette wheel selection
def roulette_wheel_selection(fitness_of_generation):
    selected_fitness = rand.uniform(0,1)
    curr_fitness_sum = 0
    selected_idx = 0
    for idx, fitness in enumerate(fitness_of_generation):
        curr_fitness_sum += fitness
        if curr_fitness_sum > selected_fitness:
            selected_idx = idx
            break

    return selected_idx

def mutate(params, mutations = 1):
    for _ in range(mutations):
        params[rand.randrange(len(params))] += np.random.normal(0, 0.01)
    return params

#Select up to beam width best solutions from pool
def select_solutions_from_gen(generation, gen_size):

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
def resetFitness(generation):
    for i in generation:
        i.fitness = 0

#Train through neuro evolution
def neuro_evolution(gen_size=5, generations=10, rewards_goal=1000, min_equal_steps=10):

    best_per_gen = []
    best_indv = Individual()

    #Initialize first gen
    generation = []
    try:
        for _ in range(gen_size):
            new_individual = Individual()
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

                child = Individual()
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
            best_20_indv = generation[0:20]

            #Reset generation's fitness
            resetFitness(generation)
        
    except KeyboardInterrupt:
        for i in range(len(best_20_indv)):
            save(best_20_indv[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")
        print("Saved Models")
        sys.exit()

    env.close()

    return best_indv, best_20_indv, best_per_gen
 
#Run the algorithms with learned models
def test_algorithm(best_nn=None, episodes=1000):
    #Set test environment
    test_env = gym.make(ENV_TYPE, render_mode="human")

    #Reset the environment, get initial state
    state, _ = test_env.reset()

    total_rewards = 0

    for _ in range(episodes):

        #Choose action
        action = test_env.action_space.sample()

        x = np.array(state, dtype=np.float32)
        x = tensor(x, dtype=float32)
        action = best_nn.choose_action(x)

        state, reward, terminated, _, _ = test_env.step(action)

        total_rewards += reward

        if terminated:
            break

        print(f"Rewards: {total_rewards}")
        test_env.render()

    test_env.close()

# RUN NEUROEVOLUTION
best_indv, best_20_indv, best_per_gen = neuro_evolution(gen_size=20, generations=100, rewards_goal=1000, min_equal_steps = 10)
for i in range(len(best_20_indv)):
            save(best_20_indv[i].model.state_dict(), f"{MODELS_PATH}/model{i}.pth")
test_algorithm(best_nn=best_indv)

# LOAD SAVED MODEL
# model = CPG_RBFN(40, 4)
# model.load_state_dict(load(f"{MODELS_PATH}/model0.pth"))
# best_indv = Individual()
# best_indv.model = model
# test_algorithm(best_nn=best_indv)

# TEST CPG
# cpg = CPG()
# x=[]
# y=[]
# for _ in range(360):
#     out=cpg.get_output()
#     x.append(out[0])
#     y.append(out[1])
# plt.plot(x)
# plt.plot(y)
# plt.show()
