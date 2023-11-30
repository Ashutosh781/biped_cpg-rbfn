import numpy as np
from torch import nn
import sys
import os

import matplotlib.pyplot as plt

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.ann_lib import postProcessing

class CPG(nn.Module):
    """Central Pattern Generator network for controlling the robot"""

    def __init__(self):
        super(CPG, self).__init__()
        # self.alpha = 1.01
        # self.phi = 0.01*np.pi
        #Test 1
        self.alpha = 1.1
        self.phi = 0.06*np.pi
        #Test 2
        # self.alpha = 1.25
        # self.phi = 0.1*np.pi
        #Test 3
        # self.alpha = 1.5
        # self.phi = 0.2*np.pi

        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])
        self.activations = np.array((0.2, 0))

        #Initialize CPG
        tau = 500
        cpg_period_postprocessor = postProcessing()
        randNum = np.random.randint(0,high=tau) % tau + 1

        for _ in range(tau+randNum):
            cpg_output = self.get_output()
            cpg_period_postprocessor.calculateAmplitude(cpg_output[0], cpg_output[1])
            self.period = cpg_period_postprocessor.getPeriod()
            self.step()

        #Store signal values to calculate centers
        self.signal_1 = []
        self.signal_2 = []
        self.max_value = 0
        self.min_value = 0
        
        for _ in range(int(self.period)):
            cpg_output = self.get_output()
            self.signal_1.append(cpg_output[0])
            self.signal_2.append(cpg_output[1])
            self.step()

        self.signal_1 = np.array(self.signal_1)
        self.signal_2 = np.array(self.signal_2)
        self.max_value = self.signal_1.max()
        self.min_value = self.signal_1.min()

    def update_weights(self):
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])

    def step(self):
        next_a1 = self.weights[0,0] * np.tanh(self.activations[0]) + self.weights[0,1] * np.tanh(self.activations[1])
        next_a2 = self.weights[1,0] * np.tanh(self.activations[0]) + self.weights[1,1] * np.tanh(self.activations[1])
        self.activations[0] = next_a1
        self.activations[1] = next_a2

    def get_output(self):
        return 5*self.activations

    def reset(self):
        self.activations = np.array((0.2012, 0))

# x=[[ 0.6409,  0.3562],
#     [-0.6456, -0.6456],
#     [-0.6456, -0.6456],
#     [ 0.6454,  0.2520],
#     [ 0.3598,  0.6409],
#     [-0.6456,  0.3643],
#     [ 0.6409,  0.3531],
#     [-0.6456,  0.3481],
#     [ 0.6454, -0.6456],
#     [ 0.3357, -0.6456],
#     [-0.6456,  0.6032],
#     [ 0.5558, -0.6456],
#     [-0.6456, -0.6456],
#     [-0.6456,  0.6454],
#     [-0.6456,  0.3600],
#     [ 0.3567, -0.6456],
#     [ 0.3557, -0.6456],
#     [-0.6456,  0.2640],
#     [-0.6456, -0.4551]]

# c1 = [i[0] for i in x]
# c2 = [i[1] for i in x]
# time = np.arange(0, len(c1), dtype=int)

# cpg = CPG()
# print(cpg.max_value)
# print(cpg.min_value)
# print(cpg.period)
# plt.scatter(time, c1)
# plt.scatter(time, c2)
# plt.plot(cpg.signal_1)
# plt.plot(cpg.signal_2)
# plt.show()


# TEST CPG
# plt.rcParams["font.size"] = 20
# cpg = CPG()
# x=[]
# y=[]
# print(cpg.max_value)
# print(cpg.min_value)
# for _ in range(100):
#     out=cpg.get_output()
#     x.append(out[0])
#     y.append(out[1])
#     cpg.step()
# plt.plot(x)
# plt.plot(y)
# # plt.title(r'$\phi=0.06\pi$   $\alpha=1.1$  Amplitude=0.65')
# plt.title(r'$\phi=0.2\pi$   $\alpha=1.5$  Amplitude=1.59')

# plt.show()
