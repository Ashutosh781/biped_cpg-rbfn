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

        #Parameters in paper
        # self.alpha = 1.01
        # self.phi = 0.01*np.pi

        #Test 1
        self.alpha = 1.1
        self.phi = 0.06*np.pi
        self.period = 34

        #Test 2
        # self.alpha = 1.1
        # self.phi = 0.08*np.pi
        # self.period = 25

        #Test 3
        # self.alpha = 1.1
        # self.phi = 0.1*np.pi
        # self.period = 20
        

        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])
        self.tau = 300
        self.reset()

        #Store signal values to calculate centers
        self.signal_1 = []
        self.signal_2 = []

        for _ in range(self.tau):
            cpg_output = self.get_output()
            self.signal_1.append(cpg_output[0])
            self.signal_2.append(cpg_output[1])
            self.step()

        self.signal_1 = np.array(self.signal_1)
        self.signal_2 = np.array(self.signal_2)

        self.max_value = self.signal_1.max()
        self.min_value = self.signal_1.min()

        self.get_one_period()

    def update_weights(self):
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])

    def step(self):
        next_a1 = self.weights[0,0] * np.tanh(self.activations[0]) + self.weights[0,1] * np.tanh(self.activations[1])
        next_a2 = self.weights[1,0] * np.tanh(self.activations[0]) + self.weights[1,1] * np.tanh(self.activations[1])
        self.activations[0] = next_a1
        self.activations[1] = next_a2

    def get_output(self):
        return self.activations

    def get_one_period(self):
        self.signal_1_one_period = []
        self.signal_2_one_period = []
        add_to_signal = False

        for _ in range(self.tau):
            cpg_output = self.get_output()
            
            #Once we find a max value start capturing period
            if np.isclose(cpg_output[0], self.max_value, 0.001):
                add_to_signal = True
            
            #Capture period until we have captured period size samples
            if add_to_signal and len(self.signal_1_one_period) <= self.period:
                self.signal_1_one_period.append(cpg_output[0])
                self.signal_2_one_period.append(cpg_output[1])
            
            #Finish when we have captured one period
            if(len(self.signal_1_one_period) == self.period+1):
                break
            
            self.step()

    def reset(self):
        #Set activations
        self.activations = np.array((0.2, 0))

        #Initialize CPG
        # cpg_period_postprocessor = postProcessing()
        randNum = np.random.randint(0,high=self.tau) % self.tau + 1

        for _ in range(self.tau+randNum):
            # cpg_output = self.get_output()
            # cpg_period_postprocessor.calculateAmplitude(cpg_output[0], cpg_output[1])
            # self.period = cpg_period_postprocessor.getPeriod()
            self.step()

# x=[[ 0.6332, -0.0246],
#         [ 0.6003, -0.1421],
#         [ 0.4889, -0.3814],
#         [ 0.4148, -0.4867],
#         [ 0.2392, -0.6218],
#         [ 0.0329, -0.6429],
#         [-0.0813, -0.6193],
#         [-0.3219, -0.5224],
#         [-0.5303, -0.3750],
#         [-0.5985, -0.2871],
#         [-0.6470, -0.0885],
#         [-0.6337,  0.0221],
#         [-0.5527,  0.2605],
#         [-0.4166,  0.4846],
#         [-0.3331,  0.5673],
#         [-0.1421,  0.6450],
#         [ 0.0787,  0.6200],
#         [ 0.1985,  0.5793],
#         [ 0.4328,  0.4557],
#         [ 0.5974,  0.2892]]

# c1 = [i[0] for i in x]
# c2 = [i[1] for i in x]
# time = np.arange(0, len(c1), dtype=int)

# cpg = CPG()
# print(cpg.max_value)
# print(cpg.min_value)
# print(cpg.period)
# # plt.scatter(time, c1)
# # plt.scatter(time, c2)
# # plt.plot(cpg.signal_1)
# # plt.plot(cpg.signal_2)
# plt.plot(cpg.signal_1_one_period)
# plt.plot(cpg.signal_2_one_period)
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
