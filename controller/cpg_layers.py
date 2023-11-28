import numpy as np
from torch import nn

import matplotlib.pyplot as plt

class CPG(nn.Module):
    """Central Pattern Generator network for controlling the robot"""

    def __init__(self):
        super(CPG, self).__init__()
        self.alpha = 1.22
        self.phi = 0.1*np.pi
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])
        self.activations = np.array((0.2012, 0))

    def update_weights(self):
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])

    def step(self):
        next_a1 = self.weights[0,0] * np.tanh(self.activations[0]) + self.weights[0,1] * np.tanh(self.activations[1])
        next_a2 = self.weights[1,0] * np.tanh(self.activations[0]) + self.weights[1,1] * np.tanh(self.activations[1])
        self.activations[0] = next_a1
        self.activations[1] = next_a2

    def get_output(self):
        return self.activations

    def reset(self):
        self.activations = np.array((0.2012, 0))

# TEST CPG
# cpg = CPG()
# x=[]
# y=[]
# for _ in range(100):
#     out=cpg.get_output()
#     x.append(out[0])
#     y.append(out[1])
#     cpg.step()
# plt.plot(x)
# plt.plot(y)
# plt.show()
