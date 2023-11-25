import math
import numpy as np
from torch import nn

class CPG(nn.Module):
    def __init__(self):
        super(CPG, self).__init__()
        self.alpha = 1.01
        self.phi = 0.1*np.pi
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])
        self.activations = np.random.normal(0, 0.5, 2)

    def update_weights(self):
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])

    def get_output(self):
        next_a1 = self.weights[0,0] * np.tanh(self.activations[0]) + self.weights[0,1] * np.tanh(self.activations[1])
        next_a2 = self.weights[1,0] * np.tanh(self.activations[0]) + self.weights[1,1] * np.tanh(self.activations[1])
        self.activations[0] = next_a1
        self.activations[1] = next_a2
        return self.activations

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
