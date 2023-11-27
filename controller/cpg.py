import numpy as np
from torch import nn

import matplotlib.pyplot as plt

class CPG(nn.Module):
    def __init__(self):
        super(CPG, self).__init__()
        self.alpha = 1.23
        self.phi = 0.06*np.pi
        self.weights = np.array([[self.alpha * np.cos(self.phi), self.alpha * np.sin(self.phi)],[-self.alpha * np.sin(self.phi), self.alpha * np.cos(self.phi)]])
        self.activations = np.array((0, 0.2012))

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
        self.activations = np.array((0, 0.2012))

# cpg = AdaptiveSO2CPGSynPlas()
# cpg.setPhi     ( 0.02*np.pi )
# cpg.setEpsilon ( 0.1 )
# cpg.setAlpha   ( 1.01)
# cpg.setGamma   ( 1.0 )
# cpg.setBeta    ( 0.0 )
# cpg.setMu      ( 1.0 )
# cpg.setBetaDynamics   ( -1.0, 0.010, 0.00)
# cpg.setGammaDynamics  ( -1.0, 0.010, 1.00)
# cpg.setEpsilonDynamics(  1.0, 0.010, 0.01)

# cpg.setOutput(0,0.2012)
# cpg.setOutput(1,0)

# x=[]
# y=[]
# for _ in range(360):
#     x.append(cpg.getOutput(0))
#     y.append(cpg.getOutput(1))
# plt.plot(x)
# plt.plot(y)
# plt.show()

# TEST CPG
# cpg = CPG()
# x=[]
# y=[]
# for _ in range(360):
#     out=cpg.get_output()
#     x.append(out[0])
#     y.append(out[1])
#     cpg.step()
# plt.plot(x)
# plt.plot(y)
# plt.show()
