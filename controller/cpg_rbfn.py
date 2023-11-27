import os
import sys
import numpy as np
from torch import nn, tanh, zeros, cat, from_numpy, flatten

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import proect modules
from utils.ann_lib import postProcessing
from controller.cpg import CPG
from controller.rbf_layer import RBF
from controller.motor_layer import MotorLayer

class CPG_RBFN(nn.Module):
  """CPG-RBFN network for controlling the robot"""

  def __init__(self, rbf_size, out_size):
    super(CPG_RBFN, self).__init__()
    self.in_size = 2
    self.rbf_kernels = rbf_size
    self.out_size = out_size

    self.cpg = CPG()

    #Set CPG Period
    cpg_period = 0
    tau = 300
    self.cpg_period_postprocessor = postProcessing()
    randNum = np.random.randint(0,high=tau) % tau + 1

    for _ in range(randNum):
        cpg_output = self.cpg.get_output()
        self.cpg_period_postprocessor.calculateAmplitude(cpg_output[0], cpg_output[1])
        cpg_period = self.cpg_period_postprocessor.getPeriod()
        self.cpg.step()

    self.rbfn = RBF(self.in_size, self.rbf_kernels, cpg_period)

    self.out = MotorLayer(self.rbf_kernels, self.out_size)

    self.dim = ((2 * self.rbf_kernels) + (self.rbf_kernels * self.out_size))
    self.params = zeros(self.dim)

    #Initialize weights and biases
    # nn.init.xavier_uniform_(self.out.weight)
    # nn.init.zeros_(self.out.bias)

  def set_rbf_cpg_period(self):
    cpg_output = self.cpg.get_output()
    self.cpg_period_postprocessor.calculateAmplitude(cpg_output[0], cpg_output[1])
    self.rbfn.cpg_period = self.cpg_period_postprocessor.getPeriod()

  def forward(self):
    self.set_rbf_cpg_period()
    self.cpg.step()
    z = from_numpy(self.cpg.get_output()).float()
    z = self.rbfn(z)
    z = tanh(self.out(z))
    return z

  def get_params(self):
    flatten_centers = flatten(self.rbfn.centres, start_dim=0)
    rbf_centers = nn.utils.parameters_to_vector(flatten_centers)
    out_weights = nn.utils.parameters_to_vector(self.out.weight)
    # out_bias = nn.utils.parameters_to_vector(self.out.bias)
    # return cat((rbf_centers, out_weights, out_bias), dim=0)
    return cat((rbf_centers, out_weights), dim=0)

    # out_weights = nn.utils.parameters_to_vector(self.out.weight)
    # out_bias = nn.utils.parameters_to_vector(self.out.bias)
    # return cat((out_weights, out_bias), dim=0)

  def set_params(self, params):
    in_to_rbf = self.in_size * self.rbf_kernels
    rbf_to_out = self.out_size * self.rbf_kernels

    self.rbfn.centres.data = params[0 : in_to_rbf].reshape((self.rbf_kernels, self.in_size))

    self.out.weight.data = params[in_to_rbf : in_to_rbf + rbf_to_out].reshape((self.out_size, self.rbf_kernels))
    #self.out.bias.data = params[in_to_rbf + rbf_to_out : in_to_rbf + rbf_to_out + self.out_size]

    # self.out.weight.data = params[0 : rbf_to_out].reshape((self.out_size, self.rbf_kernels))
    # self.out.bias.data = params[rbf_to_out : rbf_to_out + self.out_size]