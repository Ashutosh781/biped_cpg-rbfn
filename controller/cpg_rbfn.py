import os
import sys
import numpy as np
from torch import nn, tanh, zeros, cat, from_numpy, flatten

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from utils.ann_lib import postProcessing
from controller.cpg_layers import CPG
from controller.rbf_layers import RBF
from controller.motor_layers import MotorLayer

class CPG_RBFN(nn.Module):
  """CPG-RBFN network for controlling the robot"""

  def __init__(self, rbf_size, out_size, fixed_centers=True):
    super(CPG_RBFN, self).__init__()
    self.in_size = 2
    self.rbf_kernels = rbf_size
    self.out_size = out_size

    self.cpg = CPG()

    self.rbfn = RBF(self.in_size, self.rbf_kernels, self.cpg.period)

    self.out = MotorLayer(self.rbf_kernels, self.out_size)

    self.dim = ((self.in_size * self.rbf_kernels) + (self.rbf_kernels * self.out_size))

    self.fixed_centers = fixed_centers
    if self.fixed_centers:
      self.calculate_centers()
      self.dim = (self.rbf_kernels * self.out_size)

    self.params = zeros(self.dim)
  
  def reset(self):
    self.rbfn.reset()
    self.out.reset()

  def calculate_centers(self):
    centers = np.linspace(1, self.cpg.period, self.rbf_kernels)

    centers_1 = []
    centers_2 = []

    for i in range(self.rbf_kernels):
      centers_1.append(self.cpg.signal_1_one_period[int(centers[i])]) 
      centers_2.append(self.cpg.signal_2_one_period[int(centers[i])]) 

    centers = np.column_stack((centers_1, centers_2))
    self.rbfn.centres = nn.Parameter(from_numpy(centers))

  def forward(self):
    z = from_numpy(self.cpg.get_output()).float()
    z = self.rbfn(z)
    z = tanh(self.out(z))
    self.cpg.step()
    return z

  def get_params(self):
    if self.fixed_centers:
      out_weights = nn.utils.parameters_to_vector(self.out.weight)
      return out_weights
    else:
      flatten_centers = flatten(self.rbfn.centres, start_dim=0)
      rbf_centers = nn.utils.parameters_to_vector(flatten_centers)
      out_weights = nn.utils.parameters_to_vector(self.out.weight)
      return cat((rbf_centers, out_weights), dim=0)

  def set_params(self, params):
    in_to_rbf = self.in_size * self.rbf_kernels
    rbf_to_out = self.out_size * self.rbf_kernels

    if self.fixed_centers:
      self.out.weight.data = params[0 : rbf_to_out].reshape((self.out_size, self.rbf_kernels))
    else:
      nparams = params.detach().numpy()
      for i in range(in_to_rbf):
        if nparams[i] > self.cpg.max_value:
          nparams[i] = self.cpg.max_value
        elif nparams[i] < self.cpg.min_value:
          nparams[i] = self.cpg.min_value

      self.rbfn.centres.data = params[0 : in_to_rbf].reshape((self.rbf_kernels, self.in_size))
      self.out.weight.data = params[in_to_rbf : in_to_rbf + rbf_to_out].reshape((self.out_size, self.rbf_kernels))