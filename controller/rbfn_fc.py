import os
import sys
from torch import nn, tanh, zeros, cat, flatten

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from controller.rbf_layers import SimpleRBF, gaussian

class RBFN_FC(nn.Module):
  """RBFN_FC network for controlling the robot"""

  def __init__(self, in_size, rbf_size, out_size):
    super(RBFN_FC, self).__init__()
    self.in_size = in_size
    self.rbf_kernels = rbf_size
    self.out_size = out_size

    self.rbfn = SimpleRBF(self.in_size, self.rbf_kernels, gaussian)

    self.out = nn.Linear(self.rbf_kernels, self.out_size)

    self.dim = ((self.in_size * self.rbf_kernels) + (self.rbf_kernels * self.out_size)) + self.out_size
    self.params = zeros(self.dim)

    #Initialize weights and biases
    nn.init.xavier_uniform_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def forward(self, x):
    z = self.rbfn(x)
    z = tanh(self.out(z))
    return z

  def get_params(self):
    flatten_centers = flatten(self.rbfn.centres, start_dim=0)
    rbf_centers = nn.utils.parameters_to_vector(flatten_centers)
    out_weights = nn.utils.parameters_to_vector(self.out.weight)
    out_bias = nn.utils.parameters_to_vector(self.out.bias)
    return cat((rbf_centers, out_weights, out_bias), dim=0)

  def set_params(self, params):
    in_to_rbf = self.in_size * self.rbf_kernels
    rbf_to_out = self.out_size * self.rbf_kernels

    self.rbfn.centres.data = params[0 : in_to_rbf].reshape((self.rbf_kernels, self.in_size))

    self.out.weight.data = params[in_to_rbf : in_to_rbf + rbf_to_out].reshape((self.out_size, self.rbf_kernels))
    self.out.bias.data = params[in_to_rbf + rbf_to_out : in_to_rbf + rbf_to_out + self.out_size]