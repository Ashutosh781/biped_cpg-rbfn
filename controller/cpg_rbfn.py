from controller.cpg import CPG
from controller.torch_rbf import RBF, gaussian

from torch import nn, tanh, zeros, cat, from_numpy, flatten

class CPG_RBFN(nn.Module):
  def __init__(self, rbf_size, out_size):
    super(CPG_RBFN, self).__init__()
    self.in_size = 2
    self.rbf_kernels = rbf_size
    self.out_size = out_size

    self.cpg = CPG()
    self.rbfn = RBF(self.in_size, self.rbf_kernels, gaussian)

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