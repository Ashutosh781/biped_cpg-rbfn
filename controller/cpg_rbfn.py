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

  def __init__(self, rbf_size, out_size, fixed_centers: bool=False, alt_cpgs: bool=False, add_noise: bool=False, test_case: int=1):
    super(CPG_RBFN, self).__init__()
    self.in_size = 2
    self.rbf_kernels = rbf_size
    self.out_size = out_size
    self.test_num = 1
    self.add_noise = add_noise

    #Set alternating CPG
    if alt_cpgs:
      self.test_num = np.random.randint(1,4)
      self.cpg = CPG(test_num=self.test_num, add_noise=add_noise)
    else:
      self.cpg = CPG(add_noise=add_noise)

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
    # centers = np.linspace(1, self.cpg.period, self.rbf_kernels)

    # centers_1 = []
    # centers_2 = []

    # for i in range(self.rbf_kernels):
    #   centers_1.append(self.cpg.signal_1_one_period[int(centers[i])])
    #   centers_2.append(self.cpg.signal_2_one_period[int(centers[i])])

    # print(f"Centers 1: {centers_1}")
    # print(f"Centers 2: {centers_2}")

    #Fixed centers
    centers_1 =  [0.6362034868378307, 0.6056403566026877, 0.4979740682026714, 0.34278407182568793, 0.2518986861411785, 0.047582298836436485, -0.18497567644460014, -0.4208777666940527, -0.5189773961920616, -0.6333793133810519, -0.6366744959633248, -0.5599629721546054, -0.4994623530128837, -0.34471522862146653, -0.15574149978948404, 0.06309760181748794, 0.18228092859889855, 0.4184656839308781, 0.5898769526814864, 0.6470831148579375]
    centers_2 = [-0.009429322716789812, -0.126091052986264, -0.366151504623559, -0.5595098201634598, -0.6164959592472207, -0.6444954068298554, -0.5845964621957886, -0.4638175755027142, -0.3861018362300865, -0.2047054462852167, 0.006908954005284393, 0.24434699815917255, 0.36358723244028845, 0.5579006565210938, 0.6434105936053639, 0.6241561574364519, 0.585633816717373, 0.4654296411586814, 0.3013419928663006, 0.10499597417593949]

    centers = np.column_stack((centers_1, centers_2))
    self.rbfn.centres = nn.Parameter(from_numpy(centers))

  def forward(self):
    z = from_numpy(self.cpg.get_output(ignore_noise=(not self.add_noise))).float()
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