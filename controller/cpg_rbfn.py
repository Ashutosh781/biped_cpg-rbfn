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

  def __init__(self, rbf_size, out_size, fixed_centers=False):
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
  
  def calculate_centers(self):
    # centers = np.linspace(1, self.cpg.period-1, self.rbf_kernels)
    
    # centers_1 = centers
    # centers_2 = centers

    # for i in range(self.rbf_kernels):
    #   centers_1[i] = self.cpg.signal_1[int(centers[i])]
    #   centers_2[i] = self.cpg.signal_2[int(centers[i])]
    
    #alpha 1.1, phi 0.06pi
    centers_1 = np.array([0.08393318802534816, 0.20386441733936667, 0.32457769328274216, 0.43752221920685874, 0.5320930272262056, 0.5997203797584746, 0.6372961291878165, 0.6469506305542179, 0.6331759556153739, 0.6002684567187183, 0.5514075143183255, 0.4889220062518508, 0.4148490444439277, 0.33111172524365856, 0.23919300483808598, 0.13977584631043277, 0.0328835206195805, -0.08131830870550959, -0.2011637815363302, -0.3219396024443978, -0.4351710113147538, -0.5302595212850971, -0.598542771043435, -0.636778653216397, -0.6470082479916026, -0.6337043166277948, -0.6011835792808592, -0.552650094580521, -0.4904451383888828, -0.4166082282091665, -0.3330672176326337, -0.24131983014858827])
    centers_2 = np.array([0.6185200156496393, 0.5771285620473621, 0.5210420985117409, 0.45238376841631994, 0.3731473167301331, 0.28508903989133294, 0.18930511074528117, 0.08608157723196568, -0.02462497770562129, -0.14207768047230168, -0.26322823518563443, -0.381414523210522, -0.4866842728234007, -0.568859113363609, -0.6218072191359425, -0.6452439993247057, -0.6428539770640466, -0.619254799262441, -0.5782173449354142, -0.5224336778585105, -0.4540330179541663, -0.3750111034907029, -0.2871339844821752, -0.19151614136661804, -0.088458931573381, 0.022083697127469687, 0.1394079867218818, 0.26053410083680634, 0.37888589565858016, 0.4845609994003271, 0.5673374894581381, 0.6209590783522279])

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