import torch
import torch.nn as nn
import numpy as np

# Add project root to the python path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.ann_lib import Delayline

# RBF Layer
class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, cpg_period):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        self.beta = 0.04

        self.tau = 300
        self.delayline = [Delayline(self.tau), Delayline(self.tau)]
        self.cpg_period = cpg_period

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)

    def forward(self, input):
        self.delayline[0].Write(input[0])
        self.delayline[1].Write(input[1])

        c = np.zeros((self.out_features, self.in_features))

        for i in range(self.out_features):
            c[i,0] = torch.exp(-(torch.pow(self.delayline[0].Read(0) - self.centres[i,0], 2) + torch.pow(self.delayline[1].Read(0) - self.centres[i,1], 2))/self.beta).detach().numpy()
            c[i,1] = torch.exp(-(torch.pow(self.delayline[0].Read(int(self.cpg_period*0.5)) - self.centres[i,0], 2) + torch.pow(self.delayline[1].Read(int(self.cpg_period*0.5)) - self.centres[i,1], 2))/self.beta).detach().numpy()
      
        c = torch.from_numpy(c).float()
        distances = (c).sum(-1)

        for i in self.delayline:
            i.Step()

        return distances

# RBFs
def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases