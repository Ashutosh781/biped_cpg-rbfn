import os
import sys
from torch import nn, Tensor, pow, exp, from_numpy, ones_like, log
import numpy as np

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import proect modules
from utils.ann_lib import Delayline

# RBF Layer
class RBF(nn.Module):
    def __init__(self, in_features, out_features, cpg_period):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.centres = nn.Parameter(Tensor(out_features, in_features))

        self.beta = 0.04

        self.tau = 300
        self.delayline = [Delayline(self.tau), Delayline(self.tau)]
        self.cpg_period = cpg_period

        # Initialize parameters
        nn.init.normal_(self.centres, -1, 1)

    def reset(self):
        nn.init.normal_(self.centres, -1, 1)

    def forward(self, input):
        self.delayline[0].Write(input[0])
        self.delayline[1].Write(input[1])

        distances = np.zeros((self.out_features, self.in_features))

        for i in range(self.out_features):
            kernel_out = exp(-(pow(self.delayline[0].Read(0) - self.centres[i,0], 2) + pow(self.delayline[1].Read(0) - self.centres[i,1], 2))/self.beta).detach().numpy()
            kernel_out_delayed = exp(-(pow(self.delayline[0].Read(int(self.cpg_period*0.5)) - self.centres[i,0], 2) + pow(self.delayline[1].Read(int(self.cpg_period*0.5)) - self.centres[i,1], 2))/self.beta).detach().numpy()

            distances[i,0] = kernel_out
            distances[i,1] = kernel_out_delayed

        for i in self.delayline:
            i.Step()

        return from_numpy(distances).float()

class SimpleRBF(nn.Module):
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

    def __init__(self, in_features, out_features, basis_func):
        super(SimpleRBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(Tensor(out_features))
        self.basis_func = basis_func

        self.beta = 0.04

        # Initialize parameters
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        distances = (input - self.centres).pow(2).sum(-1).pow(0.5) / self.beta
        return self.basis_func(distances)

# RBFs
def gaussian(alpha):
    phi = exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = ones_like(alpha) / (ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = ones_like(alpha) / (ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * log(alpha + ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - ones_like(alpha)) * exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*ones_like(alpha)) / 2*ones_like(alpha)) \
    * alpha * exp(-alpha)
    return phi

def matern32(alpha):
    phi = (ones_like(alpha) + 3**0.5*alpha)*exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*exp(-5**0.5*alpha)
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