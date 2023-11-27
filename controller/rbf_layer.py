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

        # centers_1 = np.array([-0.19629085003193042, -0.1803803684879207, -0.15030423182556196, -0.1082190635329925, -0.05613592095571705, 0.003603295724092634, 0.06687723768703556, 0.12579827631841214, 0.17021873540086016, 0.1937555738192009, 0.1958824414584636, 0.1798116888694056, 0.14960025950775502, 0.10739906577384264, 0.05522169395134108, -0.004574948781670868, -0.06782267707017003, -0.12657531224953125, -0.17069688838001223, -0.19390211387205736])
        # centers_2 = np.array([0.006249717362699904, 0.06958848153189617, 0.1281375358718078, 0.1717668091697633, 0.19437367595071275, 0.1964003046602903, 0.18120355848904637, 0.15172887207945684, 0.11015129843719734, 0.0584812790943683, -0.00721669178311189, -0.0704724839678656, -0.12881358922563063, -0.17213263267456702, -0.19442094097003218, -0.19618250847744553, -0.18077260364239828, -0.15111838485887752, -0.10938829351684973, -0.057593993857780745])
        # centers = np.column_stack((centers_1, centers_2))
        # self.centres = nn.Parameter(torch.from_numpy(centers))
       
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        self.beta = 0.04

        self.tau = 100
        self.delayline = [Delayline(self.tau), Delayline(self.tau)]
        self.cpg_period = cpg_period

    def reset_parameters(self):
        nn.init.normal_(self.centres, -1, 1)

    def forward(self, input):
        self.delayline[0].Write(input[0])
        self.delayline[1].Write(input[1])

        distances = np.zeros((self.out_features, self.in_features))

        for i in range(self.out_features):
            kernel_out = torch.exp(-(torch.pow(self.delayline[0].Read(0) - self.centres[i,0], 2) + torch.pow(self.delayline[1].Read(0) - self.centres[i,1], 2))/self.beta).detach().numpy()
            kernel_out_delayed = torch.exp(-(torch.pow(self.delayline[0].Read(int(self.cpg_period*0.5)) - self.centres[i,0], 2) + torch.pow(self.delayline[1].Read(int(self.cpg_period*0.5)) - self.centres[i,1], 2))/self.beta).detach().numpy()
            
            distances[i,0] = kernel_out
            distances[i,1] = kernel_out_delayed

        for i in self.delayline:
            i.Step()

        return torch.from_numpy(distances).float()

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