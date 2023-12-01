from torch import nn, Tensor, from_numpy

import numpy as np

# Motor Layer
class MotorLayer(nn.Module):
    """Motor Layer for controlling the robot"""

    def __init__(self, in_features, out_features):
        super(MotorLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))

        nn.init.normal_(self.weight, -1, 1)
    
    def reset(self):
        nn.init.normal_(self.weight, -1, 1)

    def forward(self, input):

        output = np.zeros(self.out_features)

        for i in range(self.in_features):
            #Biped
            # output[0] += self.weight[0,i] * input[i,0]#kernel_out
            # output[1] += self.weight[1,i] * input[i,1]#kernel_out_delayed
            # output[2] += self.weight[2,i] * input[i,0]#kernel_out
            # output[3] += self.weight[3,i] * input[i,1]#kernel_out_delayed

            #Walker 2D and Half Cheetah
            # output[0] += self.weight[0,i] * input[i,0]#kernel_out
            # output[1] += self.weight[1,i] * input[i,1]#kernel_out_delayed
            # output[2] += self.weight[2,i] * input[i,0]#kernel_out
            # output[3] += self.weight[3,i] * input[i,1]#kernel_out_delayed
            # output[4] += self.weight[4,i] * input[i,0]#kernel_out
            # output[5] += self.weight[5,i] * input[i,1]#kernel_out_delayed

            # Hind leg - gets current output
            output[0] += self.weight[0,i] * input[i,0]#kernel_out
            output[1] += self.weight[1,i] * input[i,0]#kernel_out
            output[2] += self.weight[2,i] * input[i,0]#kernel_out
            # Front leg - gets delayed output
            output[3] += self.weight[3,i] * input[i,1]#kernel_out_delayed
            output[4] += self.weight[4,i] * input[i,1]#kernel_out_delayed
            output[5] += self.weight[5,i] * input[i,1]#kernel_out_delayed

            # # Hind leg - gets current output
            # output[0] += self.weight[0,i] * input[i,0]#kernel_out
            # output[1] += self.weight[1,i] * input[i,0]#kernel_out
            # output[2] += self.weight[2,i] * input[i,0]#kernel_out
            # # Front leg - gets delayed output
            # output[3] += self.weight[3,i] * input[i,0]#kernel_out
            # output[4] += self.weight[4,i] * input[i,0]#kernel_out
            # output[5] += self.weight[5,i] * input[i,0]#kernel_out

        output = from_numpy(output).float()

        return output

class SimpleMotorLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MotorLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, -1, 1)

    def forward(self, input):

        output = np.zeros(self.out_features)

        for i in range(self.in_features):

            output[0] += self.weight[0,i] * input[0]#kernel_out
            output[1] += self.weight[1,i] * input[0]#kernel_out
            output[2] += self.weight[2,i] * input[0]#kernel_out
            output[3] += self.weight[3,i] * input[1]#kernel_out_delayed
            output[4] += self.weight[4,i] * input[1]#kernel_out_delayed
            output[5] += self.weight[5,i] * input[1]#kernel_out_delayed

        output = from_numpy(output).float()

        return output