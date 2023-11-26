import torch
import torch.nn as nn
import numpy as np

# Motor Layer
class MotorLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MotorLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, -1, 1)

    def forward(self, input):

        output = np.zeros(self.out_features)

        for i in range(self.in_features):
            
            output[0] += self.weight[0,i] * input[i,0]#kernel_out
            output[1] += self.weight[1,i] * input[i,0]#kernel_out
            output[2] += self.weight[2,i] * input[i,1]#kernel_out_delayed
            output[3] += self.weight[3,i] * input[i,1]#kernel_out_delayed

        output = torch.from_numpy(output).float()

        return output