import torch
import numpy as np
from types import SimpleNamespace

class Individual():
    def __init__(self, model):

        # Model to be used for the individual
        self.model = model

        # Fitness of the individual in each generation
        self.fitness = 0.0

    def choose_action(self):
        """Choose action when no state is given"""

        output = self.model.forward()

        return output.detach().numpy()

    def choose_action(self, x):
        """Choose action is state is given"""

        # Convert state to tensor
        x = np.array(x, dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float32)

        # Get action from model
        output = self.model.forward(x)

        return output.detach().numpy()

    def reset_fitness(self):
        """Reset fitness for new generation"""

        self.fitness = 0.0


class Models(SimpleNamespace):
    """Namespace for model types"""

    def __init__(self):
        self.CPG_RBFN_MODEL = "CPG-RBFN"
        self.CPG_FC_MODEL = "CPG-FC"
        self.FC_MODEL = 'FC'
        self.RL_PIBB = "RL-PIBB"