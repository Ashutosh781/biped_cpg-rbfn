from controller.cpg_rbfn import CPG_RBFN

class Individual():
    def __init__(self, rbf_size, out_size):

        self.rbf_size = rbf_size
        self.out_size = out_size

        self.model = CPG_RBFN(self.rbf_size, self.out_size)

        self.fitness = 0 #Total fitness the model gets in a game

    def choose_action(self):
        output = self.model.forward()
        return output.detach().numpy()