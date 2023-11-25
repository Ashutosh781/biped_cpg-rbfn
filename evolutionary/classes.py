from controller.cpg_rbfn import CPG_RBFN

class Individual():
    def __init__(self):

        self.rbf_size = 20
        self.out_size = 4

        self.model = CPG_RBFN(self.rbf_size, self.out_size)

        self.fitness = 0 #Total fitness the model gets in a game

    def choose_action(self):
        output = self.model.forward()
        return output.detach().numpy()