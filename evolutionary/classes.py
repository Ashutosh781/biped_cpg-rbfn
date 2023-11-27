class Individual():
    def __init__(self, model):

        self.model = model

        self.fitness = 0 #Total fitness the model gets in a game

    def choose_action(self):
        output = self.model.forward()
        return output.detach().numpy()

    def choose_action(self, x=[]):
        if len(x)>0:
            output = self.model.forward(x)
        else:
            output = self.model.forward()
        return output.detach().numpy()