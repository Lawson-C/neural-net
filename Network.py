from Neuron import Perceptron


class Net:
    def __init__(self, l=3):
        self.col = [[Perceptron(i, j) for j in range(l)] for i in range(l)]
        # final row
        self.col.append([Perceptron(len(self.col), i)
                         for i in range(int(l/2))])
        # weave net
        for i in range(len(self.col) - 1):
            for n in self.col[i]:
                for con in range(len(self.col[i + 1])):
                    neuralconnection = self.col[i + 1][con]
                    n.connect(con, neuralconnection, 1)

    def interpret(self, input):
        for i in range(len(self.col[0])):
            self.col[0][i].receive(input[i])
        return self.col[2]
