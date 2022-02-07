from math import exp


class Perceptron:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.threshold = 1
        self.val = -self.threshold
        self.connections = {}

    def connect(self, input, neuron, weight):
        self.connections[input] = (neuron, weight)

    def sigmoid(self, input):
        return 1/(1 + exp(-input))

    def receive(self, val):
        self.val += val
        if self.sigmoid(self.val) >= 0:
            self.fire(self.sigmoid(self.val))

    def fire(self, v):
        if v in self.connections:
            self.connections[v][0].receive(v * self.connections[v][1])

    def __str__(self):
        return str(self.val)
