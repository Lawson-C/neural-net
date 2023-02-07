from aifc import Error
from math import exp
from operator import ne
from random import random


class Perceptron:
    def __init__(self, json_src=None, x=None, y=None):
        if json_src is not None:
            self.x = json_src['x']
            self.y = json_src['y']
            self.threshold = json_src['threshold']
            self.val = json_src['val']
            self.connections = {}
        else:
            self.x = x
            self.y = y
            self.threshold = .5
            self.val = -self.threshold
            self.connections = {}

    def connect(self, neuron, weight):
        if isinstance(neuron, list) and isinstance(weight, list):
            for i in range(len(neuron)):
                self.connections[neuron[i]] = weight[i]
        else:
            self.connections[neuron] = weight

    def sigmoid(self=None, input=None):
        if input == None and self != None:
            input = self.val
        try:
            return 1/(1 + exp(-input))
        except (OverflowError):
            return int(input >= 0)

    def receive(self, val):
        self.val += val

    def fire(self):
        for neuron in self.connections:
            neuron.receive(self.sigmoid(self.val * self.connections[neuron]))
        self.clear()
    
    def clear(self):
        self.val = -self.threshold

    def __str__(self):
        return str(self.val)

    def to_json(self):
        out = {
            'x': self.x,
            'y': self.y,
            'threshold': self.threshold,
            'val': self.val
        }
        out['connections'] = {}
        for i in self.connections:
            out['connections'][f'{i.x},{i.y}'] = self.connections[i]
        return out

class Output (Perceptron):
    def __init__(self, json_src=None, x=None, y=None):
        super().__init__(json_src, x, y)
        self.threshold = 0
        self.val = 0
        self.connections = {}

    def connect(self, neuron, weight):
        raise Exception('Output neurons cannot be connected to anything')
    
    def fire(self):
        raise Exception('Output neurons cannot fire')
    
    def clear(self):
        self.val = 0
