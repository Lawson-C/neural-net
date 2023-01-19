import json
import numpy as np
from random import random

class MNet:
    def __init__(self, json_src=None):
        if json_src is None:
            self.bias = -float(1.0)
            self.layers = [None] * 3
            self.layers[0] = np.matrix([[float(0.0)] * 2]) # input layer
            self.layers[1] = np.matrix([[-self.bias] * 2]) # layers have 1 row
            self.layers[2] = np.matrix([[float(0.0)] * 2]) # output layer
            self.weight = [None] * (len(self.layers) - 1) # weights only exist between layers
            self.weight = [ np.matrix([ [ random() for a in range(self.layers[i + 1].shape[1]) ] ] for b in range(self.layers[i].shape[1])) for i in range(len(self.weight)) ] # of rows = # of neurons in layer x, # of columns = # of neurons in layer y, shape = (# of rows, # of columns)
        else:
            self.load(json_src)

    def interpret(self, inp):
        self.layers[0] += np.matrix([inp])
        p = MNet.msigmoid(self.x() * self.w())
        self.layers[1] += p
        q = MNet.msigmoid(self.y() * self.v())
        self.layers[2] += q
        outp = self.z().A.tolist()[0]
        self.wipe()
        return outp
    
    def accomodate(self, inp, expected):
        outp = self.interpret(inp)
        performance = [ None ] * len(outp)
        dpdq = [ (expected[i] - outp[i]) * self.z(i) * (1 - self.z(i)) for i in range(self.z().shape[1]) ];
        dpdv = np.matrix([ [ dpdq[i] * self.y(j) for i in range(self.z().shape[1]) ] for j in range(self.y().shape[1]) ])
        for i in range(len(outp)):
            performance[i] = -.5*(outp[i] - expected[i])**2
            dpdw = np.matrix([ [ dpdq[i] * (1 - self.z(i)) * self.v(j, i) * self.y(j) * (1 - self.y(j)) * self.x(k) for j in range(self.y().shape[1]) ] for k in range(self.x().shape[1]) ])
            learning_rate = 1
            #self.weight[0] += dpdw * learning_rate
        self.weight[1] += dpdv * learning_rate
        return performance

    def x(self, i=None):
        return self.layers[0].item((0, i)) if i is not None else self.layers[0]
    
    def w(self, a=None, b=None):
        return self.weight[0].item((a, b)) if a is not None and b is not None else self.weight[0]

    def y(self, i=None):
        return self.layers[1].item((0, i)) if i is not None else self.layers[1]
    
    def v(self, a=None, b=None):
        return self.weight[1].item((a, b)) if a is not None and b is not None else self.weight[1]
    
    def z(self, i=None):
        return self.layers[2].item((0, i)) if i is not None else self.layers[2]

    def reset(self):
        self.wipe()
        self.weight = [ np.matrix([ [ random() for a in range(self.layers[i + 1].shape[1]) ] ] for b in range(self.layers[i].shape[1])) for i in range(len(self.weight)) ]

    def wipe(self):
        self.layers[0] = np.matrix([[float(0)] * 2])
        self.layers[1] = np.matrix([[-self.bias] * 2])
        self.layers[2] = np.matrix([[float(0)] * 2])

    def to_json(self):
        out = {
            'layers':[self.x().A.tolist(), self.y().A.tolist(), self.z().A.tolist()],
            'weights':[self.w().A.tolist(), self.v().A.tolist()],
            'bias':self.bias
        }
        return out

    def save(self, src=None):
        with open('matrix_net.json' if src is None else src, 'w') as file:
            json.dump(self.to_json(), file)
        file.close()

    def load(self, src):
        with open(src, 'r') as file:
            data = json.load(file)
        self.layers = [ None ] * len(data['layers'])
        for i in range(len(data['layers'])):
            self.layers[i] = np.matrix(data['layers'][i])
        self.weight = [ None ] * len(data['weights'])
        for i in range(len(data['weights'])):
            self.weight[i] = np.matrix(data['weights'][i])
        self.bias = data['bias']
        file.close()

    # runs callback function for each value in matrix 'm'
    def mforeach(m, cb):
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                m.itemset((x, y), cb(m, x, y))
        return m

    def sigmoid(x):
        if x == 0:
            return .5
        elif x <= -8:
            return 0
        elif x >= 8:
            return 1
        else:
            return 1/(1 + np.exp(-x))

    def msigmoid(m):
        def cb(m, x, y):
            return MNet.sigmoid(m.item((x,y)))
        return MNet.mforeach(m, cb)