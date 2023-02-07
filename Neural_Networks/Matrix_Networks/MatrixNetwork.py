import json
import numpy as np
from random import random

class MNet:
    def __init__(self, json_src=None):
        if json_src is None:
            self.bias = float(1.0)
            self.layers = [ None ] * 3
            self.layers[0] = np.matrix([ [ float(0.0) ] * 2 ]) # input layer
            self.layers[1] = np.matrix([ [ -self.bias ] * 16 ]) # layers have 1 row
            self.layers[2] = np.matrix([ [ -self.bias ] * 2 ]) # output layer
            self.weight = [ None ] * (len(self.layers) - 1) # weights only exist between layers
            self.weight = [ np.matrix( [ [ random() for a in range(self.layers[i + 1].shape[1]) ] for b in range(self.layers[i].shape[1]) ] ) for i in range(len(self.weight)) ] # of rows = # of neurons in layer i, # of columns = # of neurons in layer i + 1, shape = (# of rows, # of columns)
        else:
            self.load(json_src)

    def interpret(self, inp, expected=None):
        self.layers[0] += np.matrix([ inp ])
        p = self.x() * self.w()
        self.layers[1] = MNet.msigmoid(self.y() + p)
        q = self.y() * self.v()
        self.layers[2] = MNet.msigmoid(self.z() + q)
        outp = self.z().A.tolist()[0]
        if not expected is None:
            ze = np.matrix([expected])
            diff = ze - self.z()
            performance = -.5 * (diff * diff.transpose()).item(0, 0)
            dPdq = [ -diff.item(0, i) * self.z(i) * (1 - self.z(i)) for i in range(self.z().shape[1]) ]
            dPdv = np.matrix([ [ dPdq[i] * self.y(j) for i in range(self.z().shape[1]) ] for j in range(self.y().shape[1]) ])
            dPdy = [ 0 ] * self.y().shape[1]
            for j in range(len(dPdy)):
                for i in range(len(dPdq)):
                    dPdy[j] += dPdq[i] * self.v(j, i)
            dPdw = np.matrix([ [ dPdy[j] * self.y(j) * (1 - self.y(j)) * self.x(k) for j in range(self.y().shape[1]) ] for k in range(self.x().shape[1]) ])
            learning_rate = .001
            self.weight[0] -= dPdw * learning_rate
            self.weight[1] -= dPdv * learning_rate
            self.wipe()
            return [ outp, [ performance, diff, dPdw, dPdv ] ]
        self.wipe()
        return outp

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
        self.layers[0] = np.matrix([ [ float(0) ] * self.x().shape[1] ])
        self.layers[1] = np.matrix([ [ -self.bias ] * self.y().shape[1] ])
        self.layers[2] = np.matrix([ [ -self.bias ] * self.z().shape[1] ])

    def to_json(self):
        out = {
            'layers':[ self.x().A.tolist(), self.y().A.tolist(), self.z().A.tolist() ],
            'weights':[ self.w().A.tolist(), self.v().A.tolist() ],
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
        elif x <= -20:
            return 0
        elif x >= 20:
            return 1
        else:
            return 1/(1 + np.exp(-x))

    def msigmoid(m):
        def cb(m, x, y):
            return MNet.sigmoid(m.item((x,y)))
        return MNet.mforeach(m, cb)