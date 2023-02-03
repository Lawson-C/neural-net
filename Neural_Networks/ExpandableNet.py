import json
import numpy as np
from random import random

class ExpandableNet:
    def __init__(self, json_src=None):
        if json_src is None:
            self.depth = 5
            self.bias = float(1.0)
            self.layers = [ None ] * self.depth
            self.layers[0] = np.matrix([ [ float(0.0) ] * 2])
            for i in range(1, self.depth - 1):
                self.layers[i] = np.matrix([ [ -self.bias ] * ExpandableNet.column_height(i, self.depth) ])
            self.layers[self.depth - 1] = np.matrix([ [ -self.bias ] * 2 ])
            self.weight = [ None ] * (self.depth - 1)
            self.weight = [ np.matrix( [ [ random() for a in range(self.layers[i + 1].shape[1]) ] for b in range(self.layers[i].shape[1]) ] ) for i in range(self.depth - 1) ]
        else:
            self.load(json_src)

    def column_height(i, leng):
        return int(10 * (1 - (i - leng/2)**2 / (5 * leng)))

    def c(self, i, j=None):
        return self.layers[i] if j is None else self.layers[i].item(0, j)

    def c_max(self, j=None):
        return self.layers[self.depth - 1] if j is None else self.layers[self.depth - 1].item(0, j)

    def interpret(self, inp, expected=None):
        # process layers
        self.layers[0] += np.matrix([ inp ])
        p = [ None ] * (self.depth - 1)
        for i in range(0, self.depth - 1):
            p[i] = self.layers[i] * self.weight[i]
            self.layers[i + 1] = ExpandableNet.sigmoid(self.layers[i + 1] + p[i])
        # gradient descent
        outp = self.c_max().A.tolist()[0]
        if not expected is None:
            c_expected = np.matrix([expected])
            diff = c_expected - self.c_max()
            performance = -.5 * (diff * diff.transpose()).item(0, 0)
            # partial derivatives
            dPdp = [ [ None ] * p[i].shape[1] for i in range(self.depth - 1) ]
            dPdp_max = [ -diff.item(0, i) * self.c_max(i) * (1 - self.c_max(i)) for i in range(self.c_max().shape[1]) ]
            dPdp[self.depth - 2] = dPdp_max
            for i in reversed(range(self.depth - 2)):
                for k in range(p[i].shape[1]):
                    def cb(j, v):
                        return v * self.weight[i + 1].item(k, j)
                    dPdp[i][k] = sum(ExpandableNet.aforeach(dPdp[i + 1], cb)) * self.c(i + 1, k) * (1 - self.c(i + 1, k))
            dPdw = [ np.matrix([ [ dPdp[i][k] * self.c(i, l) for k in range(self.weight[i].shape[1]) ] for l in range(self.weight[i].shape[0]) ]) for i in range(self.depth - 1) ]
            for i in range(len(self.weight)):
                self.weight[i] -= dPdw[i] * .1
            self.wipe()
            return [ outp, [ performance, diff ] ]
        self.wipe()
        return outp

    def wipe(self):
        self.layers[0] = np.matrix([ [ float(0.0) ] * self.layers[0].shape[1] ])
        for i in range(1, self.depth):
            self.layers[i] = np.matrix([ [ -self.bias ] * self.layers[i].shape[1] ])

    def reset(self):
        self.weight = [ np.matrix( [ [ random() for a in range(self.layers[i + 1].shape[1]) ] for b in range(self.layers[i].shape[1]) ] ) for i in range(self.depth - 1) ]

    def to_json(self):
        out = {
            'layers':[ column.A.tolist() for column in self.layers ],
            'weights':[ w.A.tolist() for w in self.weight],
            'bias':self.bias
        }
        return out

    def save(self, src=None):
        with open('Neural_Networks\\expandable_net.json' if src is None else src, 'w') as file:
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
        self.depth = len(self.layers)
        file.close()

    def aforeach(a, cb):
        out = [ None ] * len(a)
        for i in range(len(a)):
            out[i] = cb(i, a[i])
        return out


    def mforeach(m, cb):
        out = np.matrix([ [ None ] * m.shape[1] ] * m.shape[0])
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                out.itemset((x, y), cb(m, x, y))
        return out

    def sigmoid(x):
        if isinstance(x, np.matrix):
            def cb(m, x, y):
                return ExpandableNet.sigmoid(m.item((x,y)))
            return ExpandableNet.mforeach(x, cb)
        elif isinstance(x, float):
            if x == 0:
                return .5
            elif x <= -20:
                return 0
            elif x >= 20:
                return 1
            else:
                return 1/(1 + np.exp(-x))
