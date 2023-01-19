import json
from random import random
from Neuron import Output, Perceptron


class Net:
    def __init__(self, json_src=None, input_length=None, output_length=None):
        if json_src is None:
            if input_length is None and output_length is None:
                self.col = [[Perceptron(x=0, y=0)], [Perceptron(x=1, y=0)], [
                    Output(x=2, y=0)]]
            else:
                self.col.append([Perceptron(len(self.col), i) for i in range(int(input_length))])
                self.col.append([Perceptron(len(self.col), i) for i in range(int(input_length + output_length / 2))])
                self.col.append([Output(len(self.col), i) for i in range(int(output_length))])
            # weave net
            for i in range(len(self.col) - 1):
                for neuron in self.col[i]:
                    neuralconnections = self.col[i + 1]
                    neuron.connect(neuralconnections, [random()
                                                       for n in neuralconnections])
        else:
            self.col = []
            self.load(json_src)

    def input_layer(self):
        return self.col[0]

    def output_layer(self):
        return self.col[len(self.col) - 1]

    # running the net
    def interpret(self, input):
        for neuron in self.output_layer():
            neuron.clear()
        for i in range(len(self.input_layer())):
            self.input_layer()[i].receive(input[i])
        for column in self.col:
            if column == self.output_layer():
                break
            for neuron in column:
                neuron.fire()
        return [neuron.val + neuron.threshold for neuron in self.output_layer()]

    # adapting the net
    def accomodate(self, input, expected):
        output = self.interpret([input])[0]
        x = self.col[0][0]
        y = self.col[1][0]
        z = self.col[2][0]
        diff = output - expected
        performance = -.5 * diff * diff
        learning_rate = 10

        try:
            dPdw2 = -diff * (output * (1 - output)) * y.val
            dPdw1 = dPdw2 * y.connections[z] * (1 - y.val) * x.val
            x.connections[y] -= learning_rate * dPdw1
            y.connections[z] -= learning_rate * dPdw2
            return [dPdw1, dPdw2]
        except (ZeroDivisionError):
            do = 'nothing'
            # print('skipped')

    def to_json(self):
        out = [[]] * len(self.col)
        for x in range(len(self.col)):
            out[x] = [None] * len(self.col[x])
            for y in range(len(self.col[x])):
                out[x][y] = self.col[x][y].to_json()
        return out

    def save(self, src=None):
        with open('current_net.json' if src is None else src, 'w') as file:
            json.dump(self.to_json(), file)
        file.close()

    def load(self, src):
        with open(src, 'r') as file:
            data = json.load(file)
        self.col = [[Perceptron(x=x, y=y) for y in range(
            len(data[x]))] for x in range(len(data) - 1)]
        self.col.append([Output(x=len(data) - 1, y=y)
                        for y in range(len(self.output_layer()))])
        for x in range(len(data)):
            self.col[x] = [None] * len(data[x])
            for y in range(len(data[x])):
                self.col[x][y] = Perceptron(data[x][y])
        for x in range(len(data)):
            for y in range(len(data[x])):
                for n in data[x][y]['connections']:
                    coord = [int(num) for num in n.split(',')]
                    self.col[x][y].connect(
                        neuron=self.col[coord[0]][coord[1]], weight=data[x][y]['connections'][n])
        file.close()

    # reset weights, thresholds, and values to defaults
    def reset(self):
        self.col = [[Perceptron(x=0, y=0)], [Perceptron(x=1, y=0)], [Output(x=2, y=0)]]
        for i in range(len(self.col) - 1):
            for neuron in self.col[i]:
                neuralconnections = self.col[i + 1]
                neuron.connect(neuralconnections, [random() for n in neuralconnections])

    def __str__(self) -> str:
        s = ''
        i = 0
        maxlen = 0
        while i <= maxlen:
            for column in self.col:
                if len(column) > maxlen:
                    maxlen = len(column)
                if i >= len(column):
                    break
                s += f'({column[i].x}, {column[i].y}): {column[i].val}' + ('\n' if column == self.output_layer() else '\t')
            i += 1
        return s