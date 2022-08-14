from random import random
from Network import Net, MNet
from time import sleep

def pattern(inp):
    if isinstance(inp, float):
        return round(inp)
    elif isinstance(inp, list):
        return [round(inp[i]) for i in range(len(inp))]

def normal_net():
    net = Net(json_src='current_net.json')
    for j in range(10):
        sleep(.5)
        inp = random()
        outp = net.interpret([inp])[0]
        net.accomodate(inp, pattern(inp))
        print(f'input: {inp}\toutput: {outp}\texpected: {pattern(inp)}')
        print(f'performance: {-.5*(outp - pattern(inp))**2}')
    net.save()

def matrix_net():
    net = MNet()
    for i in range(100):
        inp = [random() for k in range(len(net.x().A[0]))]
        for j in range(50):
            net.interpret(inp)
        outp = net.interpret(inp)
        performance = net.accomodate(inp, pattern(inp))
        print(f'input: {inp}\toutput: {outp}\texpected: {pattern(inp)}')
        print(f'performance: {performance}')
    net.save()

if __name__ == "__main__":
    matrix_net()