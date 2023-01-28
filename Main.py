from random import random
from Neural_Networks import MNet
from time import sleep

def avg(inp):
    return [ int(inp[0] >= .5), int(inp[1] >= .5) ]

def run_net(net, pattern):
    performances = []
    for i in range(20):
        for j in range(50):
            inp = [ random() for k in range(net.x().shape[1]) ]
            outp = net.interpret(inp, pattern(inp))
            performance = outp[1][0]
            performances.append(performance)
        # only print every 50
        print(f'input: {inp}\toutput: {outp[0]}\texpected: {pattern(inp)}')
        print(f'performance: {performance}\t\tdiff: {outp[1][1]}')
    net.save()
    return performances

def run_mnet():
    net = MNet(json_src="Neural_Networks\\matrix_net.json")
    return run_net(net, pattern=avg)

if __name__ == "__main__":
    run_mnet()