from random import random
from Network import Net

grid = [[round(random()) - 1 for i in range(3)] for j in range(3)]
column1 = [grid[int(i / 3)][i % 3] for i in range(len(grid) * len(grid[0]))]

net = Net(9)

z = net.interpret(column1)

s = ''
for v in z:
    s += str(v.val) + ', '
print(s)