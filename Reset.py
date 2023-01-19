from Network import MNet

net = MNet(json_src='matrix_net.json')
net.reset()
net.save()