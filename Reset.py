from Network import Net

net = Net(json_src='current_net.json')
net.reset()
net.save()