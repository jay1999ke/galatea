import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from models.boltzmann import Boltzmann

def train():
    model = Boltzmann()
    data = torch.load("data/digits.pth").view(10,100,1)
    data[data == 0] = -1

    # TODO
    # learn

    return model

def test():

    model = train()
    arr = np.arange(100)
    np.random.shuffle(arr)

    for k in range(1):
        model.state = torch.bernoulli(torch.sigmoid(torch.randn((model.total_units,1))).to(model.device))
        for j in range(2):
            for i in range(100):
                model.indexUpdate(arr[i],True)
                E = model.energy().item()
                plt.cla()
                plt.title('Iteration: '+str(j*100+i)+' Energy : '+str(E/2))
                plt.imshow(model.state.view(10,10).numpy())
                plt.pause(0.0001)



if __name__ == "__main__":
    test()