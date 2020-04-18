import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from models.hopfield import Hopfield

def train():
    model = Hopfield()
    data = torch.load("data/inf.pth").view(100,1)

    for i in range(model.total_units):
        for j in range(model.total_units):
            model.W[i][j] =  data[i]*data[j]

    eye_mask = torch.eye(100)
    eye_mask[eye_mask > 0] = 0.5
    eye_mask[eye_mask == 0] = 1
    eye_mask[eye_mask == 0.5] = 0
    model.W *= eye_mask

    return model

def digits():
    model = Hopfield()
    data = torch.load("data/digits.pth").view(10,100,1)
    data[data == 0] = -1

    for i in range(model.total_units):
        for j in range(model.total_units):

            weight = 0
            for k in range(1,7,2):
                weight+= data[k][i]*data[k][j]

            model.W[i][j] =  weight

    eye_mask = torch.eye(100)
    eye_mask[eye_mask > 0] = 0.5
    eye_mask[eye_mask == 0] = 1
    eye_mask[eye_mask == 0.5] = 0
    model.W *= eye_mask

    return model

def test():

    model = digits()
    arr = np.arange(100)
    np.random.shuffle(arr)

    for k in range(5):
        model.state = torch.sign(torch.randn((model.total_units,1)).to(model.device))
        for j in range(2):
            for i in range(100):
                model.indexUpdate(arr[i])
                E = model.energy().item()
                plt.cla()
                plt.title('Iteration: '+str(j*100+i)+' Energy : '+str(E/2))
                plt.imshow(model.state.view(10,10).numpy())
                plt.pause(0.0001)



if __name__ == "__main__":
    test()