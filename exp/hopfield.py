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

def test():

    model = train()
    arr = np.arange(100)
    np.random.shuffle(arr)

    for j in range(2):
        for i in range(100):
            model.indexUpdate(arr[i])
            E = model.energy().item()
            plt.cla()
            plt.title('Iteration: '+str(j*100+i)+' Energy : '+str(E/2))
            plt.imshow(model.state.view(10,10).numpy())
            plt.pause(0.0001)

            if E == -9900:
                time.sleep(5)
                break
        if E == -9900:
            break



if __name__ == "__main__":
    test()