import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from models.boltzmann import Boltzmann

eye_mask = torch.eye(100)
eye_mask[eye_mask > 0] = 0.5
eye_mask[eye_mask == 0] = 1
eye_mask[eye_mask == 0.5] = 0

def positive_phase(model,data):

    for i in range(model.total_units):
        for j in range(model.total_units):
            weight = 0
            for k in range(10):
                weight+= data[k][i]*data[k][j]

            model.W_temp_positive[i][j] = weight

def simulate(model):
    sim_seq = np.arange(100)
    np.random.shuffle(sim_seq)
    base_state = model.state.clone()

    model.state = base_state.clone()
    for j in range(10):
        for i in range(100):
            model.indexUpdate(sim_seq[i],True)

def negative_phase(model,data):
    
    model.clear_negatives()
    for digit in range(10):
        model.state = data[digit]
        simulate(model)

        for i in range(model.total_units):
            for j in range(model.total_units):
                model.W_temp_negative[i][j]+= (model.state[i]*model.state[j]).item()  

def train(alpha=0.01):

    model = Boltzmann()
    data = torch.load("data/digits.pth").view(10,100,1)

    positive_phase(model,data)

    for i in range(100):
        print(i,end=" ")
        negative_phase(model,data)
        model.W+= alpha*( model.W_temp_positive - model.W_temp_negative) 
        model.W *= eye_mask

        # analyze
        my_E,her_E=0,0
        for j in range(10):
            model.state = data[j]
            my_E+=model.energy()/10

            simulate(model)
            her_E+=model.energy()/10

        print(my_E,her_E)


        torch.save(model.W,"weights/boltzmann/W.pt")

    return model

def test():

    model = Boltzmann()
    model.W = torch.load("weights/boltzmann/W.pt")
    arr = np.arange(100)
    np.random.shuffle(arr)

    for k in range(1):
        model.state = torch.sigmoid(torch.randn((model.total_units,1))).to(model.device)
        for j in range(10):
            for i in range(100):
                model.indexUpdate(arr[i],True)
                E = model.energy().item()
                plt.cla()
                plt.title('Iteration: '+str(j*100+i)+' Energy : '+str(E/2))
                plt.imshow(model.state.view(10,10).numpy())
                plt.pause(0.0001)



if __name__ == "__main__":
    test()