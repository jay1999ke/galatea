import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from models.boltzmann import Boltzmann

eye_mask = torch.eye(150)
eye_mask[eye_mask > 0] = 0.5
eye_mask[eye_mask == 0] = 1
eye_mask[eye_mask == 0.5] = 0

def simulate_positive(model):
    sim_seq = np.arange(model.visible_units,model.total_units)
    np.random.shuffle(sim_seq)

    for temp_i in range(10):
        for i in range(model.hidden_units):
            model.indexUpdate(sim_seq[i],True)

def positive_phase(model,data):

    model.clear_positives()
    for pattern in range(10):
        model.random_init()
        model.state[:model.visible_units] = data[pattern]
        simulate_positive(model)

        for i in range(model.total_units):
            for j in range(model.total_units):
                model.W_temp_positive[i][j]+= (model.state[i]*model.state[j]).item()

def simulate(model):
    sim_seq = np.arange(model.total_units)
    np.random.shuffle(sim_seq)
    
    for temp_i in range(10):
        for i in range(model.total_units):
            model.indexUpdate(sim_seq[i],True)

def negative_phase(model,data):
    
    model.clear_negatives()
    for pattern in range(10):
        model.random_init()
        simulate(model)

        for i in range(model.total_units):
            for j in range(model.total_units):
                model.W_temp_negative[i][j]+= (model.state[i]*model.state[j]).item()  

def train(alpha=0.01):

    model = Boltzmann(visible=100,hidden=50)
    data = torch.load("data/digits.pth").view(10,100,1)

    for i in range(150):
        print(i,end=" ")
        positive_phase(model,data)
        negative_phase(model,data)
        model.W+= alpha*( model.W_temp_positive - model.W_temp_negative) 
        model.W *= eye_mask

        # analyze
        my_E,her_E, E_diff=0,0,0
        for j in range(10):
            model.random_init()
            model.state[:model.visible_units] = data[j]
            simulate_positive(model)
            my_E=model.energy()

            model.random_init()
            model.state[:model.visible_units] = data[j]
            simulate_positive(model)
            simulate(model)
            her_E=model.energy()
            E_diff+= her_E - my_E

        print(E_diff,my_E,her_E)


        torch.save(model.W,"weights/boltzmann/W150.pt")

    return model

def test():

    # model = train()
    model = Boltzmann(100,50)
    model.W = torch.load("weights/boltzmann/W150.pt")
    arr = np.arange(model.total_units)
    np.random.shuffle(arr)

    x = [1,0.95,0.9,0.85,0.8]
    for k in range(10):
        model.state = torch.sign(torch.randn((model.total_units,1))).to(model.device)
        model.state[model.state == -1] = 0
        for j in range(5):
            for i in range(model.total_units):
                model.indexUpdate(arr[i],True,x[j])
                E = model.energy().item()
                plt.cla()
                plt.title('Iteration: '+str(j*model.total_units+i)+' Energy : '+str(E/2))
                plt.imshow(model.state[:model.visible_units].view(10,10).numpy())
                plt.pause(0.0001)



if __name__ == "__main__":
    test()