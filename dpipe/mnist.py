import torch
import numpy as np
import scipy.io as mat

device = torch.device("cuda:0")

class smallMnist(object):

    def __init__(self):

        raw_data = mat.loadmat("data/mnist_reduced.mat")
        X = raw_data['X'].astype(np.float64)

        y = raw_data['y'].ravel()
        self.c_y=torch.Tensor(y).type(dtype=torch.FloatTensor)

        Y = np.zeros((5000, 10), dtype='uint8')
        for i, row in enumerate(Y):
            Y[i, y[i] - 1] = 1
        y = Y.astype(np.float64)

        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.c_y)

    def __getitem__(self,index):
        return self.X[index].view(-1,1), self.y[index], self.c_y[index]

class Mnist(object):

    def __init__(self):
        self.MODE = "train"

        raw_data = mat.loadmat("data/mnist.mat")
        self.trainX = torch.Tensor(raw_data["trainX"]/255).to(device)
        self.testX = torch.Tensor(raw_data["testX"]/255).to(device)
        self.trainY = torch.Tensor(raw_data["trainY"]).t().to(device)
        self.testY = torch.Tensor(raw_data["testY"]).t().to(device)

    def __len__(self):
        if self.MODE == "train":
            return int(60000/100)
        else:
            return int(10000/100)

    def __getitem__(self,index):
        if self.MODE == "train":
            return self.trainX[100*index:100*index+100].t(), self.trainY[100*index:100*index+100].view(-1).long()
        else:
            return self.testX[100*index:100*index+100].t(), self.testY[100*index:100*index+100].view(-1).long()
        


