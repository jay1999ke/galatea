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

    def __init__(self,batch_size=100,norm="div"):
        self.MODE = "train"
        self.batch_size = batch_size
        raw_data = mat.loadmat("data/mnist.mat")

        if norm == "div":
            self.trainX = torch.Tensor(raw_data["trainX"]/255).to(device)
            self.testX = torch.Tensor(raw_data["testX"]/255).to(device)
        elif norm == "zero_mean":
            self.trainX = torch.Tensor(raw_data["trainX"]).to(device) - 127
            self.testX = torch.Tensor(raw_data["testX"]).to(device) - 127
            self.trainX/=127
            self.testX/=127
        self.trainY = torch.Tensor(raw_data["trainY"]).t().to(device)
        self.testY = torch.Tensor(raw_data["testY"]).t().to(device)

    def __len__(self):
        if self.MODE == "train":
            return int(60000/self.batch_size)
        else:
            return int(10000/self.batch_size)

    def __getitem__(self,index):
        if self.MODE == "train":
            return self.trainX[self.batch_size*index:self.batch_size*index+self.batch_size].t(), self.trainY[self.batch_size*index:self.batch_size*index+self.batch_size].view(-1).long()
        else:
            return self.testX[self.batch_size*index:self.batch_size*index+self.batch_size].t(), self.testY[self.batch_size*index:self.batch_size*index+self.batch_size].view(-1).long()
        


