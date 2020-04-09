import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from torch.optim import Adam
import torch
device = torch.device("cuda:0")
import matplotlib.pyplot as plt

from models.gan import GAN
from dpipe.mnist import Mnist
import matplotlib.pyplot as plt

def train():
    k = 5

    dataloader = Mnist()

    model = GAN()
    model.to(device)
    generator = model.G
    discriminator = model.D

    loss_fn = torch.nn.BCELoss(reduction="mean")
    generator_optimizer = Adam(generator.parameters(),lr=0.001,weight_decay=0.00001)
    discriminator_optimizer = Adam(discriminator.parameters(),lr=0.001,weight_decay=0.00001)

    print("Start Training")
    for epoch in range(10):

        tots=0
        tots2 = 0
        for i,(X,y) in enumerate(dataloader):
            if torch.Size([784,0]) == X.shape:
                break
            discriminator_optimizer.zero_grad()
            X = X.t()
            if device == torch.device("cuda:0"):
                noise = torch.cuda.FloatTensor(torch.Size((100,100)))
                torch.rand(noise.shape,out=noise)
                y = torch.cuda.FloatTensor(torch.Size((200,1)))
                torch.ones(y.shape,out=y)
            else:
                noise = torch.rand((100,100))
                y = torch.ones((100,1),out=y)

            y[100:] = 0
    
            out = generator(noise)
            data = torch.cat((X,out),dim=0)

            out = discriminator(data)

            loss = loss_fn(out,y)
            loss.backward()
            discriminator_optimizer.step()
            tots+=loss.detach()

            if i%k ==0:
                generator_optimizer.zero_grad()

                if device == torch.device("cuda:0"):
                    noise = torch.cuda.FloatTensor(torch.Size((100,100)))
                    torch.rand(noise.shape,out=noise)
                    y = torch.cuda.FloatTensor(torch.Size((100,1)))
                    torch.zeros(y.shape,out=y)
                else:
                    noise = torch.rand((100,100))
                    y = torch.zeros((100,1),out=y)

                out = generator(noise)
                out = discriminator(out)

                loss = loss_fn(out,y)
                loss.backward()
                generator_optimizer.step()
                tots2+=loss.detach()

            if i%100==0:
                print("MB: ",i,tots,"\t",tots2)


if __name__ == "__main__":
    train()

