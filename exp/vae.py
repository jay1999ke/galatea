import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from torch.optim import Adam
import torch
device = torch.device("cuda:0")
import matplotlib.pyplot as plt

from models.vae import VAE, NegativeELBO
from dpipe.mnist import  Mnist


def train():

    model = VAE()
    loss_fn = NegativeELBO()
    optimizer = Adam(model.parameters(),lr=0.001)
    dataloader = Mnist()
    model.to(device)

    for i in range(30):
        tots = 0
        for batch_id,(x,_) in enumerate(dataloader):
            if torch.Size([784,0]) == x.shape:
                break
            x = x.t()
            optimizer.zero_grad()
            
            out,mean,log_variance = model(x)
            
            loss = loss_fn(x,out,mean,log_variance)
            loss.backward()
            optimizer.step()

            tots+=loss.item()

            if(batch_id%50==0):
                print(batch_id,loss.item()/100,"\t",tots/(batch_id*100+1))

        print("\n",i,tots/60000,"\n")
        torch.save(model.state_dict(),"weights/vae/z25.pth")


def check():
    noise = torch.randn((100,32))    

    model = VAE()
    model.load_state_dict(torch.load("weights/vae/z25.pth"))

    out = model.decoder(noise)

    fig, ax = plt.subplots(nrows=10, ncols=10)

    plt.axis('off')
    i=0
    for row in ax:
        for col in row:
            col.imshow(out[i].view(28,28).detach().cpu().numpy(),cmap='gray')
            col.set_axis_off()
            i+=1

    plt.show()

if __name__ == "__main__":
    check()


