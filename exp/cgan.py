import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from torch.optim import Adam
import torch
device = torch.device("cuda:0")
import matplotlib.pyplot as plt

from models.gan import cGAN
from dpipe.mnist import Mnist
import matplotlib.pyplot as plt



def train():
    

    dataloader = Mnist()

    model = cGAN()
    model.to(device)
    generator = model.G
    discriminator = model.D

    loss_fn = torch.nn.BCELoss()
    generator_optimizer = Adam(generator.parameters(),lr=0.00001,weight_decay=0.00001)
    discriminator_optimizer = Adam(discriminator.parameters(),lr=0.00002,weight_decay=0.00001)


    def train_discriminator(X,classes,posits,negits,tots):
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

        out = generator(noise,classes)

        data = torch.cat((X,out),dim=0)
        stacked_classes = torch.cat((classes,classes))

        out = discriminator(data,stacked_classes)

        posits+=out[:100,:].mean()
        negits+=out[100:,:].mean()

        loss = loss_fn(out,y)
        loss.backward()
        discriminator_optimizer.step()
        tots+=loss.detach()/200

        return posits,negits,tots

    def train_generator(tots2,classes):
        generator_optimizer.zero_grad()

        if device == torch.device("cuda:0"):
            noise = torch.cuda.FloatTensor(torch.Size((100,100)))
            torch.rand(noise.shape,out=noise)
            y = torch.cuda.FloatTensor(torch.Size((100,1)))
            torch.ones(y.shape,out=y)
        else:
            noise = torch.rand((100,100))
            y = torch.ones((100,1))

        out = discriminator(generator(noise,classes),classes)
        
        loss = loss_fn(out,y)
        loss.backward()
        generator_optimizer.step()
        tots2+=loss.detach()/100

        return tots2

    print("Start Training")

    k = 2
    classes = torch.FloatTensor(100,10).to(device)
    ###############################################################
    for epoch in range(2000):
        posits = 0
        negits = 0
        tots=0
        tots2 = 0
        for i,(X,y) in enumerate(dataloader):
            if torch.Size([784,0]) == X.shape:
                break
            
            classes.zero_()
            classes.scatter_(1,y.view(-1,1),1)

            posits,negits,tots = train_discriminator(X,classes,posits,negits,tots)

            if i%k ==0:
                tots2 = train_generator(tots2,classes)
        
            if i%100==0:
                print("MB: ",i,tots/(i*+1),"\t",tots2/(i/k+1))
        print("\nEP: ",epoch,tots/600,tots2/(600/k))
        print("Posits: ",posits/600,"\t","Negits: ",negits/600,"\n")
        
        torch.save(model.state_dict(),"weights/gan/cgan/100.pth")


def check():
    noise = torch.randn((100,100))    
    
    dataloader = Mnist()
    classes_ = dataloader[0][1].view(-1,1).cpu()
    classes = torch.FloatTensor(100,10)
    classes.zero_()
    classes.scatter_(1,classes_,1)

    model = cGAN()

    model.load_state_dict(torch.load("weights/gan/cgan/100.pth"))

    out = model.G(noise,classes)

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

