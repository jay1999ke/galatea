import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from torch.optim import Adam
import torch
device = torch.device("cuda:0")
import matplotlib.pyplot as plt

from models.rbm import RBM, RBMClassifier
from dpipe.mnist import smallMnist, Mnist
import matplotlib.pyplot as plt

def modeler():
    data = Mnist()
    model = RBM()

    ks = [3,1,4,2,6,4,2,7,2,6,8,11,15,2,13,1,2,6,7,4,6]
    for k in ks:
        model.k = k
        model.CDk(data)

def recon():
    x = torch.load("weights/rbm/test.pt")
    model = RBM()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    x_new = x.clone()
    ax[0].imshow(x.view(28,28).numpy())
    
    x_new = model.reconstruction(x_new.cuda())

    ax[1].imshow(x_new.view(28,28).detach().cpu().numpy())
    plt.show()


def recons():
    data = Mnist()
    data.MODE = "test"
    model = RBM()

    fig, ax = plt.subplots(nrows=2, ncols=10)

    c = 0
    for i in range(0,100,10):
        x = data[i+2][0].t()[0]*255

        x = x.view(784,1)
        
        mask = (torch.rand(x.shape).cuda() > 0.2).float()
        x = x * mask
        x_new = x.clone()
        ax[0][c].imshow(x.view(28,28).detach().cpu().numpy())
        
        for i in range(6):
            x_new = model.reconstruction(x_new)

        x = x_new.view(28,28).detach().cpu().numpy()
        ax[1][c].imshow(x)
        c+=1
    
    plt.show()

def classify():
    data = Mnist()
    rbm = RBM()

    model = RBMClassifier(rbm).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer  = Adam(model.classifier.parameters(),lr=0.0001,weight_decay=0.00001)

    for e in range(100):
        tots = 0
        for i,(x,y) in enumerate(data):
            if torch.Size([784,0]) == x.shape:
                break
            
            out = model(x)

            loss = loss_fn(out,y.long())
            tots+=loss.item()
            loss.backward()

            optimizer.step()

            if i%50 ==0:
                print(e,i,tots/(i+1))

        torch.save(model.state_dict(),"weights/rbm/rbmclass.pth")

def test():
    data = Mnist()
    data.MODE = "test"
    rbm = RBM()

    model = RBMClassifier(rbm).to(device)
    model.load_state_dict(torch.load("weights/rbm/rbmclass.pth"))
    sm = torch.nn.Softmax()

    x,y = data[0]

    out = model(x)

    val, ind = out.max(1)

def analyze():
    w = torch.load("weights/rbm/W100s.pt").cpu()

    fig, ax = plt.subplots(nrows=10, ncols=10)

    plt.axis('off')
    i=0
    for row in ax:
        for col in row:
            col.imshow(w[i].view(28,28))
            col.set_axis_off()
            i+=1
    
    plt.show()

def analyze2():
    w = torch.load("weights/rbm/W16.pt").cpu()

    fig, ax = plt.subplots(nrows=4, ncols=4)

    plt.axis('off')
    i=0
    for row in ax:
        for col in row:
            col.imshow(w[i].view(28,28))
            col.set_axis_off()
            i+=1
    
    plt.show()




if __name__ == "__main__":
    analyze()
