import torch
from torch import nn

device = torch.device("cuda:0")

class Generator(nn.Module):

    def __init__(self,z_dim=100,out_dim=784):
        super(Generator,self).__init__()

        self.layer1 = nn.Linear(z_dim,out_dim//8)
        self.layer2 = nn.Linear(out_dim//8,out_dim//4)
        self.layer3 = nn.Linear(out_dim//4,out_dim//2)
        self.layer4 = nn.Linear(out_dim//2,out_dim)
        
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.leakyRelu(self.layer1(x))
        x = self.leakyRelu(self.layer2(x))
        x = self.leakyRelu(self.layer3(x))
        return self.tanh(self.layer4(x))

class Discriminator(nn.Module):

    def __init__(self,in_dim=784):
        super(Discriminator,self).__init__()

        self.layer1 = nn.Linear(in_dim,in_dim//2)
        self.layer2 = nn.Linear(in_dim//2,in_dim//4)
        self.layer3 = nn.Linear(in_dim//4,in_dim//8)
        self.layer4 = nn.Linear(in_dim//8,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dpout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.dpout(self.relu(self.layer1(x)))
        x = self.dpout(self.relu(self.layer2(x)))
        x = self.dpout(self.relu(self.layer3(x)))
        return self.sigmoid(self.layer4(x))


class GAN(nn.Module):

    def __init__(self,z_dim=100,out_dim=784):
        super(GAN,self).__init__()

        self.D = Discriminator()
        self.G = Generator()



