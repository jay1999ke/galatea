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
        
        self.leakyRelu = nn.LeakyReLU(0.3)
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

        self.relu = nn.LeakyReLU(0.3)
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

class cGenerator(nn.Module):

    def __init__(self,z_dim=100,out_dim=784,classes=10):
        super(cGenerator,self).__init__()

        self.layer1_1 = nn.Linear(z_dim,out_dim//8)

        self.layer1_2 = nn.Linear(classes,out_dim//8)

        self.layer2 = nn.Linear(out_dim//4,out_dim//4)

        self.layer3 = nn.Linear(out_dim//4,out_dim//2)

        self.layer4 = nn.Linear(out_dim//2,out_dim)
        
        self.leakyRelu = nn.LeakyReLU(0.3)
        self.tanh = nn.Tanh()

    def forward(self,x,classes):
        x_z = self.leakyRelu(self.layer1_1(x))
        x_c = self.leakyRelu(self.layer1_2(classes))
        x = torch.cat((x_z,x_c),1)
        x = self.leakyRelu(self.layer2(x))
        x = self.leakyRelu(self.layer3(x))
        return self.tanh(self.layer4(x))

class cDiscriminator(nn.Module):

    def __init__(self,in_dim=784,classes=10):
        super(cDiscriminator,self).__init__()

        self.layer1_1 = nn.Linear(in_dim,in_dim//2)

        self.layer1_2 = nn.Linear(classes,in_dim//2)

        self.layer2 = nn.Linear(in_dim,in_dim//4)
        
        self.layer3 = nn.Linear(in_dim//4,in_dim//8)

        self.layer4 = nn.Linear(in_dim//8,1)

        self.relu = nn.LeakyReLU(0.3)
        self.sigmoid = nn.Sigmoid()
        self.dpout = nn.Dropout(0.2)

    def forward(self,x,classes):
        x_z = self.dpout(self.relu(self.layer1_1(x)))
        x_c = self.dpout(self.relu(self.layer1_2(classes)))
        x = torch.cat((x_z,x_c),1)
        x = self.dpout(self.relu(self.layer2(x)))
        x = self.dpout(self.relu(self.layer3(x)))
        return self.sigmoid(self.layer4(x))

class cGAN(nn.Module):

    def __init__(self,z_dim=100,out_dim=784,classes=10):
        super(cGAN,self).__init__()

        self.D = cDiscriminator()
        self.G = cGenerator()



