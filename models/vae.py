import torch
from torch import nn

device = torch.device("cuda:0")

class NegativeELBO(nn.Module):

    def __init__(self):
        super(NegativeELBO,self).__init__()

    def forward(self,x,x_hat,mean,log_variance):

        divergence = - (1+ log_variance - mean**2 - log_variance.exp()).sum()/2
        reconstruction_error = ((x-x_hat)**2).sum()

        return divergence + reconstruction_error

class Encoder(nn.Module):

    def __init__(self,inp_dim=784,z_dim=32):
        super(Encoder,self).__init__()
        self.relu = nn.ReLU()
        
        self.layer1 = nn.Linear(inp_dim,256)
        self.layer2 = nn.Linear(256,144)

        self.mean = nn.Linear(144,32)
        self.log_variance = nn.Linear(144,32)

    def forward(self,x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        mean = self.mean(x)
        log_var = self.log_variance(x)

        return mean,log_var

class Decoder(nn.Module):

    def __init__(self,out_dim=784,z_dim=32):
        super(Decoder,self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.layer1 = nn.Linear(z_dim,144)
        self.layer2 = nn.Linear(144,256)
        self.layer3 = nn.Linear(256,out_dim)

    def forward(self,x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))

        return x


class VAE(nn.Module):

    def __init__(self,inp_dim=784,z_dim=32):
        super(VAE,self).__init__()

        self.encoder = Encoder(inp_dim=inp_dim,z_dim=z_dim)
        self.decoder = Decoder(out_dim=inp_dim,z_dim=z_dim)

    def reparameterize(self,mean,log_variance):

        deviation = torch.exp(log_variance/2)
        eta = torch.randn(deviation.shape,
            device=deviation.device,
            layout=deviation.layout,
            dtype=deviation.dtype
            )  

        z = mean + deviation*eta
        return z

    def forward(self,x):
        mean,log_variance = self.encoder(x)
        z = self.reparameterize(mean,log_variance)
        self.z_dim = z.shape
        out = self.decoder(z)
        return out,mean,log_variance

    def generate(self,count=1,device=device):
        random = torch.randn(self.z_dim)
        image = self.decoder(random)
        return image
