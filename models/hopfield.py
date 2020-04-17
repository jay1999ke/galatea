import torch

device = torch.device("cpu")
sign = torch.sign

class Hopfield(object):
    
    def __init__(self,visible=100,hidden=0,bias = False):
        
        if hidden == 0:
            self.hidden_present = False
        else:
            self.hidden_present = True

        self.hidden_units = hidden
        self.visible_units = visible

        self.total_units = hidden + visible

        self.W = torch.rand((self.total_units,self.total_units)).to(device)

        self.bias = bias
        if not bias:
            self.b = torch.rand((self.total_units,1)).to(device)
        else:
            self.b = torch.zeros((self.total_units,1)).to(device)

        self.state = torch.sign(torch.randn((self.total_units,1)).to(device))

    def energy(self):
        Wy = torch.mm(self.W,self.state)
        return - torch.mm(self.state.t(),Wy)

    def indexUpdate(self,index):
        Wj = self.W[index].view(self.W[index].shape[0],-1).t()
        self.state[index] = sign(self.b[index] + torch.mm(Wj,self.state))

    def seqUpdate(self):

        for i in range(self.total_units):
            self.indexUpdate(i)
            print(i,self.energy())