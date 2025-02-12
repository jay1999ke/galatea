import torch

device = torch.device("cpu")
sigmoid = torch.sigmoid

class Boltzmann(object):
    """ TODO: Clamped visible (xyz) """
    def __init__(self,visible=16,hidden=8,bias = False):
        
        if hidden == 0:
            self.hidden_present = False
        else:
            self.hidden_present = True

        self.hidden_units = hidden
        self.visible_units = visible
        self.device = device

        self.total_units = hidden + visible

        self.W = torch.zeros((self.total_units,self.total_units)).to(device)

        self.W_temp_positive = torch.zeros(self.W.shape).to(device)
        self.W_temp_negative = torch.zeros(self.W.shape).to(device)

        self.bias = bias
        if not bias:
            self.b = torch.rand((self.total_units,1)).to(device)
        else:
            self.b = torch.zeros((self.total_units,1)).to(device)

        self.state = torch.sign(torch.randn((self.total_units,1)).to(device))

    def energy(self):
        Wy = torch.mm(self.W,self.state)
        return - torch.mm(self.state.t(),Wy)

    def random_init(self):
        self.state = torch.sign(torch.randn((self.total_units,1)).to(device))
        self.state[self.state == -1] = 0

    def clear_negatives(self):
        self.W_temp_negative = torch.zeros(self.W.shape).to(device)

    def clear_positives(self):
        self.W_temp_positive = torch.zeros(self.W.shape).to(device)

    def indexUpdate(self,index,sample=False,temperature=1):
        Wj = self.W[index].view(self.W[index].shape[0],-1).t()
        probs = sigmoid((self.b[index] + torch.mm(Wj,self.state))/temperature)

        if sample:
            self.state[index] = torch.bernoulli(probs)
        else:
            self.state[index] = probs

    def seqUpdate(self):

        for i in range(self.total_units):
            self.indexUpdate(i)
            print(i,self.energy())