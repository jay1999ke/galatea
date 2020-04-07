import torch

device = torch.device("cuda:0")

class RBM(object):

    def __init__(self,no_x=784,no_h=256,k=10,alpha=0.000001,epoch=30):
        """
        m = no_x; x,c = (m,1)
        n = no_h; h,b = (n,1)
                ; W   = (n,m)
        """
        self.m = no_x
        self.n = no_h
        self.epoch = epoch
        m = no_x
        n = no_h

        self.W = torch.rand((n,m)).to(device)
        self.c = torch.rand((m,1)).to(device)
        self.b = torch.rand((n,1)).to(device)
        self.k = k
        self.alpha = alpha

    def energy(self,x,h):
        return -(torch.mm(torch.mm(h.t(),self.W),x) + torch.mm(self.c.t(),x) + torch.mm(self.b.t(),h))

    def _h(self,x):
        return torch.sigmoid(self.b + torch.mm(self.W,x))

    def _x(self,h):
        return torch.sigmoid(self.c + torch.mm(self.W.t(),h))

    def reconstruction(self,x):
        h = self._h(x)
        h_act = torch.bernoulli(h)
        x_r = self._x(h_act)
        return x_r


    def CDk(self,dataloader):
        print("inside the beast")
        for j in range(self.epoch):
            e = 0
            for i,(x,_) in enumerate(dataloader):
                if torch.Size([784,0]) == x.shape:
                    break
                h_probs = self._h(x)
                h_act = (h_probs >= torch.rand(h_probs.shape).to(device)).float()
                positive_h = h_probs
                positive_phase = torch.mm(h_probs,x.t())

                for t in range(self.k):
                    x_probs = self._x(h_act)
                    x_act = (x_probs >= torch.rand(x_probs.shape).to(device)).float()
                    h_probs = self._h(x_act)
                    h_act = (h_probs >= torch.rand(h_probs.shape).to(device)).float()

                negative_phase = torch.mm(h_probs,x_act.t())
                negative_h = h_probs

                self.W+= self.alpha * ( positive_phase - negative_phase)
                self.b+= self.alpha * ( positive_h - negative_h ).sum()
                self.c+= self.alpha * ( x - x_act).sum()

                e+= (abs( x - x_probs)).sum().detach()
                if i%100==0 and i!=0:
                    print("MB:",i,e/(i*100+1))
            print("\n",j,e.sum()/len(dataloader)/100)



            torch.save(self.W,"weights/rbm/W.pt")
            torch.save(self.b,"weights/rbm/b.pt")
            torch.save(self.c,"weights/rbm/c.pt")
    