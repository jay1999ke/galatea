import torch

device = torch.device("cuda:0")

class RBM(object):

    def __init__(self,no_x=784,no_h=100,k=10,alpha=0.001,epoch=50,sampling=False):
        """
        m = no_x; x,c = (m,1)
        n = no_h; h,b = (n,1)
                ; W   = (n,m)
        """
        self.m = no_x
        self.no_x = no_x
        self.n = no_h
        self.no_h = no_h
        self.epoch = epoch
        m = no_x
        n = no_h

        self.W = torch.load("weights/rbm/W100s.pt").to(device)
        self.c = torch.load("weights/rbm/c100s.pt").to(device)
        self.b = torch.load("weights/rbm/b100s.pt").to(device)
        self.k = k
        self.alpha = alpha
        self.sampling = sampling

    def energy(self,x,h):
        return -(torch.mm(torch.mm(h.t(),self.W),x) + torch.mm(self.c.t(),x) + torch.mm(self.b.t(),h))

    def _h(self,x):
        return torch.sigmoid(self.b + torch.mm(self.W,x))

    def _x(self,h):
        return torch.sigmoid(self.c + torch.mm(self.W.t(),h))

    def reconstruction(self,x):
        h = self._h(x)
        if self.sampling:
            h = (h >= torch.rand(h.shape).to(device)).float()
        x_r = self._x(h)
        return x_r


    def CDk(self,dataloader):
        print("inside the beast")
        for j in range(self.epoch):
            e = 0
            for i,(x,_) in enumerate(dataloader):
                if torch.Size([784,0]) == x.shape:
                    break
                
                h_probs = self._h(x)
                positive_h = h_probs
                positive_phase = torch.mm(h_probs,x.t())

                if self.sampling:
                    h_act = (h_probs >= torch.rand(h_probs.shape).to(device)).float()

                for t in range(self.k):
                    if not self.sampling:
                        x_probs = self._x(h_probs)
                        h_probs = self._h(x_probs)
                    else:
                        x_probs = self._x(h_act)
                        h_probs = self._h(x_probs)
                        h_act = (h_probs >= torch.rand(h_probs.shape).to(device)).float()

                if self.sampling:
                    negative_phase = torch.mm(h_act,x_probs.t())
                    negative_h = h_act
                else:
                    negative_phase = torch.mm(h_probs,x_probs.t())
                    negative_h = h_probs

                self.W+= self.alpha * ( positive_phase - negative_phase)
                self.b+= self.alpha * ( positive_h - negative_h ).sum()/100
                self.c+= self.alpha * ( x - x_probs).sum()/100

                e+= (abs( x - x_probs)).sum().detach()
                if i%100==0 and i!=0:
                    print("MB:",i,e/(i*100+1))
            print("\nEP:",j,e.sum()/len(dataloader)/100,"\t",self.k,"\n")



            torch.save(self.W,"weights/rbm/W100.pt")
            torch.save(self.b,"weights/rbm/b100.pt")
            torch.save(self.c,"weights/rbm/c100.pt")


class RBMClassifier(torch.nn.Module):

    def __init__(self,RBM,classes=10):
        super(RBMClassifier,self).__init__()

        self.RBM = RBM

        self.classifier = torch.nn.Linear(RBM.no_h,classes*5)
        self.classifier2 = torch.nn.Linear(classes*5,classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        
        x = self.RBM._h(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        x = self.classifier2(x)
        return self.sigmoid(x)