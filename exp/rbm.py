import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from models.rbm import RBM
from dpipe.mnist import smallMnist, Mnist
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = Mnist()

    model = RBM()

    model.CDk(data)

    # for i in range(0,600,60):
    #     x = data[i][0].t()[0]

    #     x[580:,]=0
    #     x = x.view(784,1)
    #     print(x.shape)
    #     x_new = x.clone()
    #     plt.imshow(x.view(28,28).detach().cpu().numpy())
    #     plt.show()
    #     for i in range(2):
    #         x_new = model.reconstruction(x_new)

    #     x = x_new.view(28,28).detach().cpu().numpy()
    #     plt.imshow(x)
    #     plt.show()
