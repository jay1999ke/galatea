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

    # for i in range(0,5000,200):
    #     x = data[i][0]

    #     x[240:,]=0

    #     x_new = x
    #     for i in range(2):
    #         x_new = model.reconstruction(x_new)

    #     x = x_new.view(20,20).t()
    #     plt.imshow(x)
    #     plt.show()
