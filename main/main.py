import numpy as np
from network import Network
import math
from layer import Layer

#myNetwork = Network([2, 3, 8, 4])
#myNetwork.compute(np.array([1,2]))


myNetwork = Network([2, 4, 1])
for _ in range(1000):
    err = myNetwork.learn(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]), 0.02)
    print(err)
print(myNetwork.compute(np.array([1,0])))
print(myNetwork.compute(np.array([0,0])))
print(myNetwork.compute(np.array([1,1])))
print(myNetwork.compute(np.array([0,1])))