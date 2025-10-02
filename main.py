import numpy as np
from network import Network
import math
from layer import Layer

myNetwork = Network(2,[3, 8, 5])
myNetwork.compute(np.array([1,2]))