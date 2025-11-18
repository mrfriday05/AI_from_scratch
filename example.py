import numpy as np
from neural_network.network import Network
import matplotlib.pyplot as plt
# import math
from neural_network.layer import Layer

#myNetwork = Network([2, 3, 8, 4])
#myNetwork.compute(np.array([1,2]))
fig, ax = plt.subplots()

errors=[]
N=100
xtab=[]
ytab=[]
errtab=[]
outtab=[]

myNetwork = Network([{"neurons":2, "activation":None},
                    {"neurons":6, "activation":"relu"},
                    {"neurons":5, "activation":"sigmoid"},
                    {"neurons":1, "activation":"sigmoid"}])
for i in range(13000):
    err = myNetwork.learn(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]), 1)
    if i%1000 == 0: print(f"Epoch: {i}, err: {err} ")
print(myNetwork.compute(np.array([1,0])))
print(myNetwork.compute(np.array([0,0])))
print(myNetwork.compute(np.array([1,1])))
print(myNetwork.compute(np.array([0,1])))

for x in range (N+1):
    for y in range (N+1):
        xtab.append(x/N)
        ytab.append(y/N)
        inp=np.array([x/N,y/N])
        outtab.append(myNetwork.compute(inp))


#plt.plot(errors)
sc = ax.scatter(xtab, ytab, c=outtab, cmap='viridis', s=8)
plt.colorbar(sc, label='Output')
plt.show()