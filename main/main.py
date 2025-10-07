import numpy as np
from network import Network
import matplotlib.pyplot as plt
# import math
from layer import Layer

#myNetwork = Network([2, 3, 8, 4])
#myNetwork.compute(np.array([1,2]))
plt.ion()
fig, ax = plt.subplots()

errors=[]
k=100
xtab=[]
ytab=[]
errtab=[]
for j in range(k):
    myNetwork = Network([2, 3, 1])
    for i in range(7000):
        err = myNetwork.learn(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]), 1)
        #if i%1000 == 0: print(f"Epoch: {i}, err: {err} ")
        if i%100 ==0:
            errtab.append(err)
            xtab.append(myNetwork.layerslst[0].matrix[0,0])
            ytab.append(myNetwork.layerslst[0].matrix[0,1])
            sc = ax.scatter(xtab, ytab, c=errtab, cmap='viridis', s=8)
            ax.set_title(f"Run {j+1}, Epoch {i}, Err: {err:.4f}")
            ax.set_xlabel("Weight[0,0]")
            ax.set_ylabel("Weight[0,1]")
            plt.pause(0.001)
    print(myNetwork.compute(np.array([1,0])))
    print(myNetwork.compute(np.array([0,0])))
    print(myNetwork.compute(np.array([1,1])))
    print(myNetwork.compute(np.array([0,1])))

#plt.plot(errors)


for i in range(k):
    plt.scatter(xtab[i],ytab[i], c=errtab[i], cmap='viridis', s=3)
plt.colorbar(label='Value')
plt.ioff()
plt.show()
