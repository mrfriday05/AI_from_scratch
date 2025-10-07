import numpy as np
from layer import Layer
import math

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.layerslst = []
        for i in range(len(self.layers) - 1):
            self.layerslst.append(Layer(dimin = self.layers[i], dimout = self.layers[i + 1]))
         
    def compute(self, inp):
        for layer_ in self.layerslst:
            inp = layer_.step(inp)
            #print(inp)
        return inp
    
    def learn(self, inputarr, outputarr, deltat):
        err = 0
        for i in range (len(inputarr)):
            estimate = self.compute(inputarr[i])
            neurondeltas = 2 * (outputarr[i] - estimate)
            err += np.sum((outputarr[i] - estimate)**2)
            for j in range(len(self.layers)-2, 0, -1):
                neurondeltas = self.layerslst[j].layer_bpropag(neurondeltas, self.layerslst[j - 1].last_neuron_data)
            neurondeltas = self.layerslst[0].layer_bpropag(neurondeltas, inputarr[i])    

        for j in range(len(self.layerslst)):
            self.layerslst[j].modify(deltat, len(inputarr)) 
        return err / len(inputarr)
    