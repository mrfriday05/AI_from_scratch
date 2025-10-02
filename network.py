import numpy as np
from layer import Layer
import math

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.layerslst=[]
        for i in range(len(self.layers)-1):
            self.layerslst.append(Layer(dimin=self.layers[i],dimout=self.layers[i+1]))
         
    def compute(self, inp):
        for layer_ in self.layerslst:
            inp=layer_.step(inp)
            print(inp)
        return inp
    
    def learn (self, inputarr, outputarr):
        for i in range (len(inputarr)):
            estimate=self.compute(inputarr)
            neurondeltas=2*(outputarr-estimate)
            for i in range(,len(layers),-1):
                neurondeltas=self.layerslst[i].layer_bpropag(neurondeltas,layerslst[i-1].last_neuron_data)
        