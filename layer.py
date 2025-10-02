import numpy as np
import math
class Layer:
    def __init__(self,dimin,dimout, activation=0):
        self.matrix=2*np.random.rand(dimout,dimin)-1
        self.dmatrix=np.nul(dimout,dimin)
        self.bias=np.random.rand(dimout,)
        self.dbias=np.null(dimout,)
        self.last_neuron_data
        
    def step(self,vectorIn):
        v = self.matrix @ vectorIn
        v+=self.bias
        for i, num in enumerate(v):
            v[i] = self.activation(num)
        self.last_neuron_data = v
        return v

    
    def layer_bpropag(self,neurondeltas,previous_layer):
        
        for i, num in enumerate(neurondeltas):
            neurondeltas[i] = self.activation_derivative(num)
        self.dbias+=neurondeltas
        self.dmatrix += np.outer(neurondeltas,previous_layer)
        previous_neurondeltas=np.transpose(self.matrix) @ neurondeltas

        return(previous_neurondeltas)
        
    def modify(self,deltat,n):
        self.matrix += self.dmatrix*deltat/n
        self.dmatrix *= 0
        self.bias += self.dbias*deltat/n
        self.dbias *= 0

    def activation(self,num) -> float:
        return(1/(1+math.exp(-num)))
    
    def activation_derivative(self,num) -> float:
        return(num*(1-num))