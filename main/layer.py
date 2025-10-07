import numpy as np


class Layer:
    last_neuron_data: np.ndarray

    def __init__(self, dimin, dimout):
        self.matrix = 2 * np.random.rand(dimout, dimin) - 1
        self.dmatrix = np.zeros((dimout, dimin))
        self.bias = 2*np.random.rand(dimout,) - 1
        self.dbias = np.zeros((dimout,))
        
    def step(self, vectorIn):
        v = self.matrix @ vectorIn
        v += self.bias
        for i, num in enumerate(v):
            v[i] = self.activation(num)
        self.last_neuron_data = v
        return v

    
    def layer_bpropag(self, neurondeltas, previous_layer_data):
        delta = self.activation_derivative(self.last_neuron_data)
        self.dbias += neurondeltas * delta
        self.dmatrix += np.outer(neurondeltas * delta, previous_layer_data)
        previous_neurondeltas = np.transpose(self.matrix) @ (neurondeltas * delta)
        

        return(previous_neurondeltas)
        
    def modify(self, deltat, n):
        self.matrix -= self.dmatrix * deltat / n
        self.dmatrix *= 0
        self.bias -= self.dbias * deltat / n
        self.dbias *= 0

    def activation(self,num) -> float:
        '''if num > 0:
            return num
        else: 
            return(0)
        '''
        
        return(1 / (1 + np.exp(-num)))
    
    def activation_derivative(self,num) -> float:
        #if num >= 0:
        #    return 1
        #else: 
        #    return(0)
        return(num * (1 - num))