## main\layer.py ##
import numpy as np

class Layer:
    last_neuron_data: np.ndarray
    last_raw_data: np.ndarray

    def __init__(self, dimin: int, dimout: int, activation: str = 'sigmoid'):
        self.matrix = 2 * np.random.rand(dimout, dimin) - 1
        self.dmatrix = np.zeros((dimout, dimin))
        self.bias = 2 * np.random.rand(dimout,) - 1
        self.dbias = np.zeros((dimout,))
        # Robustly handle if activation is None (for input layer)
        self.activation_fun = activation.lower() if activation else "linear"

        if self.activation_fun not in ["softmax", "linear"]:
            match self.activation_fun:
                case "relu":
                    self.vec_activation = np.vectorize(self.activation_relu)
                    self.vec_activation_derivative = np.vectorize(self.activation_derivative_relu)
                case "sigmoid":
                    self.vec_activation = np.vectorize(self.activation_sigmoid)
                    self.vec_activation_derivative = np.vectorize(self.activation_derivative_sigmoid)
                
    def step(self, vectorIn):
        v = self.matrix @ vectorIn + self.bias
        self.last_raw_data = v
        
        if self.activation_fun == "softmax":
            exps = np.exp(v - np.max(v)) # Numerically stable
            v = exps / np.sum(exps)
        elif self.activation_fun == "linear":
            pass # No activation
        else:
            v = self.vec_activation(v)
            
        self.last_neuron_data = v
        return v

    def layer_bpropag(self, neurondeltas, previous_layer_data):
        if self.activation_fun == "softmax":
            delta = neurondeltas
        elif self.activation_fun == "linear":
            delta = neurondeltas # Derivative of linear is 1
        else:
            delta = self.vec_activation_derivative(self.last_neuron_data) * neurondeltas
            
        self.dbias += delta
        self.dmatrix += np.outer(delta, previous_layer_data)
        previous_neurondeltas = np.transpose(self.matrix) @ delta
        return previous_neurondeltas
        
    def modify(self, deltat: float, n: int):
        self.matrix -= self.dmatrix * deltat / n
        self.dmatrix *= 0
        self.bias -= self.dbias * deltat / n
        self.dbias *= 0

    def activation_relu(self, num: float) -> float:
        return max(0, num)

    def activation_sigmoid(self, num: float) -> float:
        return 1 / (1 + np.exp(-num))
        
    def activation_derivative_relu(self, num: float) -> float:
        # num is the activated value
        return 1 if num > 0 else 0
        
    def activation_derivative_sigmoid(self, num: float) -> float:
        # num is the activated value
        return num * (1 - num)