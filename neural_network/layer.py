## main\layer.py ##
import numpy as np

class Layer:
    last_neuron_data: np.ndarray
    last_raw_data: np.ndarray

    def __init__(self, dimin: int = 0, dimout: int = 0, activation: str = 'sigmoid', matrix: np.ndarray = None, bias: np.ndarray = None):
        if matrix is None:
            self.matrix = 2 * np.random.rand(dimout, dimin) - 1  
        else:
            self.matrix = matrix 
        self.dmatrix = np.zeros((dimout, dimin))

        if bias is None:    
            self.bias = 2 * np.random.rand(dimout,) - 1
        else:
            self.bias=bias

        self.dbias = np.zeros((dimout,))
        # Robustly handle if activation is None (for input layer)
        self.activation_fun = activation.lower() if activation else "linear"
        if self.activation_fun not in ["softmax", "linear"]:
            match self.activation_fun:
                case "relu":
                    self.vec_activation = np.vectorize(self.activation_relu, otypes=[float])
                    self.vec_activation_derivative = np.vectorize(self.activation_derivative_relu, otypes=[float])
                case "sigmoid":
                    self.vec_activation = np.vectorize(self.activation_sigmoid, otypes=[float])
                    self.vec_activation_derivative = np.vectorize(self.activation_derivative_sigmoid, otypes=[float])
                
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
    
    def get_layer(self) -> dict:
        matrix=self.matrix.tolist()
        bias=self.bias.tolist()
        return {
            "Matrix" : matrix,
            "Bias" : bias,
            "Activation" : self.activation_fun
        }

    def set_layer(self, data: dict):
        self.__init__(dimin = len(data["Matrix"][0]),
                      dimout = len(data["Matrix"]), 
                      activation = data["Activation"],
                      matrix = np.array(data["Matrix"]),
                      bias = np.array(data["Bias"]))