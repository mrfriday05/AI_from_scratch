## main\network.py ##
import numpy as np
from .layer import Layer
import json

class Network:
    def __init__(self, layers: list = []):
        self.layers = layers
        self.layerslst = []
        for i in range(len(self.layers) - 1):
            self.layerslst.append(Layer(dimin=self.layers[i]['neurons'], 
                                       dimout=self.layers[i+1]['neurons'], 
                                       activation=self.layers[i+1].get('activation')))
    
    def compute(self, inp: np.ndarray) -> np.ndarray:
        for layer_ in self.layerslst:
            inp = layer_.step(inp)
        return inp
    
    def learn(self, inputarr: np.ndarray, outputarr: np.ndarray, deltat: float) -> float:
        err = 0
        is_softmax_output = self.layerslst[-1].activation_fun == 'softmax'

        for i in range(len(inputarr)):
            estimate = self.compute(inputarr[i])
            
            if is_softmax_output:
                # For Softmax, use Cross-Entropy loss derivative: (prediction - truth)
                neurondeltas = estimate - outputarr[i]
                # Calculate Cross-Entropy Loss for reporting
                err += -np.sum(outputarr[i] * np.log(estimate + 1e-9)) 
            else:
                # For other activations, use Mean Squared Error derivative
                neurondeltas = 2 * (estimate - outputarr[i])
                err += np.sum((outputarr[i] - estimate)**2)

            # Backpropagate the gradient
            for j in range(len(self.layerslst) - 1, 0, -1):
                neurondeltas = self.layerslst[j].layer_bpropag(neurondeltas, self.layerslst[j-1].last_neuron_data)
            self.layerslst[0].layer_bpropag(neurondeltas, inputarr[i])    
        
        # Update weights after processing the batch
        for layer_ in self.layerslst:
            layer_.modify(deltat, len(inputarr)) 
        
        return err / len(inputarr)
    
    def __repr__(self):
        parameters = 0
        ret = ""
        for i, layer in enumerate(self.layers):
            ret += f"Layer {i}:\n\tNeurons: {layer['neurons']}\n"
            if 'activation' in layer and layer['activation'] is not None:
                ret += f"\tActivation: {layer['activation']}\n"
            if i > 0:
                parameters += self.layerslst[i-1].matrix.size + self.layerslst[i-1].bias.size
        ret += f"Trainable parameters: {parameters}"
        return ret
    
    def get_network(self) -> dict:
        outlst=[]
        for layer in self.layerslst:
            outlst.append(layer.get_layer())
        return {"Meta" : "1.0",
                "Data" : outlst,
                "Layers" : self.layers}

    def save_network(self, filename: str):
        with open(f"networks/{filename}.json", "w") as f:
            json.dump(self.get_network(), f, indent=4)
        print(f"===== Model saved, filename: {filename}.json =====")

    def load_network(self, filename):
        with open(f"networks/{filename}.json", "r") as f:
            data = json.load(f)
        self.set_network(data)
        print (f"===== Model loaded, filename:{filename} =====")

    def set_network(self, data: dict):
        if data["Meta"] == "1.0":
            self.layers = data["Layers"]
            for layer in data["Data"]:
                layer_inst = Layer()
                layer_inst.set_layer(layer)
                self.layerslst.append(layer_inst)