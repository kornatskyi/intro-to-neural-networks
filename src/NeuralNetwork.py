import random

from Value import Value

class Neuron:
    def __init__(self, nin):
            self.weights = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)] 
            self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, xs: list[Value]):
        z = zip(self.weights, xs)
        act = sum((wi*xi for wi, xi in z), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.b]
    
class Layer:
#     nin - number of inputs
#     nout - number of outputs
    def __init__(self, nin:int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x: list[Value]):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()] 

class MLP:
    #     nin - number of inputs
    #     nouts - number of outputs(layer size) on each level
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers:list[Layer] = []
        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i + 1]))
    
    def __call__(self, xs: list[float]):
        # xvalues = []
        xvalues = xs
        
        
        # for i, x in enumerate(xs):
        #     xvalues.append(Value(x, label=f'x{i}'))
        
        for layer in self.layers:
            xvalues = layer(xvalues)
        return xvalues
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
