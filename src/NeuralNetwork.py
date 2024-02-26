import random
from typing import Literal
from .Value import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(
        self, nin, nonlin=True, activationFunction: Literal["ReLU", "tanh"] = "ReLU"
    ):
        self.weights = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.activationFunction = activationFunction if nonlin == True else ""

    def __call__(self, xs: list[Value]):
        weightsAndXs = zip(self.weights, xs)
        act = sum((wi * xi for wi, xi in weightsAndXs), self.b)
        if self.activationFunction == "ReLU":
            return act.relu()
        elif self.activationFunction == "tanh":
            return act.tanh()
        else:
            return act  # if layer is linear

    def parameters(self):
        return self.weights + [self.b]


class Layer(Module):
    #     nin - number of inputs
    #     nout - number of outputs
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):
    #     nin - number of inputs
    #     nouts - number of outputs(layer size) on each level
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
