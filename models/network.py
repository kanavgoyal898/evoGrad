import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
    
    def __repr__(self):
        return f'{'tanh' if self.nonlin else 'linear'} Neuron({len(self.w)})'

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def __repr__(self):
        return f'Layer of [{', '.join(str(n) for n in self.neurons)}]'

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f'MLP of [{', '.join(str(layer) for layer in self.layers)}]'

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
