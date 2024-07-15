import math

class Value:
    """ stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        # internal variables used for auto-grad graph construction
        self._backward = lambda : None
        self._prev = set(_children)
        # operator that produced this node, for debugging purposes
        self._op = _op

    # basic arithmetic operations

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supports int/float exponents'
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad = out.grad * (other * self.data**(other-1))
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward

        return out

    # basic activation functions

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    # backpropagation function
    
    def backward(self):
        # topological order for all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # one variable at a time, applying chain rule to get its gradient
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()

    # reverse arithmetic operations
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __repr__(self):
        return f'Value(data={self.data:.4f}, grad={self.grad:.4f})'