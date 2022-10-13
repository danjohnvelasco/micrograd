import math
import numpy as np
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        assert isinstance(data, (int, float)), "Value object only supports int or float"
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
    
    def __repr__(self): # pragma: no cover
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if not a Value object, convert to Value object
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other): # other(int or float) + self(Value)
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other(int or float) + self(Value)
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if not a Value object, convert to Value object
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other): # other(int or float) * self(Value)
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if not a Value object, convert to Value object
        tfms_divisor = (other.data**-1)
        out = Value(self.data * tfms_divisor, (self, other), '/')

        def _backward():
            self.grad += tfms_divisor * out.grad
            other.grad += self.data * (-1*(other.data**-2)) * out.grad

        out._backward = _backward

        return out
    
    def __rtruediv__(self, other): # other(int or float) * self(Value)
        return Value(other) / self

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), _op='exp')

        def _backward():
            self.grad += out.data * out.grad # derivatve of exp(x) is exp(x)

        out._backward = _backward

        return out
    
    def pow(self, other):
        assert isinstance(other, (int, float)), "pow() only supports int or float"
        out = Value(self.data ** other, (self,), _op='pow')

        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)
        
        self.grad = 1.0 # base case
        
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __repr__(self):
        return f"weights: {self.w}\nbias: {self.b}\nshape: {len(self.w)}"

    def __call__(self, x_inputs):
        acts = sum([w*x for w, x in zip(self.w, x_inputs)], self.b)
        out = acts.tanh()

        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    # 
    def __repr__(self):
        return f"neurons: {self.neurons}"

    def __call__(self, x_inputs):
        outs = [n(x_inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        """
            nin: integer
            nouts: List[int]
        """
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x_inputs):
        # My version
        out = self.layers[0](x_inputs) # input layer

        for layer_idx in range(1, len(self.layers)):
            out = self.layers[layer_idx](out)

        # # Karpathy's version
        # for layer in self.layers:
        #     x_inputs = layer(x_inputs)

            
        return out