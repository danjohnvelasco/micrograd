from micrograd.engine import Value
import random

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

    def parameters(self):
        return self.w + [self.b] # list of weights and bias at the end

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    # 
    def __repr__(self):
        return f"neurons: {self.neurons}"

    def __call__(self, x_inputs):
        outs = [n(x_inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # flatten to one list

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

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] # flatten to one list
        
class MSELoss:
    def __init__(self):
        self.data = None

    def __call__(self, ytrue, ypred):
        loss = sum([(yt - yp)**2 for yt, yp in zip(ytrue, ypred)])/len(ypred)
        self.data = loss
        return loss

class Optimizer:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.lr = learning_rate

    def __repr__(self):
        return f"lr: {self.lr}\n\nparameters:\n{self.parameters}"
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
