from value import Value
import random

# Base class similar to PyTorch
class Module:
    # resets all parameters grads to zero for each iteration
    def zero_grad(self):
        for _ in self.parameters():
            _.grad = 0.0
    
    # default fall back function, if not implemented natively
    def parameters(self):
        return self.weights + [self.bias]
    

class Neuron(Module):
    def __init__(self, nin, bias=True) -> None:
        # create nin of neuron weights
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1)) if bias else None

    # change this function according for the use-case
    def __call__(self, X):
        linear_res = sum(([wi*xi for wi, xi in zip(self.weights, X)]), self.bias)
        out = linear_res.tanh() # apply activation func
        return out
    
class Layer(Module):
    def __init__(self, nin, out) -> None:
        self.neurons = [Neuron(nin) for _ in range(out)]
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    def __init__(self, nin, outs) -> None:
        layers = [nin] + outs
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]

if __name__ == '__main__':
    # unit tests
    # neurons count
    x = Neuron(4)
    n = MLP(3, [4, 4, 1])
    assert len(x.parameters()) == 5
    assert len(n.parameters()) == 41

    # back propagation
    y = [2.0, 4.2, -1.9]
    t = x(y)
    u = n(y)
    # should back propagate without errors
    t.backward()
    u.backward()




    # Test loss function 
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired target

    y_pred = [n(i) for i in xs]
    # print(list(zip(y_pred, ys)))
    
    # loss
    loss = sum([(yp - y)** 2 for yp, y in zip(y_pred, ys)])
    # zero_grad
    n.zero_grad()
    
    # backpropagation
    loss.backward()
    
    # step or update
    for each in n.parameters():
        each.data += -0.1 * each.grad
    
    y_pred = [n(i) for i in xs]
    # print(list(zip(y_pred, ys)))
    
    # loss
    loss_after = sum([(yp - y)** 2 for yp, y in zip(y_pred, ys)])
    # loss should be less than before
    assert loss_after.data < loss.data
    print('All tests passed')
