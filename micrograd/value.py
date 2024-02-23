import math

class Value:
    def __init__(self, data, _children=(), op='', label='') -> None:
        # accept only numbers
        assert isinstance (data, (int, float))
        # variables
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self.op = op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        # handle cases where power is multipled by Value variables
        if (isinstance(other, Value)):
            other = other.data
        # power can only be int or float
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data) ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    # below are most important activation functions
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / ((math.exp(2*x)) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - (t ** 2)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        t = 1 / (1 + math.exp(-self.data))
        # sigmoid(x)*(1- sigmoid(x)) 
        out = Value(t, (self, ), 'Sigmoid')
        
        def _backward():
            self.grad += (t * (1 - t)) * out.grad
        
        out._backward = _backward

        return out



    
    # handle how negative sign
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    # handle reverse arithmatic calculations

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other

    # print
    def __repr__(self) -> str:
        return f'Value(data={self.data})'
    

if __name__ == '__main__':
    # test
    x, y = 4, 8
    a = Value(x)
    b = Value(y)
    c = a + b 
    d = a - b
    e = a * b
    f = b / a
    h = a ** b
    i = a ** 2


    # arithmatic tests
    assert c.data == (x + y)
    assert d.data == (x - y)
    assert e.data == (x * y)
    assert f.data == (y / x)
    assert h.data == (x ** y)
    assert i.data == (x ** 2)
    print('All Arithmatic tests passed')

    # grad tests
    # Addition grad test
    c.grad = 1.0
    c._backward()
    assert (a.grad, b.grad, c.grad) == (1, 1, 1)

    # Multiplication grad test
    a.grad, b.grad, e.grad = 0, 0, 1
    e._backward()
    assert (a.grad, b.grad, e.grad) == (8, 4, 1)
