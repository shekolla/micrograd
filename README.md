# MicroGrad

MicroGrad is a minimalistic deep learning framework implemented in Python, providing a simple yet educational foundation for understanding the basics of automatic differentiation and neural network construction. The framework is designed to be lightweight, making it an ideal tool for learning and experimentation.

## Features

- **Automatic Differentiation**: MicroGrad implements automatic differentiation to calculate gradients efficiently.
- **Activation Functions**: The framework includes essential activation functions such as Tanh, ReLU, and Sigmoid.
- **Neural Network Modules**: MicroGrad provides base classes for building neural network modules, including neurons, layers, and multi-layer perceptrons (MLPs).
- **Random Initialization**: Neurons in the framework are initialized with random weights within a specified range.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

Clone the repository and include the MicroGrad files in your project:

```bash
git clone https://github.com/shekolla/micrograd.git
```

### Usage

#### 1. Import MicroGrad

```python
from value import Value
from nn import Neuron, Layer, MLP
```

#### 2. Create a Value

```python
# Example of creating a Value with data 2.0
v = Value(2.0)
```

#### 3. Build a Neural Network

```python
# Example of building a simple neural network
nin = 5
outs = [3, 2]
model = MLP(nin, outs)
```

## Acknowledgments

MicroGrad is inspired by the principles of automatic differentiation and serves as a learning resource for deep learning enthusiasts.
