Custom Neural Network Library (NumPy)

A lightweight and modular neural network library implemented from scratch using Python and NumPy.
The purpose of this project is to deeply understand and implement the internal mechanics of neural networks — including forward propagation, backpropagation, activation functions, and optimization algorithms — without relying on existing deep learning frameworks such as PyTorch or TensorFlow.
This library prioritizes clarity, correctness, and educational value over computational performance.


Project Goals

. Understand how neural networks work internally at a low level
. Implement backpropagation manually using the chain rule
. Design a modular architecture similar to modern deep learning libraries
. Explore optimization techniques such as Adam
. Integrate regularization to control overfitting

Key Features

Modular Design
Layers are independent and reusable
Networks are built by dynamically stacking layers
Easy to extend with new layers, activation functions, or loss functions


Activation Functions

The following activation functions are implemented with both forward and backward passes:
. ReLU
. Sigmoid
. Tanh


Elastic Net Regularization (L1 & L2)

The library supports L1 and L2 regularization at the layer level:
. L1 Regularization encourages sparsity in weights
. L2 Regularization penalizes large weights to improve numerical stability

Regularization terms are added directly to the weight gradients during backpropagation.
Bias terms are not regularized.


Optimizers

. SGD (baseline optimizer)
. Adam Optimizer, featuring:
   . First moment (momentum) estimation
   . Second moment (adaptive scaling)
   . Bias correction
   . Independent optimizer state per layer

Design & Mathematical Notes

. Gradients are computed manually (no automatic differentiation)
. All tensors are handled explicitly as NumPy arrays with fixed shapes
. Each layer caches its forward-pass inputs for use during backpropagation
. Optimizer state (moments, time step) is explicitly managed
. Regularization is applied only to weight matrices

This design mimics the internal mechanics of modern deep learning frameworks while remaining fully transparent.


Example Usage: 

import numpy as np
from network import NeuralNetwork
from layer import Dense
from optimizers import Adam
from losses import MSE

# Dummy dataset
X = np.random.randn(100, 4)
Y = np.random.randn(100, 1)

# Build network
nn = NeuralNetwork()
nn.add(Dense(4, 8, activation='relu', l1_lambda=0.01, l2_lambda=0.01))
nn.add(Dense(8, 1, activation='sigmoid'))

# Set loss and optimizer
nn.set_loss(MSE())
optimizer = Adam(learning_rate=0.001)

# Train
nn.train(X, Y, epochs=100, optimizer=optimizer)

Disclaimer

This project is intended strictly for learning and experimentation.
It is not designed for production use or large-scale training workloads.

Possible Extensions

. Mini-batch training
. Additional loss functions (e.g. Cross-Entropy)
. Dropout and Batch Normalization
. Computational graph abstraction
. Model saving and loading 

 Project Structure
.
├── network.py        # Manages layers, loss function, and training loop
├── layer.py          # Dense layer implementation (forward & backward)
├── optimizers.py     # Optimizers (SGD, Adam)
├── functions.py      # Activation functions (ReLU, Sigmoid, Tanh)
├── losses.py         # Loss functions (e.g. MSE)
├── neuron.py         # Standalone single-neuron implementation (educational)
└── test.py           # Testing and experimentation script
