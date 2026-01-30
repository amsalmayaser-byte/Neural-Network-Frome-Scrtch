import numpy as np
from nn_library.activations.functions import Sigmoid, ReLU, Tanh

class Layer:
    def __init__(self):
        pass
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        # تعريف الأوزان والانحياز
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        # تهيئة الجرادينت كـ Attributes (لحل إيرور Adam)
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        
        # ربط دالة التنشيط
        if activation == 'relu': self.activation = ReLU()
        elif activation == 'sigmoid': self.activation = Sigmoid()
        elif activation == 'tanh': self.activation = Tanh()
        else: self.activation = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        if self.activation:
            return self.activation.forward(self.z)
        return self.z

    def backward(self, output_gradient):
        output_gradient
        # 1. حساب dz عبر دالة التنشيط
        if self.activation:
            dz = self.activation.backward(output_gradient)
        else:
            dz = output_gradient

        if dz.ndim == 1:
           dz= dz.reshape(1, -1)    
        if self.input.ndim == 1:
           self.input = self.input.reshape(1,-1) 

        # 2. تحديث الجرادينت في self (هذا هو التعديل المطلوب)
        self.weights_gradient = np.dot(self.input.T,dz)
        self.bias_gradient = np.sum(dz, axis=0, keepdims=True)
        
        # 3. حساب الجرادينت العائد للخلف
        input_gradient = np.dot(dz, self.weights.T)
        return input_gradient

    def update_weights(self, optimizer):
        optimizer.update(self)
