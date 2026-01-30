import numpy as np
# استيراد الدوال التي جهزناها سابقاً في الملفات الأخرى
from nn_library.activations.functions import Sigmoid, ReLU, Tanh

class Neuron:
    def __init__(self, n_inputs, activation='sigmoid'):
        # TODO 1: تهيئة الأوزان والانحياز (قانون التهيئة العشوائية)
        # نستخدم توزيعاً عشوائياً للأوزان ونبدأ الـ bias بصفر
        self.w = np.random.randn(n_inputs) * 0.01 
        self.b = 0.0
        
        # اختيار الكائن (Object) الخاص بدالة التنشيط بناءً على الاسم
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = None

    def forward(self, x):
        # TODO 2: تنفيذ الـ Forward Pass
        # المعادلة العالمية للعصبون: (الوزن × المدخل) + الانحياز
        self.x = x # نحفظ المدخلات لأننا سنحتاجها في الباكورد
        self.z = np.dot(self.w, x) + self.b
        
        # إذا كان هناك دالة تنشيط، نمرر النتيجة من خلالها
        if self.activation:
            return self.activation.forward(self.z)
        return self.z

    def backward(self, dout, learning_rate=0.01):
        # TODO 3: تنفيذ الـ Backward Pass (قانون الاشتقاق المتسلسل)
        # 1. نحسب مشتقة دالة التنشيط أولاً (dz)
        if self.activation:
            dz = self.activation.backward(dout)
        else:
            dz = dout
            
        # 2. حساب مشتقة الأوزان (dw) والانحياز (db)
        # قانون: مشتقة الوزن هي (dz × المدخلات)
        dw = dz * self.x
        db = dz
        
        # 3. تحديث الأوزان (Update Rules)
        # القانون: الوزن الجديد = الوزن القديم - (معدل التعلم × المشتقة)
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
        
        return dw, db
