import numpy as np
class Optimizer:
    """القالب الأساسي للمحسنات"""
    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):
        # TODO: تحديث الأوزان والانحياز باستخدام معدل التعلم والاشتقاق
        layer.weights -= self.lr * layer.weights_gradient
        layer.bias -= self.lr * layer.bias_gradient
class Adam(Optimizer):
    """المحسن الذكي (Adam) - يحتاج تخزين متوسطات الحركة"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w, self.v_w = None, None # لتخزين العزم (Moments) للأوزان
        self.m_b, self.v_b = None, None # لتخزين العزم للانحياز
        self.t = 0 # عداد الخطوات

    def update(self, layer):
        # تهيئة المصفوفات في أول مرة
        if self.m_w is None:
            self.m_w = np.zeros_like(layer.weights)
            self.v_w = np.zeros_like(layer.weights)
            self.m_b = np.zeros_like(layer.bias)
            self.v_b = np.zeros_like(layer.bias)

        self.t += 1
        
        # TODO: تنفيذ معادلات Adam المعقدة لتحديث الأوزان
        # تحديث العزم الأول والثاني
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * layer.weights_gradient
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (layer.weights_gradient**2)
        
       
        m_w_hat = self.m_w / (1 - self.beta1**self.t)
        v_w_hat = self.v_w / (1 - self.beta2**self.t)
        
        # التحديث النهائي
        layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
