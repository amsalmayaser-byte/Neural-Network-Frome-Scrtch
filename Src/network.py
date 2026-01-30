import numpy as np

class NeuralNetwork:
    def __init__(self):
        # مصفوفة لتخزين الطبقات التي سنضيفها
        self.layers = []
        self.loss = None

    def add(self, layer):
        # TODO: إضافة طبقة للشبكة
        self.layers.append(layer)

    def set_loss(self, loss_function):
        # تحديد دالة الخسارة (مثل MSE)
        self.loss = loss_function

    def predict(self, input_data):
        # TODO: تنفيذ Forward pass عبر كل الطبقات
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs,optimizer ):
        # TODO: حلقة التدريب الأساسية
        for i in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # 1. Forward pass (التوقع)
                output = self.predict(x)
                
                # 2. حساب الخطأ (Loss)
                error += self.loss.forward(output, y)
                
                # 3. Backward pass (الانتشار العكسي)
                # نبدأ من دالة الخسارة
                gradient = self.loss.backward()
                # ثم نمرر الخطأ عبر الطبقات من النهاية للبداية
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)
                    layer.update_weights(optimizer)
            
            # حساب متوسط الخطأ لكل Epoch
            error /= len(x_train)
            if (i+1) % 10 == 0:
                print(f"Epoch {i+1}/{epochs}, Error: {error}")
