from .layer import *
import numpy as np

class Network(object):
    def __init__(self):

        self.layers = []
        
        # 儲存每個層次的前向傳播輸出形狀，用於反向傳播時的維度還原
        self.layer_outputs = []

        # --- 卷積層部分 ---
        # 輸入形狀: (28, 28, 1)
        self.layers.append(ConvolutionalLayer(filter_size=3, num_filters=16, input_shape=(28, 28, 1)))
        self.layers.append(Activation1())
        
        # 輸入形狀: (26, 26, 16)
        self.layers.append(ConvolutionalLayer(filter_size=3, num_filters=32, input_shape=(26, 26, 16)))
        self.layers.append(Activation1())
        

        # --- 全連接層部分 --- 
        # 攤平 (Flatten)：24x24x32 -> 18432 
        flatten_input_shape = 24 * 24 * 32 
        self.layers.append(FullyConnected(flatten_input_shape, 400, weight_decay=5e-4)) 
        self.layers.append(Activation2()) # 
        self.layers.append(Dropout(0.25)) 
        self.layers.append(FullyConnected(400, 100, weight_decay=5e-4)) 
        self.layers.append(Activation2()) 
        self.layers.append(Dropout(0.25)) 
        self.layers.append(FullyConnected(100, 10, weight_decay=5e-4)) 
        self.layers.append(SoftmaxWithloss())
        


    def forward(self, input, target, training=True):
        self.layer_outputs = []  # 清空上次的記錄
        # 在將資料傳給第一層之前，先進行自動塑形
        if input.ndim == 2:
            # 假設輸入是 (batch_size, 784)，將其重新塑形為 (batch_size, 28, 28, 1)
            input = input.reshape(-1, 28, 28, 1)

        output = input
        for layer in self.layers:
            if isinstance(layer, FullyConnected) and output.ndim > 2:
                batch_size = output.shape[0]
                output = output.reshape(batch_size, -1)
            # 儲存每一層的輸出形狀，用於反向傳播的維度還原
            self.layer_outputs.append(output.shape)

            if isinstance(layer, Dropout):
                output = layer.forward(output, training)
            elif isinstance(layer, SoftmaxWithloss):
                pred, loss = layer.forward(output, target, training)
                # ⬇ 加上 L2 loss
                l2_loss = 0.0
                for lyr in self.layers:
                    if isinstance(lyr, FullyConnected) and lyr.weight_decay > 0:
                        l2_loss += 0.5 * lyr.weight_decay * np.sum(lyr.weight ** 2)
                loss += l2_loss
                return pred, loss
            else:
                output = layer.forward(output)
        
        return output, None

    def backward(self):
        grad = self.layers[-1].backward()
        debug = False  # 或 False
        #print(f"[DEBUG] After SoftmaxWithloss.backward: {grad.shape}")

        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            #print(f"[DEBUG] Layer {i} ({layer.__class__.__name__}) grad.shape before backward: {grad.shape}")

            # grad = layer.backward(grad)

            #print(f"[DEBUG] Layer {i} ({layer.__class__.__name__}) grad.shape after backward: {grad.shape}")
            # 如果 FullyConnected 的 grad 維度不對，reshape
            if isinstance(layer, FullyConnected) and grad.ndim > 2:
                batch_size = grad.shape[0]
                grad = grad.reshape(batch_size, -1)

            grad = layer.backward(grad)
            if isinstance(layer, FullyConnected) and i > 0 and isinstance(self.layers[i-1], Activation1):
                # 前一層是卷積層的輸出，取卷積層 forward 時的輸出形狀
                conv_output_shape = self.layer_outputs[i-1]  # 這裡會是 (batch_size, 24, 24, 32)
                #print(f"[DEBUG] Reshaping grad from {grad.shape} to {conv_output_shape}")
                grad = grad.reshape(conv_output_shape)
            if debug:
                if hasattr(layer, "weight_grad"):
                    print(f"[DEBUG] Layer {i} ({layer.__class__.__name__}) weight_grad mean: {np.mean(layer.weight_grad):.6f} | max: {np.max(layer.weight_grad):.6f}")
                if hasattr(layer, "bias_grad"):
                    print(f"[DEBUG] Layer {i} ({layer.__class__.__name__}) bias_grad mean: {np.mean(layer.bias_grad):.6f}")

        return grad  # 最後回傳 grad，雖然上層不用
    
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):  # 只有有 update 方法的層才更新
                layer.update(lr)
            
    def save_params(self):
        for layer in self.layers:
            if isinstance(layer, ( FullyConnected)):
                layer.save_params()

    def load_params(self):
        for layer in self.layers:
            if isinstance(layer, ( FullyConnected)):
                layer.load_params()

class CosineAnnealingWarmRestarts:
    def __init__(self, eta_max, eta_min, T_i, T_mult=1):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_i = T_i
        self.T_mult = T_mult
        self.T_cur = T_i 
        self.current_epoch = 0

    def get_lr(self, epoch):
        if epoch == 0:
            return self.eta_max

        if self.current_epoch >= self.T_cur:
            self.current_epoch = 0
            self.T_cur *= self.T_mult

        cosine_decay = 0.5 * (1 + np.cos(np.pi * self.current_epoch / self.T_cur))
        learning_rate = self.eta_min + (self.eta_max - self.eta_min) * cosine_decay

        self.current_epoch += 1

        return learning_rate
    
class CosineAnnealingWarmRestartswithWarmUp:
    def __init__(self, base_lr, max_lr, min_lr, warmup_steps, total_steps, restart_interval):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.restart_interval = restart_interval
        self.current_step = 0

    def get_lr(self):
        cycle = np.floor(1 + self.current_step / self.restart_interval)
        x = self.current_step - cycle * self.restart_interval
        if x < 0:
            x = 0
        
        if self.current_step < self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / self.restart_interval
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        return lr

    def step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0
