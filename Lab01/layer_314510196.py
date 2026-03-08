import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))
        self.input = None
        self.weight_grad = None
        self.bias_grad = None
        # 加上這行，避免 save_params() 報錯
        self.params_saved = {}
        import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features, weight_decay=0.0):
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))
        self.input = None
        self.weight_grad = None
        self.bias_grad = None
        # 加上這行，避免 save_params() 報錯
        self.params_saved = {}
        self.weight_decay = weight_decay  # 新增 weight decay 參數

    def forward(self, input, training=True):
        self.input = input
        batch_size = input.shape[0]
        input_flat = input.reshape(batch_size, -1)
        output = np.dot(input_flat, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        batch_size = self.input.shape[0]
        input_flat = self.input.reshape(batch_size, -1)
        self.weight_grad = np.dot(input_flat.T, output_grad)  # (in_features, out_features)
        if self.weight_decay > 0:
            self.weight_grad += self.weight_decay * self.weight  # ⬅ L2 正則化梯度

        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        input_grad = np.dot(output_grad, self.weight.T)  # (batch_size, in_features)
        input_grad = input_grad.reshape(self.input.shape)  # reshape回原 input shape
        return input_grad

    def update(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad
    
    def save_params(self):
        self.params_saved['weight'] = self.weight
        self.params_saved['bias'] = self.bias

    def load_params(self):
        self.weight = self.params_saved['weight']
        self.bias = self.params_saved['bias']

## by yourself .Finish your own NN framework
class Activation1(_Layer): # ReLU
    def __init__(self):
        self.input = None

    def forward(self, input, training=True):
        self.input = input
        output = np.maximum(0, input)

        return output

    def backward(self, output_grad):
        # do not modify output_grad in-place
        mask = (self.input > 0).astype(output_grad.dtype)
        return output_grad * mask

class ConvolutionalLayer:
    def __init__(self, filter_size, num_filters, input_shape, padding=0, stride=1):
        """
        初始化卷積層。
        
        參數:
        - filter_size (int): 濾波器的大小（假設高度和寬度相同）。
        - num_filters (int): 濾波器的數量。
        - input_shape (tuple): 輸入資料的形狀 (高度, 寬度, 通道數)。
        - padding (int): 在輸入資料周圍填充的零像素數量。
        - stride (int): 濾波器滑動的步幅。
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        
        # 儲存用於反向傳播的變數
        self.input_data = None
        self.input_padded = None
        
        input_channels = input_shape[2]
        
        # 初始化權重（濾波器）和偏置
        # self.weights = np.random.randn(filter_size, filter_size, input_channels, num_filters)  * np.sqrt(2.0 / (filter_size*filter_size*input_channels))
        # self.bias = np.zeros(num_filters)
        
        # self.grad_weights = np.zeros_like(self.weights)
        # self.grad_bias = np.zeros_like(self.bias)
        
        # ✅ He 初始化
        scale = np.sqrt(2.0 / (filter_size * filter_size * input_channels))
        self.weights = np.random.randn(filter_size, filter_size, input_channels, num_filters) * scale
        self.bias = np.zeros(num_filters)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, input_data):
        """
        執行前向傳播。
        
        參數:
        - input_data (np.ndarray): 輸入資料，形狀為 (批量大小, 高度, 寬度, 通道數)。
        
        回傳:
        - np.ndarray: 輸出的特徵圖。
        """
        batch_size, input_h, input_w, input_c = input_data.shape
        
        # 應用填充（Padding）
        self.input_padded = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        self.input_data = input_data

        padded_h, padded_w = self.input_padded.shape[1:3]

        # 計算輸出維度
        output_h = int((padded_h - self.filter_size) / self.stride) + 1
        output_w = int((padded_w - self.filter_size) / self.stride) + 1
        
        output = np.zeros((batch_size, output_h, output_w, self.num_filters))
        reshaped_weights = self.weights.reshape(-1, self.num_filters)
        
        for i in range(output_h):
            for j in range(output_w):
                # 提取當前濾波器覆蓋的輸入區域
                h_start = i * self.stride
                w_start = j * self.stride
                input_patch = self.input_padded[:, h_start:h_start+self.filter_size, w_start:w_start+self.filter_size, :]
                
                # 卷積運算：點積
                # 為了向量化，我們將 patch 和 weights reshape
                reshaped_patch = input_patch.reshape(batch_size, -1)
                # reshaped_weights = self.weights.reshape(-1, self.num_filters)
                
                output[:, i, j, :] = np.dot(reshaped_patch, reshaped_weights) + self.bias
        # self.output = output
        return output

    def backward(self, output_grad):
        """
        執行反向傳播。
        
        參數:
        - output_grad (np.ndarray): 來自下一層的梯度。
        
        回傳:
        - np.ndarray: 傳遞給前一層的梯度。
        """
        batch_size, padded_h, padded_w, _ = self.input_padded.shape
        _, output_h, output_w, num_filters = output_grad.shape
        
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.sum(output_grad, axis=(0, 1, 2))
        
        input_grad_padded = np.zeros_like(self.input_padded)
        reshaped_weights = self.weights.reshape(-1, num_filters)
        
        for i in range(output_h):
            for j in range(output_w):
                h_start = i * self.stride
                w_start = j * self.stride
                
                input_patch = self.input_padded[:, h_start:h_start+self.filter_size, w_start:w_start+self.filter_size, :]
                output_grad_patch = output_grad[:, i, j, :]
                
                # 計算權重梯度
                reshaped_patch = input_patch.reshape(batch_size, -1).T
                self.grad_weights += np.dot(reshaped_patch, output_grad_patch).reshape(self.filter_size, self.filter_size, self.input_shape[2], num_filters)
                
                # 計算輸入梯度
                # reshaped_weights = self.weights.reshape(-1, num_filters)
                # grad_input_patch = np.dot(output_grad_patch, reshaped_weights.T).reshape(batch_size, self.filter_size, self.filter_size, self.input_shape[2])
                grad_input_patch = np.tensordot(output_grad_patch, self.weights, axes=([1], [3]))

                input_grad_padded[:, h_start:h_start+self.filter_size, w_start:w_start+self.filter_size, :] += grad_input_patch

        # remove padding
        if self.padding > 0:
            input_grad = input_grad_padded[:, self.padding:padded_h - self.padding, self.padding:padded_w - self.padding, :]
        else:
            input_grad = input_grad_padded
        return input_grad

    def update(self, lr):
        """
        使用學習率更新權重和偏置。
        """
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias

class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.input = None
        self.pred = None

        self.y = None

    def forward(self, input, target, training=True):
        self.input = input
        self.y = target
        '''Softmax'''
        x = input - np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(x)
        predict = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.pred = predict
        '''Average loss'''
        predict_clip = np.clip(predict, 1e-12, 1-1e-12 )
        your_loss = -np.mean(np.sum(np.log(predict_clip) * target, axis=1))

        return predict, your_loss

    def backward(self):
        input_grad = self.pred - self.y

        return input_grad

class Activation2(_Layer): # Sigmoid
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input, training=True):
        self.input = input
        input = np.clip(input, a_min=-709, a_max=None)
        self.output = 1 / (1 + np.exp(-input))

        return self.output
    
    def backward(self, output_grad):
        input_grad = output_grad * (1 - self.output) * self.output

        return input_grad
    
class Dropout(_Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    
    def forward(self, input, training=True):
        if training:
            self.mask = (np.random.rand(*input.shape) > self.p)
            return input * self.mask / (1.0 - self.p)
        else:
            return input
        
    def backward(self, output_grad):
        # if forward wasn't in training, mask might be None -> handle
        if self.mask is None:
            return output_grad
        return output_grad * self.mask / (1.0 - self.p)
    

    # def update(self, lr):
    #     self.weight -= lr * self.weight_grad
    #     self.bias -= lr * self.bias_grad
    
    # def save_params(self):
    #     self.params_saved['weight'] = self.weight
    #     self.params_saved['bias'] = self.bias

    # def load_params(self):
    #     self.weight = self.params_saved['weight']
    #     self.bias = self.params_saved['bias']

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.argmax = None  # 用來記錄每個池化格子最大值的位置，方便 backward

    def forward(self, input):
        self.input = input
        batch_size, h, w, c = input.shape
        pool_h, pool_w = self.pool_size, self.pool_size
        stride = self.stride

        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1

        # 建立輸出和 argmax
        output = np.zeros((batch_size, out_h, out_w, c))
        self.argmax = np.zeros_like(input, dtype=bool)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + pool_h
                w_start = j * stride
                w_end = w_start + pool_w

                # 取出當前 pooling window 的值
                window = input[:, h_start:h_end, w_start:w_end, :]
                max_val = np.max(window, axis=(1, 2))
                output[:, i, j, :] = max_val

                # 記錄最大值位置 (one-hot mask)
                max_mask = (window == max_val[:, None, None, :])
                self.argmax[:, h_start:h_end, w_start:w_end, :] |= max_mask

        return output

    def backward(self, output_grad):
        # output_grad shape: (batch, out_h, out_w, c)
        batch_size, h, w, c = self.input.shape
        dx = np.zeros_like(self.input)

        pool_h, pool_w = self.pool_size, self.pool_size
        stride = self.stride

        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + pool_h
                w_start = j * stride
                w_end = w_start + pool_w

                mask = self.argmax[:, h_start:h_end, w_start:w_end, :]
                dx[:, h_start:h_end, w_start:w_end, :] += mask * output_grad[:, i:i+1, j:j+1, :]

        return dx