import numpy as np
import math


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        relu_func = np.frompyfunc(lambda i: max(i,0.), 1, 1)
        output = relu_func(input)
        self._saved_for_backward(output.copy())
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        dRelu = self._saved_tensor[self._saved_tensor > 0.] = 1.
        gradient = grad_output * dSigmod
        return gradient


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        sigmod_func = np.frompyfunc(lambda i: 1./(1.+math.exp(i)), 1, 1)
        output = sigmod_func(input)
        self._saved_for_backward(output.copy())
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        dSigmod = self._saved_tensor * (1 -self._saved_tensor)
        gradient = grad_output * dSigmod
        """
        print("grad_output = ")
        print(grad_output[0:10])
        print("dSigmod = ")
        print(dSigmod[0:10])
        """
        return gradient


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        output = np.dot(input, self.W) + self.b
        self._saved_for_backward(input.copy())
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        self.grad_W = np.dot(self._saved_tensor.T, grad_output) / grad_output.shape[0]
        self.grad_b = np.mean(grad_output, axis=0)
        gradient = np.dot(grad_output, self.W.T)
        """
        print("grad_output = ")
        print(grad_output)
        print("input = ")
        print(self._saved_tensor)
        print("grad_w = ")
        print(self.grad_W)
        print("grad_b = ")
        print(self.grad_b)
        """
        return gradient

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
