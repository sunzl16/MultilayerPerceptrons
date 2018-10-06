from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        input[input > 0.5] = 1.
        input[input <= 0.5] = 0.
        loss_value = np.sum((target - input) ** 2 / 2.0)
        return loss_value

    def backward(self, input, target):
        '''Your codes here'''
        input[input > 0.5] = 1.
        input[input <= 0.5] = 0.
        grad = target - input
        return grad
