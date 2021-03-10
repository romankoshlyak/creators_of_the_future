import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, w1, w2, b1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data[0, 0] = w1
        self.linear.weight.data[0, 1] = w2
        self.linear.bias.data[0] = b1
        self.max_value = torch.tensor(-10.0)
        self.activation = LINEAR
 
    def set_activation(self, activation):
        self.activation = activation
        return self

    def forward(self, x):
        x = self.linear(x)
        activation = ACTIVATIONS[self.activation]
        if activation is not None:
            x = activation(x)
        return x

class StudyLineModel(nn.Module):
    def __init__(self, w0, w1, b):
        super(StudyLineModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data[0, 0] = w0
        self.linear.weight.data[0, 1] = w1
        self.linear.bias.data[0] = b

    def get_w0(self):
        return self.linear.weight.data[0, 0]

    def get_w1(self):
        return self.linear.weight.data[0, 1]

    def get_b(self):
        return self.linear.bias.data[0]

    def forward(self, x):
        x = self.linear(x)
        return x