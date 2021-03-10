import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, w1, w2, b1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data[0, 0] = w1
        self.linear.weight.data[0, 1] = w2
        self.linear.bias.data[0] = b1

    def forward(self, x):
        return self.linear(x)

class StudyLineModel(LinearModel):
    def __init__(self, w0, w1, b):
        super(StudyLineModel, self).__init__(w0, w1, b)

    def get_w0(self):
        return self.linear.weight.data[0, 0]

    def get_w1(self):
        return self.linear.weight.data[0, 1]

    def get_b(self):
        return self.linear.bias.data[0]