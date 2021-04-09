import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, w1, w2, b1, bias=True):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1, bias)
        self.set_weights(w1, w2, b1)

    def set_weights(self, w1, w2, b1):
        self.linear.weight.data[0, 0] = w1
        self.linear.weight.data[0, 1] = w2
        if self.linear.bias is not None:
            self.linear.bias.data[0] = b1
        return self

    def get_weights(self):
        weights = self.linear.weight.data.view(-1).tolist()
        if self.linear.bias is not None:
            weights += self.linear.bias.view(-1).tolist()
        return weights

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