import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, weights, bias=True):
        super().__init__()
        input_weights_count = len(weights) - (1 if bias else 0)
        self.linear = nn.Linear(input_weights_count, 1, bias)
        self.set_weights(weights)

    def set_weights(self, weights):
        if self.linear.bias is not None:
            self.linear.bias.data[0] = weights[0]
            weights = weights[1:]
        self.linear.weight.data = torch.as_tensor(weights, dtype=self.linear.weight.data.dtype).view_as(self.linear.weight.data)
        return self

    def get_weights(self):
        if self.linear.bias is not None:
            weights = [] if self.linear.bias is None else self.linear.bias.view(-1).tolist()
        weights += self.linear.weight.data.view(-1).tolist()
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