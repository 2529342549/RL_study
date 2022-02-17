import torch

from torch import nn


class T(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


t = T()
x = torch.tensor(1.0)
output = t(x)
print(output)
