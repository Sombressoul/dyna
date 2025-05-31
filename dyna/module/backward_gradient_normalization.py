import torch.nn as nn

from dyna.functional import backward_gradient_normalization

class BGNLayer(nn.Module):
    """
    An implementation of the paper "Backward Gradient Normalization in Deep Neural Networks"
    from June 18, 2021 by Alejandro Cabana and Luis F. Lago-Fernández.

    For more information, read the original publication: https://arxiv.org/abs/2106.09475
    """
    def __init__(self):
        super(BGNLayer, self).__init__()

    def forward(self, x):
        return backward_gradient_normalization(x)
