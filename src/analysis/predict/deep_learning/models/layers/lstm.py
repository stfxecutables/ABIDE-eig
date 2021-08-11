from typing import Dict, Tuple

from torch import Tensor
from torch.nn import BatchNorm3d, ConstantPad3d, Conv3d, Module, PReLU
from torch.nn.modules.pooling import MaxPool3d

"""
# Examples

https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
https://github.com/czifan/ConvLSTM.pytorch
https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py


# General Notes

We either want to implement the "C-LSTM" of https://arxiv.org/pdf/2002.05981.pdf or the FC-LSTM of
https://arxiv.org/pdf/1506.04214.pdf. This means the equations:

    @ = convolution
    * = elementwise multiplication
    i = input, f = forget, C = "context" / cell state, o = output, H = hidden state

    i_t = sigmoid( W_xi @ X_t + W_hi @ H_t−1 + W_ci * C_t−1 + b_i )
    f_t = sigmoid( W_xf @ X_t + W_hf @ H_t−1 + W_cf * C_t−1 + b_f )
    C_t = f_t * C_t−1 + i_t * tanh( W_xc * X_t + W_hc @ H_t−1 + b_c )
    o_t = sigmoid( W_xo @ X_t + W_ho @ H_t−1 + W_co * C_t + b_o )
    H_t = o_t * tanh( C_t )

If we let K be the kernel size, pad so there is no size reduction, and let H_n be the number of
hidden stats, then dimensions in our case are:

    X   = (C, H, W, D, T) = 5D = 4D + channels, except C == 1
    X_t = (C, H, W, D) = 4D = 3D + channels
    H   = (H_n, T)
    W_xi = (H_n, C, K, K, K)



# Implementation Notes

We can acheive speed boosts by concatenating, performing certain operations at once, and then
splitting using either torch.split or Tensor.chunk:

    https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py#L41-L44
    https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L106-L108

Likewise, you often see an undocumented "4" in the channels or dimensions of some ConvLSTM or LSTM
implementations:

    https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py#L33
    https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L98-L101
    https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L108

This is because there are four weight matrices (W_xi, W_xo, W_xf, W_xc) to be re-implemented as
convolutions, four hidden weight channeled-vectors (W_hi, W_hf, W_hc, W_ho), and four cell weight
channeled-vectors
the

Note that biases must be initialized as torch.Parameter objects to properly update/track gradients.
"""


class ConvLSTMCell(Module):
    def __init__(self):
        super().__init__()
        self.w_xi
