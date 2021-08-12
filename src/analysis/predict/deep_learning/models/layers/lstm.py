from typing import Dict, Tuple

from torch import Tensor
from torch.nn import BatchNorm3d, ConstantPad3d, Conv3d, Module, PReLU
from torch.nn.modules.pooling import MaxPool3d

r"""
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

                       Can be implemented with 1 Conv that operates on concatenated [X_t, H_t-1]
                       and which is then split apart along the channel dimensions for later sums.
                                     |
                      /------------------------------\
    i_t =    sigmoid( W_xi @ X_t + W_hi @ H_t−1 + b_i + W_ci * c_t−1)
    f_t =    sigmoid( W_xf @ X_t + W_hf @ H_t−1 + b_f + W_cf * c_t−1)
    c_t = i_t * tanh( W_xc @ X_t + W_hc @ H_t−1 + b_c) + f_t * c_t−1
    o_t =    sigmoid( W_xo @ X_t + W_ho @ H_t−1 + b_o + W_co * c_t  )
    H_t = o_t * tanh( c_t )

We let K be the kernel size, and pad so there is no size reduction. H_d, the number
(dimension) of hidden states.

Hidden states truly are hidden *states*, in the sense that an input {X1, ..., XT} is T states, and
an input with C channels is like C*T states. Then hidden states are concatenated along the channel
dimension, so if we have H hidden states, we have (H + C) * T states total.

and so if we say there are are H hidden states, truly is a dimension, in that an LSTM cell with n hidden states is
like running that cell n times in parallel at each timepoint. For a ConvLSTM3D, the cell is 4D
(3D + channel dim), so stacking is along the channel dim, i.e. there is an equivalence between
number of filters and number of hidden states. So we use instead F = H_d to be the number of filters
or hidden states of each component / gate below.

    X    = (C_in, H, W, D, T) = 5D = 4D + channels: C_in == 1
    X_t  = (C_in, H, W, D)    = 4D = 3D + channels: C_in == 1

    W_xi = (F * C_in, K, K, K)    W_hi = (F,)    W_ci = (F,)    b_i = (1, F)  # see below
    W_xf = (F * C_in, K, K, K)    W_hi = (F,)    W_cf = (F,)    b_f = (1, F)
    W_xc = (F * C_in, K, K, K)    W_hi = (F,)    W_cc = (F,)    b_c = (1, F)
    W_xo = (F * C_in, K, K, K)    W_hi = (F,)    W_co = (F,)    b_o = (1, F)

Therefore:

    W_xi @ X_t = (F * C_in, H, W, D)

    W  = (4*F, K, K, K)    W_h = (4, F)   W_c = (4, F)     b = (4, F)
    Output shapes:       o = H = (T, F)  Cell = (T, F)

Biases can be either (H_d, F) fully independent (strange, not sure if would work) or (1, F) each if
tied (shared within a channel, one bias per filter) or I guess technically could be scalar values if
shared across channels. I don't think anyone does the latter, and tied is the default, so we don't
have to handle the biases if we just do `bias=True` in our conv layers.

Note if C_in > 1,



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
