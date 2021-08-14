from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm3d,
    ConstantPad3d,
    Conv3d,
    Dropout3d,
    Module,
    ModuleList,
    Parameter,
    PReLU,
    Sequential,
)
from torch.nn.modules.pooling import MaxPool3d

IntSeq = Union[int, Tuple[int, ...]]
State = Tuple[Tensor, Tensor]

r"""
# Examples

https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
https://github.com/czifan/ConvLSTM.pytorch  # uses hx for H_t-1, hy for H_t, cx for c_t-1, etc.
https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py


# General Notes

We either want to implement the "C-LSTM" of https://arxiv.org/pdf/2002.05981.pdf or the FC-LSTM of
https://arxiv.org/pdf/1506.04214.pdf. All key elements except biases are 3D with equations:

    @ = convolution
    .* = elementwise multiplication
    + = element-wise or channel-wise, is often ambiguous due to broadcasting
    i = input, f = forget, C = "context" / cell state, o = output, H = hidden state

                       Can be implemented with 1 Conv that operates on concatenated [X_t, H_t-1]
                       and which is then split apart along the channel dimensions for later sums.
                                     |
                      /------------------------------\
    i_t =     sigmoid( W_xi @ X_t + W_hi @ H_t−1 + b_i + W_ci .* c_t−1)
    f_t =     sigmoid( W_xf @ X_t + W_hf @ H_t−1 + b_f + W_cf .* c_t−1)
    c_t = i_t .* tanh( W_xc @ X_t + W_hc @ H_t−1 + b_c) + f_t .* c_t−1
    o_t =     sigmoid( W_xo @ X_t + W_ho @ H_t−1 + b_o + W_co .* c_t  )
    H_t = o_t * tanh( c_t )

We let K be the kernel size, and pad so there is no size reduction. H_d, the number
(dimension) of hidden states.

"States" here is vague, but it it clear the intention is that a "state" is the system values at one
time t of T times. So for 1-channel 1D data (e.g. single stock price), the state is 1D, but for a
video file, the state is 2D. Thus a "hidden-state vector" of all computed hidden states is more
accuratly a tensor, or better, list of hidden-state tensors, and so if we have H = {h_1, ..., h_T},
then H.shape == (H_d, *SPATIAL, T) for any input with spatial dimensions SPATIAL, and hidden size /
dimension H_d. Thus H_t.shape == (H_d, *SPATIAL).  By contrast X_t.shape == (C_in, *SPATIAL).

 Then hidden states are concatenated along the channel
dimension, so if we have H_d hidden states, we have (H_d + C) * T states total. For a Conv LSTM, H_d
is the number of output channels needed, or number of filters, F. So we use instead F = H_d to be
the number of filters or hidden states of each component / gate below.

    X    = (C_in, H, W, D, T) = 5D = 4D + channels: C_in == 1
    X_t  = (C_in, H, W, D)    = 4D = 3D + channels: C_in == 1

    W_xi = (F + C_in, K, K, K)    W_hi = (F,)    W_ci = (F,)    b_i = (1, F)  # see below
    W_xf = (F + C_in, K, K, K)    W_hi = (F,)    W_cf = (F,)    b_f = (1, F)
    W_xc = (F + C_in, K, K, K)    W_hi = (F,)    W_cc = (F,)    b_c = (1, F)
    W_xo = (F + C_in, K, K, K)    W_hi = (F,)    W_co = (F,)    b_o = (1, F)

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

# FMRI NOTES

The ConvLSTM3d *makes sense* for fMRI. fMRI data is a sequence of 3D states. The eigenimage is *not*
a sequence in any traditional sense. The eigenindex does not represent sequentiality in any way. The
eigenimage is a truly 4D block where no dimension can be interpreted as sequentially preceding
another. However, each eigenindex *can* be validly thought of as a unique *channel*. This is
extremely accurate (though treating each t of an fMRI as a channel is not). In this sense, the
ConvLSTM3d makes perfect sense for fMRI, but a a multi-channel Conv3D or Conv4d makes most sense
for the eigenimages. In fact, the Conv3D is likely to outperform a Conv4D because the Conv4D would
struggle to find long-range relationships between eigenplanes

Of course, DL and LSTMS aprroximate everything with linearities. It could well be that treating
volumes defined by egenindexes as linearly related works *very well*.
"""


class ConvLSTMCell3d(Module):
    """Expects inputs to be (T, B, C, *SPATIAL), where len(SPATIAL) == 3.

    Notes
    -----
    GPU memory cost is a function of both spatial size, sequence length, hidden_size, and
    num_layers. An input tensor is O(H, W, D, T). Our base values are (47, 59, 42, 175). The
    input x size is thus (47*59*42*175), which according to

        x.element_size() * x.nelements / 1e6

    is about 80 MB. Thus regardless of the architecture, inputs alone with a batch size of B cost
    about 80*B MB. So already at a batch size of 12 inputs, we have used 1GB of GPU memory.




    weights for a conv that has kernel**3 * (hidden_size + in_channels) floats.  at 175 timepoints
    about , and saving

    """

    def __init__(
        self,
        in_channels: int,
        in_spatial_dims: Tuple[int, int, int],
        hidden_size: int,
        kernel_size: int = 3,
        dilation: int = 2,
        depthwise: bool = False,
        spatial_dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(spatial_dropout, float):
            raise ValueError("`spatial_dropout` must be None or in (0, 1)")
        if not (0 <= float(spatial_dropout) <= 0.99):
            raise ValueError("`spatial_dropout` must be None or in (0, 1)")
        spatial_dropout = float(spatial_dropout)  # type: ignore
        self.in_channels = in_channels
        self.spatial_dims = in_spatial_dims
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depthwise = depthwise
        self.effective_size = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        self.padding = (self.effective_size - 1) // 2  # need to maintain size
        self.has_dropout = spatial_dropout != 0
        self.dropout = float(spatial_dropout)

        if self.depthwise:
            in_ch = self.in_channels + self.hidden_size
            out_ch = 4 * self.hidden_size
            valid = None
            for groups in range(2, in_ch + 1):
                if in_ch % groups == 0 and out_ch % groups == 0:
                    valid = groups
            if valid is None:
                raise ValueError(
                    f"No valid depthwise group size for current hidden size {self.hidden_size}"
                )

        self.layers = [
            Conv3d(
                in_channels=self.in_channels + self.hidden_size,
                out_channels=4 * self.hidden_size,  # i, f, c, o, see notes above
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.in_channels + self.hidden_size if self.depthwise else 1,
                bias=True,
            ),
            BatchNorm3d(4 * self.hidden_size),
        ]
        if self.has_dropout:
            self.layers.append(Dropout3d(self.dropout))

        # any function that can be applied recursively (maintains shape) works here?
        self.recursor = Sequential(*self.layers)

        self.Wci = Parameter(torch.zeros((1, self.hidden_size, *self.spatial_dims)))
        self.Wcf = Parameter(torch.zeros((1, self.hidden_size, *self.spatial_dims)))
        self.Wco = Parameter(torch.zeros((1, self.hidden_size, *self.spatial_dims)))

    def forward(self, x: Tensor, state: State) -> State:
        """Requires x.shape == (B, C, *SPATIAL)"""
        h, c = state  # current state (or equivalently, previous state, h_t-1)
        hx = torch.cat((x.to(device="cuda"), h), dim=1)  # channel dimension must be 1
        convolved = self.recursor(hx)
        ii, ff, cc, oo = torch.chunk(convolved, 4, dim=1)  # un-concatenated combined terms

        i_t = torch.sigmoid(ii + self.Wci * c)
        f_t = torch.sigmoid(ff + self.Wcf * c)
        c_t = i_t * torch.tanh(cc) + f_t * c
        o_t = torch.sigmoid(oo + self.Wco * c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def initialize(self, batch_size: int, device: str) -> State:
        """Returns hidden and cell states for t = 0 initial case, and initializes
        Wc weights on correct device.
        """
        size = (batch_size, self.hidden_size, *self.spatial_dims)
        if self.Wci is None:
            self.Wci.to(device)
            self.Wcf.to(device)
            self.Wco.to(device)
        return torch.zeros(size).to(device), torch.zeros(size).to(device)


class ConvLSTM3d(Module):
    """Implement an multi-layer LSTM.

    Parameters
    ----------
    in_channels: int
        Number of channels in input tensors.

    in_spatial_dims: Tuple[int, int, int]
        Spatial dimensions. Needed for defining W_c elementwise product weights.

    num_layers: int | Sequence[int]
        Number of ConvLSTM cells.

    hidden_sizes: int | Sequence[int]
        Size of the hidden dimension of each cell.

    kernel_size: int | Sequence[int]
        Size of the convolution kernel in each cell.

    dilations: int | Sequence[int]
        Amount of dilation for each kernel of each cell.

    inner_spatial_dropout: float = 0.0
        If a value p in (0, 1), randomly drops some channels (of hidden states + input) with
        probability p at *each* timestep t. I.e. the Conv layer is followed by a spatial dropout
        layer (Dropout3D(p)).

    spatial_dropout: float = 0.0
        If a value p in (0, 1), randomly drops some channels (of hidden states + input) with
        probability p between each layer. is followed by a spatial dropout layer (Dropout3D(p)).

    Notes
    -----
    Expects inputs to be (B, T, C, *SPATIAL), where len(SPATIAL) == 3

    """

    def __init__(
        self,
        in_channels: int,
        in_spatial_dims: Tuple[int, int, int],
        num_layers: int = 1,
        hidden_sizes: Sequence[int] = [32],
        kernel_sizes: Sequence[int] = [3],
        dilations: Sequence[int] = [2],
        depthwise: bool = False,
        inner_spatial_dropout: float = 0.0,
        spatial_dropout: float = 0.0,
        min_gpu: bool = True,  # save results on cpu to save GPU memory
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_spatial_dims = in_spatial_dims
        self.num_layers = num_layers
        self.hidden_sizes = self.listify(hidden_sizes)
        self.kernel_sizes = self.listify(kernel_sizes)
        self.dilations = self.listify(dilations)
        self.depthwise = depthwise
        self.inner_spatial_dropout = inner_spatial_dropout
        self.spatial_dropout = spatial_dropout
        self.min_gpu = min_gpu
        if self.spatial_dropout > 0:
            raise NotImplementedError("Spatial dropout between layers is not yet implemented")

        cell_args: Dict = dict(
            in_spatial_dims=in_spatial_dims,
            spatial_dropout=inner_spatial_dropout,
            depthwise=self.depthwise,
        )

        layers = []
        for i in range(num_layers):
            layers.append(
                ConvLSTMCell3d(
                    in_channels=self.in_channels if i == 0 else self.hidden_sizes[i - 1],
                    hidden_size=self.hidden_sizes[i],
                    kernel_size=self.kernel_sizes[i],
                    dilation=self.dilations[i],
                    **cell_args,
                )
            )
        self.layers: ModuleList = ModuleList(layers)  # properly register

    def forward(self, x: Tensor) -> State:
        """x.shape ==  (B, T, C, *SPATIAL), where len(SPATIAL) == 3"""
        # we only need to initialize the state to zeros for t=0, but since that would occur for i=0,
        # t=0 in the loops below, we need to initialize before
        layer: ConvLSTMCell3d
        state: State
        T, batch_size = x.size(1), x.size(0)
        x_layer = x
        for i, layer in enumerate(self.layers):
            state = layer.initialize(batch_size, x.device)
            hs = []
            for t in range(T):
                if self.min_gpu:
                    x_t = x_layer[:, t].clone().to(device="cpu")
                else:
                    x_t = x_layer[:, t]
                state = layer(x_t, state)
                if self.min_gpu:
                    hs.append(state[0].clone().to(device="cpu"))
                else:
                    hs.append(state[0])
            x_layer = torch.stack(hs, dim=1)  # Tensor/list of all computed hidden states
        return state

    def listify(self, item: Union[int, Sequence[int]]) -> List[int]:
        if isinstance(item, int):
            return [item for _ in range(self.num_layers)]
        if not isinstance(item, list):
            items = [*item]
            if len(items) == self.num_layers:
                return items
        raise ValueError(
            "Layer arguments must be an int or sequence of ints with length `num_layers`."
        )
