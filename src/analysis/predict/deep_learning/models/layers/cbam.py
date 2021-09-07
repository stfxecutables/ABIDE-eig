import torch
from torch import Tensor, nn
from torch.nn import BatchNorm3d, Conv3d, Linear, Module, ReLU, Sequential


# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L6-L20
class ConvCBAM(Module):
    def __init__(self) -> None:
        IN_CH, OUT_CH, KERNEL = 2, 1, 7
        PAD = (KERNEL - 1) // 2
        super().__init__()
        self.conv = Conv3d(
            in_channels=IN_CH, out_channels=OUT_CH, kernel_size=KERNEL, padding=PAD, bias=False
        )
        self.bnorm = BatchNorm3d(num_features=OUT_CH, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bnorm(x)
        return x


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        return x


# Tidied and simplified version of CBAM channel gate author from original paper:
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26-L60
# changed to Conv3d
class ChannelGate(Module):
    def __init__(self, in_channels: int, reduction: int):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = Sequential(
            Flatten(),
            Linear(in_channels, in_channels // reduction),
            ReLU(),
            Linear(in_channels // reduction, in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # on GitHub we see the code:
        #
        #       F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        #
        # F.avg_pool2d is (input, kernel_size, stride), so this ridiculous operation is just a very
        # slow (https://github.com/pytorch/pytorch/issues/14615) and unreadable version of:
        #
        #       `torch.mean(x, axis=(2,3), keepdim=True)`
        dims = dict(dim=(2, 3, 4), keepdim=True)
        attention = self.mlp(torch.mean(x, **dims))
        # torch.amax (https://stackoverflow.com/a/61157056) not implemented yet in version we need...
        mx = x.view(x.size(0), x.size(1), -1).max(dim=-1)[0].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        attention += self.mlp(mx)
        attention = torch.sigmoid(attention).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x = x * attention
        return x


# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L68-L70
class ChannelPool(Module):
    def forward(self, x: Tensor) -> Tensor:
        dims = dict(dim=1, keepdim=True)
        x = torch.cat((torch.max(x, **dims)[0], torch.mean(x, **dims)), dim=1)
        return x


# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L72-L82
class SpatialGate(Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel_pool = ChannelPool()
        self.conv = ConvCBAM()

    def forward(self, x: Tensor) -> Tensor:
        attention = self.channel_pool(x)
        attention = self.conv(attention)
        attention = torch.sigmoid(attention)
        x = x * attention
        return x


# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L84-L95
class CBAM(Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.channel_gate = ChannelGate(in_channels=in_channels, reduction=reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x
