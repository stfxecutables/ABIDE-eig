import torch

from src.analysis.predict.deep_learning.models.layers.conv import Conv3Plus1D


def test_shape() -> None:
    x = torch.rand([1, 1, 8, 6, 6, 6], device="cuda")
    conv = Conv3Plus1D(
        in_channels=1,
        channel_expansion=2,
        spatial_in_shape=x.shape[3:],
        temporal_in_shape=x.shape[2],
        spatial_kernel=3,
        spatial_dilation=1,
        temporal_kernel=5,
        temporal_dilation=1,
    ).to("cuda")
    res = conv(x)
    assert res.shape == x.shape
