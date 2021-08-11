from typing import Tuple, Union, no_type_check

import torch
from torch import Tensor
from torch.nn import Module


class GlobalAveragePooling(Module):
    def __init__(self, dims: Union[int, Tuple[int, ...]] = (2, 3, 4), keepdim: bool = False):
        super().__init__()
        self.dims = dims
        self.keepdim = bool(keepdim)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self.dims, keepdim=self.keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} reducing_dims={self.dims})"

    __repr__ = __str__


class Downsample(Module):
    def __init__(self) -> None:
        super().__init__()
