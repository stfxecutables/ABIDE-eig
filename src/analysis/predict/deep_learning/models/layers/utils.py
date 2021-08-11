from math import floor
from typing import List, Tuple, Union

Tuple3d = Union[int, Tuple[int, int, int]]

# need to pad (47, 59, 42) to (48, 60, 42)
EVEN_PAD = (0, 0, 1, 0, 1, 0)


def outsize_3d(s: int, kernel: int, dilation: int, stride: int, padding: int) -> int:
    """Get the output size for a dimension of size `s`"""
    p, d = padding, dilation
    k, r = kernel, stride
    return int(floor((s + 2 * p - d * (k - 1) - 1) / r + 1))


# see https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
# for padding logic
def padding_same(
    input_shape: Tuple[int, ...], kernel: int, dilation: int
) -> Tuple[int, int, int, int, int, int]:
    """Assumes symmetric (e.g. `(n, n, n)`) kernels, dilations, and stride=1"""

    def pad(k: int) -> Tuple[int, int]:
        p = max(k - 1, 0)
        p_top = p // 2
        p_bot = p - p_top
        return p_top, p_bot

    k = kernel + (kernel - 1) * (dilation - 1)  # effective kernel size
    pads: List[int] = []
    for dim in input_shape:
        pads.extend(pad(k))
    return tuple(pads)  # type: ignore
