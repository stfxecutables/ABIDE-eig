import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ants import image_read

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants.paths import DATA, EIGS

ATLAS_DIR = DATA / "atlases"
ATLAS = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND = ATLAS_DIR / "CC400_ROI_labels.csv"


def plot4d(nii: Path, vmin: int = 5, vmax: int = 95, norm: str = "diff") -> None:
    img = image_read(str(nii)).numpy()
    extensions = "".join(nii.suffixes)
    eigs = np.load(EIGS / nii.name.replace(extensions, ".npy").replace("_eigimg", ""))
    if norm == "diff":
        img = eigs - img
    elif norm == "div":
        img = eigs / img
    else:
        pass
    ts = np.linspace(0, img.shape[-1] - 1, 5, dtype=int)
    xs = [int(img.shape[0] * 0.333), img.shape[0] // 2, int(img.shape[0] * 0.666)]
    ys = [int(img.shape[1] * 0.333), img.shape[1] // 2, int(img.shape[1] * 0.666)]
    zs = [int(img.shape[2] * 0.333), img.shape[2] // 2, int(img.shape[2] * 0.666)]
    fig: plt.Figure
    fig, axes = plt.subplots(ncols=3 * len(xs), nrows=len(ts) + 1)
    # CMAP = "Greys"
    CMAP = "magma"
    for i, t in enumerate(ts):
        # fmt: off
        im = img[xs[0], :, :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][0].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[xs[1], :, :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][1].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[xs[2], :, :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][2].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, ys[0], :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][3].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, ys[1], :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][4].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, ys[2], :, t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][5].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, :, zs[0], t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][6].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, :, zs[1], t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][7].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        im = img[:, :, zs[2], t]; vmn, vmx = np.nanpercentile(im, [vmin, vmax])  # noqa
        axes[i][8].imshow(im, cmap=CMAP, vmin=vmn, vmax=vmx)
        # fmt: on
        fig.text(0.025, 0.8 - 0.15 * i, f"t = {t}")

    ax: plt.Axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(nii.stem)
    fig.set_size_inches(w=16, h=12)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img", type=Path)
    parser.add_argument("--vmax", type=float, default=95)
    parser.add_argument("--vmin", type=float, default=5)
    parser.add_argument("--norm", choices=["", "div", "diff"], default=5)

    args = parser.parse_args()
    path = args.img
    vmin, vmax = args.vmin, args.vmax
    plot4d(path, vmin, vmax, args.norm)
