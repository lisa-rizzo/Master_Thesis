"""Shared plotting style. Usage in every notebook:
    import plotstyle; plotstyle.use()
    fig, ax = plt.subplots(figsize=plotstyle.figsize(0.49))
    plotstyle.savefig(fig, "name")
"""
from pathlib import Path
import shutil, warnings
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 6.5
GOLDEN = 0.618
_HERE = Path(__file__).resolve().parent
_STYLE = _HERE / "lisathesis.mplstyle"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

def use():
    plt.style.use(_STYLE)
    if shutil.which("latex") is None:
        warnings.warn("LaTeX not found - falling back to STIX fonts. "
                      "Figures will NOT exactly match the thesis font.")
        mpl.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        })

def figsize(frac=1.0, aspect=GOLDEN):
    """Width as fraction of \\textwidth; height = width * aspect."""
    w = TEXTWIDTH_IN * frac
    return (w, w * aspect)

def savefig(fig, name, **kwargs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", **kwargs)

def fix_axes(fig, legends=(), like=None, pad=0.04):
    """Lay out `fig` with `legends` left out of the size calculation, freeze that layout, then grow the
    figure at the bottom until the legends fit below it. The axes boxes are therefore the same as in a
    figure of the same size without legends. Returns the frozen layout; pass it as `like` to give another
    figure of the same size exactly the same boxes. Call after plotting, before saving."""
    legends = [leg for leg in legends if leg is not None]
    if like is None:
        fig_legs = [leg for leg in legends if leg in fig.legends]
        for leg in legends:
            leg.set_in_layout(False)
        for leg in fig_legs:              # constrained layout keeps room for "outside" figure legends regardless
            fig.legends.remove(leg)
        fig.draw_without_rendering()
        fig.legends.extend(fig_legs)
        like = {"size": tuple(fig.get_size_inches()), "boxes": [ax.get_position().bounds for ax in fig.axes]}
    W, H = like["size"]
    assert abs(fig.get_figwidth() - W) < 1e-6 and len(fig.axes) == len(like["boxes"]), "figure differs from `like`"
    fig.set_layout_engine("none")
    fig.set_size_inches(W, H)
    for ax, box in zip(fig.axes, like["boxes"]):
        ax.set_position(box)
    fig.draw_without_rendering()
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    extra = 0.0
    for leg in legends:
        bb = leg.get_window_extent(r)
        # figure legend (lower center): needs a strip up to its top; axes legend: only its overhang below 0
        extra = max(extra, (bb.y1 if leg.parent is fig else -bb.y0) / dpi + pad)
    if extra > 0:
        fig.set_size_inches(W, H + extra)
        for ax, (x, y, w, h) in zip(fig.axes, like["boxes"]):
            ax.set_position([x, (y * H + extra) / (H + extra), w, h * H / (H + extra)])
    return like

def inset_box(ax, width, height, pad=0.01):
    """[x0, y0, w, h] in axes fraction for an image inset in the top-left corner, or in the top-right
    corner if the lines already plotted on `ax` run through more of the top-left box.
    Call after plotting and after setting the axis limits."""
    import numpy as np
    ax.get_xlim(), ax.get_ylim()                          # make sure autoscaled limits are final
    x_left, x_right, y0 = pad, 1 - pad - width, 1 - pad - height
    to_axes = ax.transAxes.inverted()
    t = np.linspace(0, 1, 25)[:, None]
    pts = [to_axes.transform(ax.transData.transform(np.concatenate([a + (b - a) * t for a, b in zip(xy[:-1], xy[1:])])))
           for xy in (line.get_xydata() for line in ax.get_lines()) if len(xy) > 1]
    if not pts:
        return [x_left, y0, width, height]
    p = np.concatenate(pts)
    p = p[(p[:, 1] >= y0) & (p[:, 1] <= 1)]                # only the visible part inside the box's height
    hits = lambda x0: int(((p[:, 0] >= x0) & (p[:, 0] <= x0 + width)).sum())
    return [x_right, y0, width, height] if hits(x_right) < hits(x_left) else [x_left, y0, width, height]
