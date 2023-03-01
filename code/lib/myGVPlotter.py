import numpy as np  # type: ignore
import gvar as gv  # type: ignore

colours = ['tab:blue', 'tab:green', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:orange', 'tab:red', 'tab:brown', 'tab:gray', 'tab:cyan']

def plot_gvEbar(x: np.ndarray, y: np.ndarray, ax, ma=None, ls=None, lab=None, col=None, alpha=1):
    """
    Just tidies up an error bar plot of a GV some
    """
    if col is None:
        ax.errorbar(x, y=gv.mean(y), yerr=gv.sdev(y), marker=ma, linestyle=ls, label=lab, alpha=alpha)  # noqa: E501
    else:
        ax.errorbar(x, y=gv.mean(y), yerr=gv.sdev(y), marker=ma, linestyle=ls, label=lab, color=col, alpha=alpha)  # noqa: E501
    return ax


def myHSpan(y: np.ndarray, ax, ls='--', colour=None, lab=None, alpha=1.0):
    """
    Just a small function to handle doing doing horizontal
    spans of gvars
    """
    if colour is None:
        ax.axhline(gv.mean(y), linestyle=ls, label=lab, alpha=alpha)
        col = ax.get_lines()[-1].get_color()
    else:
        col = colour
        ax.axhline(gv.mean(y), linestyle=ls, color=col, label=lab, alpha=alpha)
    ax.axhspan(gv.mean(y) - gv.sdev(y), gv.mean(y) + gv.sdev(y),
               alpha=alpha * 0.25, color=col, linewidth=0)
    return ax


def myVSpan(y: np.ndarray, ax, ls='--', colour=None, lab=None, alpha=1.0):
    """
    Just a small function to handle doing doing vertical
    spans of gvars
    """
    if colour is None:
        ax.axvline(gv.mean(y), linestyle=ls, label=lab, alpha=alpha)
        col = ax.get_lines()[-1].get_color()
    else:
        col = colour
        ax.axvline(gv.mean(y), linestyle=ls, color=col, label=lab, alpha=alpha)
    ax.axvspan(gv.mean(y) - gv.sdev(y), gv.mean(y) + gv.sdev(y),
               alpha=alpha * 0.25, color=col, linewidth=0)
    return ax


def myFill_between(x: np.ndarray, y: np.ndarray, ax, ma=None, ls=None, lab=None, alpha=0.15, colour=None):  # noqa: E501
    """
    Just a small function to handle doing doing fill_between
    for gvar
    """
    if colour is None:
        ax.plot(x, gv.mean(y), marker=ma, linestyle=ls, label=lab)
        col = ax.get_lines()[-1].get_color()
    else:
        col = colour
        ax.plot(x, gv.mean(y), marker=ma, linestyle=ls, color=col, label=lab)
    ax.fill_between(x, gv.mean(y) - gv.sdev(y), gv.mean(y) + gv.sdev(y),
                    alpha=alpha, color=col, linewidth=0)
    return ax
