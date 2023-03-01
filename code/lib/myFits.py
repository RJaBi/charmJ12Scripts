
from typing import List, Dict, Any
import numpy as np


def makeFitWins(fitMin: int, fitMax: int, Nt: int, twoSided=True):
    # Making all the fit windows
    mins = range(fitMin, fitMax)
    # Fit windows on left hand side of correlator
    lFitWinStart = []
    lFitWinEnd = []
    # Fit windows on right hand side of correlator
    rFitWinStart = []
    rFitWinEnd = []
    for mi in mins:
        for xma in range(mi + 1, fitMax + 1):
            lFitWinStart.append(mi)
            lFitWinEnd.append(xma)
            rFitWinStart.append((Nt - 0) - xma)
            rFitWinEnd.append((Nt - 0) - mi)
        # Putting into a single variable of tuples
    if twoSided:
        fWin = zip(lFitWinStart, lFitWinEnd, rFitWinStart, rFitWinEnd)
    else:
        fWin = zip(lFitWinStart, lFitWinEnd)  # type: ignore
    return fWin


def plotWinLines(ax, wins: List[str]):
    """
    Plots a vertical line where the start of the fit window changes
    Assumes that wins is sorted.
    """
    for ii, w in enumerate(wins):
        wstart = int(w.split('-')[0])
        if ii == 0:
            start = wstart
            ax.axvline(x=w, linestyle='--', alpha=0.25, color='black')
        else:
            if wstart > start:
                start = wstart
                ax.axvline(x=w, linestyle='--', alpha=0.25, color='black')
        # This is just to get spacing right for some fits
        # It's alpha=0 i.e. fully transparent
        ax.axvline(x=w, linestyle='--', alpha=0.0, color='black')
    return ax


def linFunc(x: np.ndarray, p: Dict[str, Any]):
    """
    A linear fit function suitable for lsqfit
    """
    return p['a0'] + p['a1'] * x


def sqrtLinFunc(x: np.ndarray, p: Dict[str, Any]):
    """
    A sqrt(linear) fit function suitable for lsqfit
    """
    return np.sqrt(p['a0'] + p['a1'] * x)
