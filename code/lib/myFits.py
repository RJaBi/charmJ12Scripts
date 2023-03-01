
from typing import List
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


def getWinTicks(ax, wins: List[str]) -> List[str]:
    """
    Returns a list of ticklabels where only the start t0 is labelled
    """
    # Getting all the ticks
    startTicks = []
    # Just a list of the start points
    starts = []
    # and how many fit windows for that start
    counts = []
    for ii, w in enumerate(wins):
        wstart = int(w.split('-')[0])
        if ii == 0:
            start = wstart
            startTicks.append(str(wstart))
            starts.append(str(wstart))
            counts.append(1)
        else:
            if wstart > start:
                start = wstart
                startTicks.append(str(wstart))
                starts.append(str(wstart))
                counts.append(1)
            else:
                startTicks.append('')
                counts[-1] = counts[-1] + 1
    # So now startTicks is a list, putting the startpoint at the 1st window
    # We want to move it to the middle of that region
    middles = np.asarray(counts)/2
    # So do a new startTicks
    startTicks = []
    # and also put in the actual middle, not just closest
    xticks = []
    for ss, st in enumerate(starts):
        startTicks.append(st)
        xPos = np.sum(counts[:ss]) + middles[ss]
        xticks.append(xPos)
    # vertically offsetting every 2nd labelled
    tickCount = 0
    for tt, st in enumerate(startTicks):
        if st == '':
            continue
        if tickCount % 2 != 0:
            startTicks[tt] = '\n' + st
        tickCount = tickCount + 1
    # Changing the ticks to only be the ones we want
    ax.set_xticks(xticks)
    return startTicks


# As suggested in 2007.04188
def arctan(x, p):
    return p['c0'] + p['c1'] * np.arctan(p['c2'] * (x - p['Tpc']))
