"""
Plot as function of temperature
each temperature is a separate subplot
Simpler script which loads fits from massChoices.csv
"""

import numpy as np  # type: ignore
import os
import sys
import toml
import gvar as gv  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Any
import matplotlib.pyplot as plt  # type: ignore
# from matplotlib.container import ErrorbarContainer  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from matplotlib import rcParams  # type: ignore
# Grabbing some modules from elsewhere
from pathlib import Path
insertPath = os.path.join(Path(__file__).resolve().parent, '..', 'lib')
sys.path.insert(0, insertPath)
import myModules as mo  # type: ignore # noqa: E402
import myGVPlotter as GVP  # type: ignore # noqa: E402


def setYlim2Dim(params, ax, axScale, ytrim = 0):
    """
    Sets the ylims on each of the subplots
    such that they have the same scale
    Set the scale on ax, using the scale of axScale
    also returns the yScale
    only updates ymin, ymax if abs(ymin -ymax) > ytrim 
    """
    # First iterate over each axis
    # yScale is ymax - ymin
    yScale = 0
    # Middles is the middle point of each subplot
    middles = np.empty(np.shape(ax))
    yLims = np.empty(list(np.shape(ax)) + [2])
    # Determine the yscale
    for vv in range(0, np.shape(ax)[0]):
        for hh in range(0, np.shape(ax)[1]):
            # Need to consider the difference across all temperatures
            if hh == 0:
                ymin, ymax = axScale[vv, hh].get_ylim()
            else:
                thisYMin, thisYMax = axScale[vv, hh].get_ylim()
                if abs(thisYMin - thisYMax) > ytrim:
                    if thisYMax > ymax:
                        ymax = thisYMax
                    if thisYMin < ymin:
                        ymin = thisYMin
            hhvvYScale = abs(ymax - ymin)
            if hhvvYScale > yScale:
                yScale = hhvvYScale
            middles[vv, hh] = ymax - hhvvYScale / 2
            yLims[vv, hh, :] = ymin, ymax
            if 'ylim' in params.keys():
                ax[vv, hh].set_ylim(params['ylim'])
                # put y-axis tick labels on or not
                if hh == 0:
                    labelleft = True
                    labelright = False
                elif hh == np.shape(ax)[1] - 1:
                    labelleft = False
                    labelright = True
                else:
                    labelleft = False
                    labelright = False
                # and make ticks visible
                ax[vv, hh].yaxis.set_tick_params(
                    which='both',
                    left='on',
                    right='on',
                    direction='inout',
                    labelleft=labelleft,
                    labelright=labelright
                )
    if 'ylim' in params.keys():
        print('Setting manually')
        return ax
    # Now set ylims
    print(f'ysScale is {yScale}')
    for vv in range(0, np.shape(ax)[0]):
        vertMid = np.median(middles[vv, :])
        yMinVV = vertMid - yScale / 2
        yMaxVV = vertMid + yScale / 2
        # Determine the middle point
        for hh in range(0, np.shape(ax)[1]):
            ymin = vertMid - yScale / 2
            ymax = vertMid + yScale / 2
            # Shift it up/down if necessary
            while ymax < yLims[vv, hh, 1]:
                ymax = ymax + 0.05 * yScale / 2
                ymin = ymin + 0.05 * yScale / 2
            while ymin > yLims[vv, hh, 0]:
                ymax = ymax - 0.05 * yScale / 2
                ymin = ymin - 0.05 * yScale / 2
            if ymin < yMinVV or ymax > yMaxVV:
                yMaxVV = ymax
                yMinVV = ymin
        # and finally set it
        for hh in range(0, np.shape(ax)[1]):
            # put y-axis tick labels on or not
            if hh == 0:
                labelleft = True
                labelright = False
            elif hh == np.shape(ax)[1] - 1:
                labelleft = False
                labelright = True
            else:
                labelleft = False
                labelright = False
            # set the y limits
            ax[vv, hh].set_ylim([yMinVV, yMaxVV])
            # and make ticks visible
            ax[vv, hh].yaxis.set_tick_params(
                which='both',
                left='on',
                right='on',
                direction='inout',
                labelleft=labelleft,
                labelright=labelright
            )
    # and return
    return ax, yScale


def setYlim1Dim(params, ax, axScale, setY=True, ytrim = 0, right=True):
    """
    Sets the ylims on each of the subplots
    such that they have the same scale
    Set the scale on ax, using the scale of axScale
    also returns the yScale
    only updates ymin, ymax if abs(ymin -ymax) > ytrim 
    """

    # First iterate over each axis
    # yScale is ymax - ymin
    yScale = 0
    # Middles is the middle point of each subplot
    middles = np.empty(np.shape(ax))
    yLims = np.empty(list(np.shape(ax)) + [2])
    yLimsAX = np.empty(list(np.shape(ax)) + [2])
    # Determine the yscale
    for vv in range(0, np.shape(ax)[0]):
        # Need to consider the difference across all temperatures
        yminAX, ymaxAX = ax[vv].get_ylim()
        #print(yminAX, ymaxAX)
        if vv == 0:
            ymin, ymax = axScale[0].get_ylim()
        else:
            thisYMin, thisYMax = axScale[vv].get_ylim()
            if abs(thisYMin - thisYMax) > ytrim:
                if thisYMax > ymax:
                    ymax = thisYMax
                if thisYMin < ymin:
                    ymin = thisYMin
        hhvvYScale = abs(ymax - ymin)
        hhvvYScaleAX = abs(ymaxAX - yminAX)
        if hhvvYScale > yScale:
            yScale = hhvvYScale
        middles[vv] = ymaxAX - hhvvYScaleAX / 2
        yLims[vv, :] = ymin, ymax
        yLimsAX[vv, :] = yminAX, ymaxAX
        if 'ylim' in params.keys():
            print('Setting ylim from file')
            # put y-axis tick labels on or not
            if vv == 0:
                labelleft = True
                labelright = False
            elif vv == np.shape(ax)[0] - 1:
                labelleft = False
                if right:
                    labelright = True
                else:
                    labelRight = False
            else:
                labelleft = False
                labelright = False
            # and make ticks visible
            if setY:
                ax[vv].set_ylim(params['ylim'])
                ax[vv].yaxis.set_tick_params(
                    which='both',
                    left='on',
                    right='on',
                    direction='inout',
                    labelleft=labelleft,
                    labelright=labelright
                )
    if 'ylim' in params.keys():
        return ax
    # Now set ylims
    # print(f'ysScale is {yScale}')
    # print('yLims', yLims)
    # print('middles', middles)
    if 'yScale' in params.keys():
        yScale = float(params['yScale'])
    for vv in range(0, np.shape(ax)[0]):
        vertMid = np.median(middles[:])
        yMinVV = vertMid - yScale / 2
        yMaxVV = vertMid + yScale / 2
        # Determine the middle point
        ymin = vertMid - yScale / 2
        ymax = vertMid + yScale / 2
        # Shift it up/down if necessary
        while ymax < yLimsAX[vv, 1]:
            ymax = ymax + 0.05 * yScale / 2
            ymin = ymin + 0.05 * yScale / 2
        while ymin > yLimsAX[vv, 0]:
            ymax = ymax - 0.05 * yScale / 2
            ymin = ymin - 0.05 * yScale / 2
        if ymin < yMinVV or ymax > yMaxVV:
            yMaxVV = ymax
        yMinVV = ymin
        # and finally set it
        # put y-axis tick labels on or not
        if vv == 0:
            labelleft = True
            labelright = False
        elif vv == np.shape(ax)[0] - 1:
            labelleft = False
            if right:
                labelright = True
            else:
                labelRight = False
        else:
            labelleft = False
            labelright = False
        if setY:
            # set the y limits
            ax[vv].set_ylim([yMinVV, yMaxVV])
            # and make ticks visible
            ax[vv].yaxis.set_tick_params(
                which='both',
                left='on',
                right='on',
                direction='inout',
                labelleft=labelleft,
                labelright=labelright
            )
    # and return
    return ax, yScale


def main(args: list):
    """
    Reads some parameters in from the command line
    Puts that toml file in
    then reads it. Does a bunch of analysis
    """

    params = mo.GetArgs(args)
    # Printing the input toml back out - easier to read than printing as a dictionary
    toml.dump(params, f=sys.stdout)
    # refining ylims
    params = mo.refineXYLims(params, subDict=None)
    # For conversion to physical units
    hbarc = 0.1973269804  # GeV fm
    a_s = gv.gvar(params['latMass']['as'])
    xi = gv.gvar(params['latMass']['xi'])
    a_t = a_s/xi
    # Load the csv
    print(f"Loading the csv from {params['latMass']['massCSV']}")
    massDF = pd.read_csv(params['latMass']['massCSV'])
    print(massDF.head())
    # Set up where to save
    anaDir = os.path.join(params['latMass']['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)

    if 'norm' in params.keys():
        sys.exit('Norm not implemented yet')
        if params['norm'] == 'EP0':
            pdfMod = 'EP0'
        else:
            sys.exit(f'bad normamlisation norm={params["norm"]} selected. Exiting')
        pdfName = os.path.join(anaDir, f'singlePlotSep_norm_{pdfMod}.pdf')
        pdfNameM = os.path.join(anaDir, f'singlePlotSep_NegParity_norm_{pdfMod}.pdf')
        pdfNameB = os.path.join(anaDir, f'singlePlotSep_BothParity_norm_{pdfMod}.pdf')
        pdfNameDiff = os.path.join(anaDir, f'singlePlotSep_Diff_norm_{pdfMod}.pdf')
    else:
        pdfName = os.path.join(anaDir, 'singlePlotSep.pdf')
        pdfNameM = os.path.join(anaDir, 'singlePlotSep_NegParity.pdf')
        pdfNameB = os.path.join(anaDir, 'singlePlotSep_BothParity.pdf')
        pdfNameDiff = os.path.join(anaDir, 'singlePlotSep_Diff.pdf')
    # now this is hwere savign to
    print(f'Saving pdf to {pdfName} and {pdfNameM}')
    temperatures = params['latMass']['temperatures']
    # horz number
    numChannels = params['latMass']['numChannels']
    if numChannels > 1:
        sharey = False
    else:
        sharey = True
    # A plot for negative and positive parities
    fig, ax = plt.subplots(1, numChannels, figsize=(16.6, 11.6), sharey=sharey, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    figM, axM = plt.subplots(1, numChannels, figsize=(16.6, 11.6), sharey=sharey, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    # Both on same plot
    figB, axB = plt.subplots(1, numChannels, figsize=(16.6, 11.6), sharey=True, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    figDiff, axDiff = plt.subplots(1, numChannels, figsize=(16.6, 11.6), sharey=True, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    if numChannels == 1:
        ax = [ax]
        axM = [axM]
        axB = [axB]
        axDiff = [axDiff]
    # Iterate over different temperatures
    cols = params['latMass']['colours']
    marks = params['latMass']['markers']
    # for the legend
    allHandlesDict: Dict[int, List[Any]] = {}
    allLegendsDict: Dict[int, List[Any]] = {}
    expDict: Dict[int, List[bool]] = {}
    # Setting up the x-axis
    horzDict = {}
    horzCountDict = {}
    uniHorz = np.unique(params['latMass']['horzGroup'], return_counts=True)
    xRan = [0, 1]
    for ii in range(0, numChannels):
        if numChannels > 1:
            horzDict.update({uniHorz[0][ii]: uniHorz[1][ii]})
        else:
            horzDict.update({uniHorz[0][ii]: uniHorz[0][ii]})
        # For the legend
        allHandlesDict.update({uniHorz[0][ii]: []})
        allLegendsDict.update({uniHorz[0][ii]: []})
        expDict.update({uniHorz[0][ii]: [False, False]})
    for tt, temp in enumerate(temperatures):
        for ii in range(0, numChannels):
            horzCountDict.update({uniHorz[0][ii]: 1})
        NT = params['latMass']['NT'][tt]
        # and then over different hadrons
        for aa, aName in enumerate(params['latMass']['anaName']):
            thisVGroup = params['latMass']['horzGroup'][aa]
            if numChannels > 1:
                thisAX = ax[thisVGroup]
                thisAXM = axM[thisVGroup]
                thisAXB = axB[thisVGroup]
                thisAXDiff = axDiff[thisVGroup]
            elif numChannels == 1:
                thisAX = ax[0]
                thisAXM = axM[0]
                thisAXB = axB[0]
                thisAXDiff = axDiff[0]
            else:
                thisAX = ax[tt]
                thisAXM = axM[tt]
                thisAXB = axB[tt]
                thisAXDiff = axDiff[tt]
            # Doing the physical mass
            # as lines
            if params['eMass'][aName]['P'] != '':
                physEP = gv.gvar(params['eMass'][aName]['P'])
                expDict[thisVGroup][0] = True
            else:
                physEP = gv.gvar(None)
            if params['eMass'][aName]['M'] != '':
                physEM = gv.gvar(params['eMass'][aName]['M'])
                expDict[thisVGroup][1] = True
            else:
                physEM = gv.gvar(None)
            thisAX = GVP.myHSpan(physEP, thisAX, colour='tab:green', alpha=0.8)
            thisAXM = GVP.myHSpan(physEM, thisAXM, colour='tab:orange', alpha=0.8)
            thisAXB = GVP.myHSpan(physEP, thisAXB, colour='tab:green', alpha=0.8)
            thisAXB = GVP.myHSpan(physEM, thisAXB, colour='tab:orange', alpha=0.8, ls='-.')
            #thisAXB = GVP.myHSpan(physEP, thisAXB, colour=cols[aa], alpha=0.8)
            #thisAXB = GVP.myHSpan(physEM, thisAXB, colour=cols[aa], alpha=0.8)
            thisAXDiff = GVP.myHSpan(-1.0 * (physEP - physEM) / (physEM + physEP), thisAXDiff, colour=cols[aa], alpha=0.8)  # noqa: E501
            if NT not in params['latMass']['mALabels'][aa]:
                continue
            # print(aa, aName)
            aDF = massDF.query('Operator == @aName')
            aDF = aDF.rename(columns=lambda x: x.strip())
            # get the hadron name
            had = aDF['Hadron'].values[0][1:-1]  # Strip the $ from it
            thisNT = int(NT)  # noqa: F841
            df = aDF.query('Nt == @thisNT')
            EP = df['EP'].values[0]
            EPSys = df['EPSys'].values[0]
            if 'NegMALabels' in params['latMass'].keys():
                # Check if temp is ini these
                negTemps = params['latMass']['NegMALabels'][aa]
                if NT not in negTemps:
                    negMass = False
                else:
                    negMass = True
                    EM = df['EM'].values[0]
                    EMSys = df['EMSys'].values[0]
            else:
                negMass = True
                EM = df['EM'].values[0]
                EMSys = df['EMSys'].values[0]
            # Convert to physical
            EP = gv.gvar(np.asarray(EP)) * hbarc / a_t  # type: ignore
            EPSys = gv.gvar(np.asarray(EPSys)) * hbarc / a_t  # type: ignore
            if negMass:
                EM = gv.gvar(np.asarray(EM)) * hbarc / a_t  # type: ignore
                EMSys = gv.gvar(np.asarray(EMSys)) * hbarc / a_t  # type: ignore
            if tt == 0:
                lab = '$' + had + '$'
            else:
                lab = None
            # Put on the plot at some xcoords
            """
            # Width of the box
            width = abs(xRan[0] - xRan[1]) / (2 * horzDict[thisVGroup] + 1)
            xStart = xRan[0] + width * horzCountDict[thisVGroup]
            xStart = xStart - 0.15 * width
            xEnd = xRan[0] + width + width * horzCountDict[thisVGroup]
            xEnd = xEnd + 0.15 * width
            xMid = xStart + abs(xEnd - xStart) / 2
            """
            xMid = int(temp)
            xPlot = xMid + 0.025 * int(temperatures[0]) * aa
            horzCountDict[thisVGroup] = horzCountDict[thisVGroup] + 2
            # Fill in the legend
            if tt == 0:
                thisPointLabel = Line2D([], [], color=cols[aa], marker=marks[aa], linestyle='', label=lab, markersize=16)  # noqa: E501
                allHandlesDict[thisVGroup].append(thisPointLabel)
                allLegendsDict[thisVGroup].append(lab)
            # and now plot
            # pos parity
            thisAX = GVP.plot_gvEbar(xPlot+0.01*xPlot, physEP, thisAX, alpha=0, ma='D')
            thisAX = GVP.plot_gvEbar(xPlot, EP, thisAX, ma=marks[aa], ls='', col='tab:green')
            thisAX = GVP.plot_gvEbar(xPlot, EPSys, thisAX, ma=marks[aa], ls='', col='tab:green', alpha=0.5)
            # neg parity
            if negMass:
                thisAXM = GVP.plot_gvEbar(xPlot+0.01*xPlot, physEM, thisAXM, alpha=0, ma='D')
                thisAXM = GVP.plot_gvEbar(xPlot, EM, thisAXM, ma=marks[aa], ls='', col='tab:orange')
                thisAXM = GVP.plot_gvEbar(xPlot, EMSys, thisAXM, ma=marks[aa], ls='', col='tab:orange', alpha=0.5)
            # Both on same plot
            thisAXB = GVP.plot_gvEbar(xPlot, EP, thisAXB, ma=marks[aa], ls='', col='tab:green')
            thisAXB = GVP.plot_gvEbar(xPlot, EPSys, thisAXB, ma=marks[aa], ls='', col='tab:green', alpha=0.5)
            thisAXB.get_xaxis().set_ticks([50,100,150,200])
            thisAXB.set_xticklabels([50,100,150,200], rotation=315)
            thisAXB.set_xlim([30,210])
            thisAXDiff.get_xaxis().set_ticks([50,100,150,200])
            thisAXDiff.set_xticklabels([50,100,150,200], rotation=315)
            thisAXDiff.set_xlim([30,210])
            # neg parity
            if negMass:
                thisAXB = GVP.plot_gvEbar(xPlot, EM, thisAXB, ma=marks[aa], ls='', col='tab:orange')
                thisAXB = GVP.plot_gvEbar(xPlot, EMSys, thisAXB, ma=marks[aa], ls='', col='tab:orange', alpha=0.5)
                thisAXB.plot(xPlot, gv.mean(EM), marker=marks[aa], ls='', alpha=1.0, color='tab:orange', markerfacecolor='white', zorder=10)
            # the difference
            if negMass:
                thisAXDiff = GVP.plot_gvEbar(xPlot, ((EM - EP) / (EM + EP)), thisAXDiff, ma=marks[aa], ls='', col=cols[aa])
                thisAXDiff = GVP.plot_gvEbar(xPlot, ((EMSys - EPSys) / (EMSys + EPSys)), thisAXDiff, ma=marks[aa], ls='', col=cols[aa], alpha=0.5)
            thisAXDiff.axhline(y=0, linestyle=':', alpha=0.2, color='black')
        # outside loop
        fig.supxlabel('Temperature (MeV)')
        fig.supylabel('Mass (GeV)')
        for vs in range(0, numChannels):
            had = allLegendsDict[vs][0][1:-1]
            if numChannels > 1:
                ax[vs].text(0.15, 0.8, '$' + f'{had}' + '$', transform=ax[vs].transAxes, usetex=True)  # noqa: E501
                axM[vs].text(0.15, 0.85, '$' + f'{had}' + '$', transform=axM[vs].transAxes, usetex=True)  # noqa: E501
                axB[vs].text(0.15, 0.90, '$' + f'{had}' + '$', transform=axB[vs].transAxes, usetex=True)  # noqa: E501
                axDiff[vs].text(0.15, 0.85, '$' + f'{had}' + '$', transform=axDiff[vs].transAxes, usetex=True)  # noqa: E501
            else:
                ax[vs].text(0.15, 0.8, '$' + f'{had}' + '$', transform=ax[vs].transAxes, usetex=True)  # noqa: E501
                axM[vs].text(0.15, 0.85, '$' + f'{had}' + '$', transform=axM[vs].transAxes, usetex=True)  # noqa: E501
                axB[vs].text(0.15, 0.90, '$' + f'{had}' + '$', transform=axB[vs].transAxes, usetex=True)  # noqa: E501
                axDiff[vs].text(0.15, 0.85, '$' + f'{had}' + '$', transform=axDiff[vs].transAxes, usetex=True)  # noqa: E501
        # Removing xticks
        
        figM.supxlabel('Temperature (MeV)')
        figM.supylabel('Mass (GeV)')
        # Removing xticks FOR BOTH
        figB.supxlabel('Temperature (MeV)', y=0.05)
        figB.tight_layout()
        figB.supylabel('Mass (GeV)')
        # and for the diff
        figDiff.supxlabel('Temperature (MeV)',y=0.05)
        figDiff.supylabel('$\\frac{M^{-} - M^{+}}{M^{-} + M^{+}}$', x=0.05)

    posExpLine = False
    negExpLine = False
    for k, v in expDict.items():
        if v[0]:
            posExpLine = True
        if v[1]:
            negExpLine = True
        
    # Doing the legend
    allHandles = []
    allLegends = []
    line_dashed = Line2D([], [], color='tab:green', linestyle='--', label='$P=+$ Exp.')  # noqa: E501
    point_pos= Line2D([0], [0], color='tab:green', linestyle='', marker='D', label='$P=+$')  # noqa: E501
    allHandles.append(point_pos)
    allLegends.append('Pos. Par. Lat.')
    if posExpLine:
        allHandles.append(line_dashed)
        allLegends.append('Pos. Par. Exp.')
    axB[0].legend(allHandles, allLegends, handlelength=1.5, loc='lower center', frameon=False)  # noqa: E501
    allHandles = []
    allLegends = []
    line_dashed = Line2D([], [], color='tab:orange', linestyle='-.', label='$P=-$ Exp.')  # noqa: E501
    point_pos= Line2D([0], [0], color='tab:orange', linestyle='', marker='D', label='$P=-$', markerfacecolor='white')  # noqa: E501
    allHandles.append(point_pos)
    allLegends.append('Neg. Par. Lat.')
    if negExpLine:
        allHandles.append(line_dashed)
        allLegends.append('Neg. Par. Exp.')
    axB[1].legend(allHandles, allLegends, handlelength=1.5, loc='lower center', frameon=False)  # noqa: E501
    diffLegends = []
    diffHandles = []
    line_dashed = Line2D([], [], color='black', linestyle='--')  # noqa: E501
    point_pos= Line2D([0], [0], color='black', linestyle='', marker='D')  # noqa: E501
    diffHandles.append(point_pos)
    diffLegends.append('Lat.')
    if negExpLine and posExpLine:
        diffHandles.append(line_dashed)
        diffLegends.append('Exp.')
    axDiff[0].legend(diffHandles, diffLegends, handlelength=1.5, loc='lower center', frameon=False)  # noqa: E501            

    # Now determine  and set y-limits and y-ticks
    # determine scales indivudually
    # so can set according to the largest
    if numChannels > 1:
        ax, axScale = setYlim1Dim(params, ax, ax)
        axM, axMScale = setYlim1Dim(params, axM, axM, ytrim=0.2)
        print('axScale, axMScale', axScale, axMScale)
        skip = True
        if not skip:
            if axScale > axMScale:
                ax, axScale = setYlim1Dim(params, ax, ax)
                axM, axMScale = setYlim1Dim(params, axM, ax)
            else:
                ax, axScale = setYlim1Dim(params, ax, axM)
                axM, axMScale = setYlim1Dim(params, axM, axM)
        # and set axDiff separately
        # doesn't work if set on ax currently
        axDiff = setYlim1Dim(params, axDiff, axDiff, right=False)
        axB = setYlim1Dim(params, axB, axB)
    elif numChannels == 1:
        ax, axScale = setYlim1Dim(params, ax, ax, setY=False, ytrim=0.2)
        axM, axMScale = setYlim1Dim(params, axM, axM, ytrim=0.2, setY=False)
        print('axScale, axMScale', axScale, axMScale)
        skip = False
        if not skip:
            if axScale > axMScale:
                ax, axScale = setYlim1Dim(params, ax, ax, ytrim=0.2)
                axM, axMScale = setYlim1Dim(params, axM, ax, ytrim=0.2)
            else:
                ax, axScale = setYlim1Dim(params, ax, axM)
                axM, axMScale = setYlim1Dim(params, axM, axM)
        # and set axDiff separately
        # doesn't work if set on ax currently
        axDiff = setYlim1Dim(params, axDiff, axDiff)
        axB = setYlim1Dim(params, axB, axB)
    else:
        ax, axScale = setYlim1Dim(params, ax, ax, setY=False)
        axM, axMScale = setYlim1Dim(params, axM, axM, setY=False)
        if axScale > axMScale:
            ax, axScale = setYlim1Dim(params, ax, ax)
            axM, axMScale = setYlim1Dim(params, axM, ax)
        else:
            ax, axScale = setYlim1Dim(params, ax, axM)
            axM, axMScale = setYlim1Dim(params, axM, axM)
        # and set axDiff separately
        # doesn't work if set on ax currently
        axDiff = setYlim1Dim(params, axDiff, axDiff)
        axB = setYlim1Dim(params, axB, axB)
    # if numChannels > 1:
    #     ax[0, 0].set_xlim(xRan)
    # else:
    #     ax[0].set_xlim(xRan)
    fig.tight_layout()
    fig.savefig(pdfName)
    plt.close(fig)
    figM.tight_layout()
    figM.savefig(pdfNameM)
    plt.close(figM)
    figB.tight_layout()
    figB.savefig(pdfNameB)
    plt.close(figB)
    figDiff.tight_layout()
    figDiff.savefig(pdfNameDiff)
    plt.close(figDiff)
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    rcParams['lines.markersize'] = 8
    rcParams['font.size'] = 32
    rcParams['ytick.labelsize'] = 32
    rcParams['lines.markersize'] = 12.0
    main(sys.argv[1:])
