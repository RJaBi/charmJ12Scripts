
import numpy as np  # type: ignore
import os
import sys
import toml
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
import gvar as gv  # type: ignore
# from typing import Dict, Any
from pathlib import Path
# Loading some functions from elsewhere in this repo
insertPath = os.path.join(Path(__file__).resolve().parent, '..','lib')
sys.path.insert(0, insertPath)
import myIO as io  # type: ignore  # noqa: E402
import myModules as mo  # type: ignore  # noqa: E402
import myGVPlotter as GVP  # type: ignore  # noqa: E402


def main(args: list):
    """
    Reads some parameters in from the command line
    Puts that toml file in
    then reads it. Does a bunch of analysis
    """

    params = mo.GetArgs(args)
    #  Printing the input toml back out - easier to read than printing as a dictionary
    toml.dump(params, f=sys.stdout)
    # Setting x, limits to None if they were 'None' in the toml
    params = mo.refineXYLims(params)

    # Loading correlators
    cfDict, cfLabelsList = io.initCorrelatorsGvar(params)
    # where we put the analysis
    anaDir = os.path.join(params['analysis']['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)

    # Save averaged
    # Save all labels
    if 'nameMod' in params.keys():
        name = params['nameMod'] + '_'
    else:
        name = ''
    io.saveCSV(anaDir, f'{name}all', cfDict)
    # Save only labelsToAnalyse
    # Saves only if they exist
    io.saveCSV(anaDir, f'{name}ana', cfDict, params['analysis']['labelsToAnalyse'])
    # Now that we've got everything we want.
    # Time to do analysis.
    # Tells us if we only plotted some of the data
    plotAll = True
    # If plot only some, we actually plot all (rest is faded)
    # but use the x, y limits from only some
    ylims = []
    xlims = []
    fig, ax = plt.subplots(figsize=(16.6, 11.6))
    for aa, ana in enumerate(params['analysis']['labelsToAnalyse']):
        if ana not in cfLabelsList:
            print('label ' + ana + ' is not in cfLabelsList:', cfLabelsList)
            sys.exit('Exiting')
        # Now do the analysis
        G = cfDict['data'][ana]
        # Taking mean and 2nd order jackknifes
        # GMean = np.mean(G, axis=0)
        # GJ2 = oldMo.doJack(G, 2)
        # Also do the plot for each analysis separately
        thisAnaDir = os.path.join(anaDir, ana)
        if not os.path.exists(thisAnaDir):
            os.makedirs(thisAnaDir)
        figAna, axAna = plt.subplots(figsize=(16.6, 11.6))
        if 'xMins' in params['analysis'].keys():
            plotXMin = np.argmin(cfDict['data'][params['analysis']['xMins'][aa]])
            plotAll = False
        else:
            plotXMin = len(G)  # type: ignore
        if 'xStart' in params['analysis'].keys():
            plotXStart = int(params['analysis']['xStart'])
            plotAll = False
        else:
            plotXStart = 0
        # Do plot
        x = np.asarray(range(1, len(G)+1))
        if not plotAll:
            mark = 'D'
            mpl.rcParams['lines.markersize'] = 8.0
        else:
            mark = 'd'
        if 'labelsToPlot' in params['analysis'].keys():
            myLab = params['analysis']['labelsToPlot'][aa]
        else:
            myLab = ana
        if 'xlabel' in params['analysis'].keys():
            myXLab = params['analysis']['xlabel']
        else:
            myXLab = '$\\tau / a_\\tau$'
        if 'ylabel' in params['analysis'].keys():
            myYLab = params['analysis']['ylabel']
        else:
            myYLab = '$\\mathcal{O}(\\tau)$'  # noqa: W605
        ax = GVP.plot_gvEbar(x[plotXStart:plotXMin], G[plotXStart:plotXMin], ax, ma=mark, ls='', lab=myLab)  # type: ignore  # noqa: E501
        axAna = GVP.plot_gvEbar(x[plotXStart:plotXMin], G[plotXStart:plotXMin], axAna, ma=mark, ls='', lab=myLab)  # type: ignore  # noqa: E501
        axAna.set_xlabel(myXLab)
        axAna.set_ylabel(myYLab)
        axAna.legend(loc='best', ncol=2)

        # Plot the rest of the data faded
        # Reset the ylimits back to the subset of data
        # This gets overridden by GyLim, GxLim
        if not plotAll:
            xlims.append(axAna.get_xlim())
            ylims.append(axAna.get_ylim())
            col = ax.get_lines()[-1].get_c()
            # First plot the data with some error bars and alpha
            ax = GVP.plot_gvEbar(x[plotXMin:], G[plotXMin:], ax, ma=mark, ls='', alpha=0.35, col=col, mac=(1,1,1,1))  # type: ignore  # noqa: E501
            # Now make the marker face white
            # This removes the line from crossing the marker face
            ax.plot(x[plotXMin:], gv.mean(G[plotXMin:]), marker=mark, linestyle='', alpha=1.0, color=col, markerfacecolor='white')  # type: ignore
            # Similarly for the single plot
            col = axAna.get_lines()[-1].get_c()
            axAna = GVP.plot_gvEbar(x[plotXMin:], G[plotXMin:], axAna, ma=mark, ls='', alpha=0.35, col=col, mac='none')  # type: ignore  # noqa: E501
            axAna.plot(x[plotXMin:], gv.mean(G[plotXMin:]), marker=mark, linestyle='', alpha=1.0, color=col, markerfacecolor='white')  # type: ignore
            axAna.set_ylim(ylims[-1])
            axAna.set_ylim(xlims[-1])
        axAna.set_xlim(params['analysis']['GxLim'])
        axAna.set_ylim(params['analysis']['GyLim'])
        figAna.savefig(os.path.join(thisAnaDir, 'G.pdf'))
        plt.close(figAna)
    # Outside that loop
    # ax.axhline(y=0.0, color='black', linestyle='--')
    # ax.axhline( y=1.0, color='black', linestyle='--')
    # add the labels, etc
    if not plotAll:
        # Changing the x,y limits back
        ymin = ylims[0][0]
        ymax = ylims[0][1]
        for yl in ylims[1:]:
            if yl[0] < ymin:
                ymin = yl[0]
            if yl[1] > ymax:
                ymax = yl[1]
        ax.set_ylim([ymin, ymax])
        xmin = xlims[0][0]
        xmax = xlims[0][1]
        for xl in xlims[1:]:
            if xl[0] < xmin:
                xmin = xl[0]
            if xl[1] > xmax:
                xmax = xl[1]
        ax.set_xlim([xmin, xmax])
    # and back to the rest
    ax.set_xlabel(myXLab)
    ax.set_ylabel(myYLab)
    ax.legend(loc='best', ncol=2, fontsize=28)
    ax.set_xlim(params['analysis']['GxLim'])
    ax.set_ylim(params['analysis']['GyLim'])
    # plt.show()
    if 'nameMod' in params.keys():
        name = f'G_{params["nameMod"]}.pdf'
    else:
        name = 'G.pdf'
    fig.savefig(os.path.join(anaDir, name))
    plt.close()


if __name__ == '__main__':
    mo.initBigPlotSettings()
    mpl.rcParams['font.size'] = 28
    main(sys.argv[1:])
