"""
Plots the spectrum of hadrons from lattice
with experimental values
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import os
import sys
# import pickle
import toml
import gvar as gv  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
# Grabbing some modules from elsewhere
from pathlib import Path
insertPath = os.path.join(Path(__file__).resolve().parent, '..','lib')
sys.path.insert(0, insertPath)
import myModules as mo  # type: ignore # noqa: E402


def main(args: list):
    """
    Reads some parameters in from the command line
    Puts that toml file in
    then reads it. Does a bunch of analysis
    """

    params = mo.GetArgs(args)
    # Printing the input toml back out - easier to read than printing as a dictionary
    toml.dump(params, f=sys.stdout)
    # Setting x, limits to None if they were 'None' in the toml
    params = mo.refineXYLims(params, subDict=None)

    unit = params['eMass']['unit']
    # For conversion to physical units
    hbarc = 0.1973269804  # GeV fm
    a_s = gv.gvar(params['mAve']['as'])
    xi = gv.gvar(params['mAve']['xi'])
    a_t = a_s/xi

    # Set up where to save
    anaDir = os.path.join(params['mAve']['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)
    pdfName = os.path.join(anaDir, f'spectrumPlot.pdf')
    print(f'Saving pdf to {pdfName}')
    pdf = PdfPages(pdfName)
    # ylimit of plots
    if 'ylim' in params.keys():
        ylim = params['ylim']
    else:
        ylim = [None, None]
    # Storing the data after we load it
    data = {}
    # Load the data
    for aa, aLab in enumerate(params['mAve']['anaLab']):
        print(aa, aLab)
        thisDict = {}
        # Getting the experimental numbers
        if params['eMass'][aLab]['P'] != '':
            thisDict.update({'eP': gv.gvar(params['eMass'][aLab]['P'])})
        if params['eMass'][aLab]['M'] != '':
            thisDict.update({'eM': gv.gvar(params['eMass'][aLab]['M'])})
        thisDict.update({'J': params['eMass'][aLab]['J']})
        name = params['eMass'][aLab]['name'] + 'J' + thisDict['J']
        # Now load the lattice data
        posFile = os.path.join(params['mAve'][aLab]['pDir'], params['mAve'][aLab]['pFile'])
        posData = gv.load(posFile) * hbarc / a_t
        thisDict.update({'lP': posData[0]})
        # and for negative parity
        negFile = os.path.join(params['mAve'][aLab]['mDir'], params['mAve'][aLab]['mFile'])
        negData = gv.load(negFile) * hbarc / a_t
        thisDict.update({'lM': negData[1]})
        thisDict.update({'order': params['mAve'][aLab]['order']})
        data.update({name: thisDict})
    # Put it into a dataframe for easier handling
    # Sorting to ensure we're increasing in mass
    dataDF = pd.DataFrame.from_dict(data, orient='index').sort_values(by='order')
    # and do some plots
    fig, ax = plt.subplots(figsize=(16.6, 11.6))
    # xStr = ['$'+ii+'$' for ii in dataDF.index]
    # The distance between items
    xStep = 1.0/(len(dataDF.index)*0.5) * float(params['xStepMod'])
    # Iterate over each 'row'
    count = 0
    # have we put an experiment label in?
    expLabel = False
    for xV, row in dataDF.iterrows():
        print(count, xV)
        # order = dataDF['order'][count]
        x0 = count - xStep
        x1 = count + xStep
        lP = dataDF['lP'][count]
        eP = dataDF['eP'][count]
        lM = dataDF['lM'][count]
        eM = dataDF['eM'][count]
        J = dataDF['J'][count]
        xStr = '${}^{' + J + '}' + xV.split('J')[0] + '$'+'  '
        if J == '1/2':
            colP = 'tab:blue'
            colM = 'tab:red'
        else:
            colP = 'tab:green'
            colM = 'tab:orange'
        colExp = 'tab:grey'
        # Plot the lattice data points
        if count == 0:
            ax.errorbar(xStr, y=gv.mean(lP), yerr=gv.sdev(lP), color=colP, marker='d', label='Positive Parity')  # noqa: E501
            ax.errorbar(xStr, y=gv.mean(lM), yerr=gv.sdev(lM), color=colM, marker='*', label='Negative Parity')  # noqa: E501
        else:
            ax.errorbar(xStr, y=gv.mean(lP), yerr=gv.sdev(lP), color=colP, marker='d')
            ax.errorbar(xStr, y=gv.mean(lM), yerr=gv.sdev(lM), color=colM, marker='*')
        # Plot the experimental data points
        if not np.isnan(gv.mean(eP)):
            if not expLabel:
                ax.fill_between([x0, x1], gv.mean(eP) - gv.sdev(eP), gv.mean(eP) + gv.sdev(eP), alpha=1.0, color=colExp, label='Experiment')  # , linewidth=0)  # noqa: E501
                expLabel = True
            else:
                ax.fill_between([x0, x1], gv.mean(eP) - gv.sdev(eP), gv.mean(eP) + gv.sdev(eP), alpha=1.0, color=colExp)  # , linewidth=0)  # noqa: E501
        if not np.isnan(gv.mean(eM)):
            ax.fill_between([x0, x1], gv.mean(eM) - gv.sdev(eM), gv.mean(eM) + gv.sdev(eM), alpha=1.0, color=colExp)  # , linewidth=0)  # noqa: E501
        # add to the count
        count = count + 1
    # finalise plots
    ax.set_ylim(ylim)
    ax.set_xlabel('Baryon')
    ax.set_ylabel(f'Mass ({unit})')
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, mpl.container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, loc='upper left', ncol=2)
    pdf.savefig(fig)
    plt.show()
    pdf.close()
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['errorbar.capsize'] = 10
    mpl.rcParams['lines.markeredgewidth'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 28
    mpl.rcParams['xtick.labelsize'] = 28
    mpl.rcParams['font.size'] = 28

    main(sys.argv[1:])
