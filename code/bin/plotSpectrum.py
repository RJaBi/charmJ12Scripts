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
insertPath = os.path.join(Path(__file__).resolve().parent, '..', 'lib')
sys.path.insert(0, insertPath)
import myModules as mo  # type: ignore # noqa: E402
import myGVPlotter as GVP  # type: ignore # noqa: E402


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

    if 'unit' in params.keys():
        unit = params['unit']
    else:
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
    pdfName = os.path.join(anaDir, 'spectrumPlot.pdf')
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
    # From csv if it exists
    if 'massCSV' in params['mAve'].keys():
        print(f"Loading the lattice csv from {params['mAve']['massCSV']}")
        massDF = pd.read_csv(params['mAve']['massCSV'])
        NT = params['mAve']['NT']
    if 'expCSV' in params.keys():
        print(f'loading the experimental csv from {params["expCSV"]}')
        expDF = pd.read_csv(params['expCSV'])
        expDF = expDF.rename(columns=lambda x: x.strip())
        expCSV = True
        # The experimental spreadsheet is in MeV
        if unit == 'GeV':
            PhysMult = 1.0/1000.0
        else:
            PhysMult = 1.0
    for aa, aLab in enumerate(params['mAve']['anaLab']):
        print(aa, aLab)
        thisDict = {}
        if expCSV:
            aDF = expDF.query('Operator == @aLab')
            EPAve = aDF['EPAve'].values[0]
            if EPAve != 'nan':
                thisDict.update({'eP': gv.gvar(EPAve) * PhysMult})
            EMAve = aDF['EMAve'].values[0]
            if EMAve != 'nan':
                thisDict.update({'eM': gv.gvar(EMAve) * PhysMult})
            thisDict.update({'J': aDF['J'].values[0]})
            thisDict.update({'quarks': aDF['quarks'].values[0]})
            name = aDF['name'].values[0] + 'J' + thisDict['J']
        else:
            # Getting the experimental numbers
            if params['eMass'][aLab]['P'] != '':
                thisDict.update({'eP': gv.gvar(params['eMass'][aLab]['P'])})
            if params['eMass'][aLab]['M'] != '':
                thisDict.update({'eM': gv.gvar(params['eMass'][aLab]['M'])})
            thisDict.update({'J': params['eMass'][aLab]['J']})
            thisDict.update({'quarks': params['eMass'][aLab]['quarks']})
            name = params['eMass'][aLab]['name'] + 'J' + thisDict['J']
        # Now load the lattice data
        if 'massCSV' in params['mAve'].keys():
            # Load the mass data from here
            aDF = massDF.query('Operator == @aLab')
            aDF = aDF.rename(columns=lambda x: x.strip())
            thisNT = int(NT)  # noqa: F841
            df = aDF.query('Nt == @thisNT')
            EP = df['EP'].values[0]
            EPSys = df['EPSys'].values[0]
            EM = df['EM'].values[0]
            EMSys = df['EMSys'].values[0]
            # Convert to physical
            EP = gv.gvar(np.asarray(EP)) * hbarc / a_t  # type: ignore
            EPSys = gv.gvar(np.asarray(EPSys)) * hbarc / a_t  # type: ignore
            EM = gv.gvar(np.asarray(EM)) * hbarc / a_t  # type: ignore
            EMSys = gv.gvar(np.asarray(EMSys)) * hbarc / a_t  # type: ignore
            thisDict.update({'lP': EP})
            thisDict.update({'lM': EM})
            thisDict.update({'lPSys': EPSys})
            thisDict.update({'lMSys': EMSys})
        else:
            # or load from individual fit methods
            posFile = os.path.join(params['mAve'][aLab]['pDir'], params['mAve'][aLab]['pFile'])
            posData = gv.load(posFile) * hbarc / a_t
            thisDict.update({'lP': posData[0]})
            # and for negative parity
            negFile = os.path.join(params['mAve'][aLab]['mDir'], params['mAve'][aLab]['mFile'])
            negData = gv.load(negFile) * hbarc / a_t
            thisDict.update({'lM': negData[1]})
        # and set the order
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
        qqq = dataDF['quarks'][count]
        if 'lPSys' in dataDF.keys():
            # if lPSys exists, so must lMSys
            lPSys = dataDF['lPSys'][count]
            lMSys = dataDF['lMSys'][count]
        xStr = '${}^{' + J + '}' + xV.split('J')[0] + '$'+'  ' + '\n$' + f'({qqq})$'
        if J == '1/2':
            colP = 'tab:blue'
            colM = 'tab:red'
        else:
            colP = 'tab:green'
            colM = 'tab:orange'
        colExp = 'tab:grey'
        # Plot the lattice data points
        ax = GVP.plot_gvEbar(xStr, lP, ax, col=colP, ma='d')
        ax = GVP.plot_gvEbar(xStr, lM, ax, col=colM, ma='*')
        if 'lPSys' in dataDF.keys():
            # Add the systematic if it exists
            ax = GVP.plot_gvEbar(xStr, lPSys, ax, col=colP, alpha=0.5)
            ax = GVP.plot_gvEbar(xStr, lMSys, ax, col=colM, alpha=0.5)
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
    # Add pos/neg parity symbols to legend manually
    line_dashed = mpl.lines.Line2D([], [], color='black', linestyle='', label='Positive Parity', marker='d', markersize=12)  # noqa: E501
    handles.append(line_dashed)
    labels.append('Positive Parity')
    # and neg
    line_dashed = mpl.lines.Line2D([], [], color='black', linestyle='', label='Negative Parity', marker='*', markersize=16)  # noqa: E501
    handles.append(line_dashed)
    labels.append('Negative Parity')
    ax.legend(handles, labels, loc='upper left', ncol=2)
    pdf.savefig(fig)
    # plt.show()
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
