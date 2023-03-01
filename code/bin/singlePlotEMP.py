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
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.container import ErrorbarContainer  # type: ignore
# Grabbing some modules from elsewhere
from pathlib import Path
insertPath = os.path.join(Path(__file__).resolve().parent, '..','lib')
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
        pdfName = os.path.join(anaDir, f'singleEMPPlot_norm_{pdfMod}.pdf')
        pdfNameM = os.path.join(anaDir, f'singleEMPPlot_NegParity_norm_{pdfMod}.pdf')
    else:
        pdfName = os.path.join(anaDir, f'singleEMPPlot.pdf')
        pdfNameM = os.path.join(anaDir, f'singleEMPPlot_NegParity.pdf')
    # now this is hwere savign to
    print(f'Saving pdf to {pdfName} and {pdfNameM}')
    temperatures = params['latMass']['temperatures']
    # A plot for negative and positive parities
    fig, ax = plt.subplots(1, len(temperatures), figsize=(16.6, 11.6), sharey=True, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    figM, axM = plt.subplots(1, len(temperatures), figsize=(16.6, 11.6), sharey=True, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})  # noqa: E501
    # Iterate over different temperatures
    cols = params['latMass']['colours']
    marks = params['latMass']['markers']
    for tt, temp in enumerate(temperatures):
        NT = params['latMass']['NT'][tt]
        # print(tt, temp, NT)
        thisAX = ax[tt]
        thisAXM = axM[tt]
        # and then over different hadrons
        for aa, aName in enumerate(params['latMass']['anaName']):
            # Doing the physical mass
            # as lines
            if params['eMass'][aName]['P'] != '':
                physEP = gv.gvar(params['eMass'][aName]['P'])
            else:
                physEP = gv.gvar(None)
            if params['eMass'][aName]['M'] != '':
                physEM = gv.gvar(params['eMass'][aName]['M'])
            else:
                physEM = gv.gvar(None)
            thisAX = GVP.myHSpan(physEP, thisAX, colour=cols[aa], alpha=0.3)
            thisAXM = GVP.myHSpan(physEM, thisAXM, colour=cols[aa], alpha=0.3)
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
            EM = df['EM'].values[0]
            EMSys = df['EMSys'].values[0]
            # Convert to physical
            EP = gv.gvar(np.asarray(EP)) * hbarc / a_t  # type: ignore
            EPSys = gv.gvar(np.asarray(EPSys)) * hbarc / a_t  # type: ignore
            EM = gv.gvar(np.asarray(EM)) * hbarc / a_t  # type: ignore
            EMSys = gv.gvar(np.asarray(EMSys)) * hbarc / a_t  # type: ignore
            if tt == 0:
                lab = '$' + had + '$'
            else:
                lab = None
            # Put on the plot
            thisAX = GVP.plot_gvEbar(aa, EP, thisAX, ma=marks[aa], ls='', col=cols[aa], lab=lab)
            thisAX = GVP.plot_gvEbar(aa, EPSys, thisAX, ma=marks[aa], ls='', alpha=0.5, col=cols[aa])  # noqa: E501
            thisAXM = GVP.plot_gvEbar(aa, EM, thisAXM, ma=marks[aa], ls='', col=cols[aa], lab=lab)
            thisAXM = GVP.plot_gvEbar(aa, EMSys, thisAXM, ma=marks[aa], ls='', alpha=0.5, col=cols[aa])  # noqa: E501
        # Get the labels from tt == 0
        if tt == 0:
            handles, legendLabels = thisAX.get_legend_handles_labels()
            # also the symbol only
            hLeg = [h[0] if isinstance(h, ErrorbarContainer) else h for h in handles]
        if tt == len(temperatures) - 1:
            thisAX.legend(hLeg, legendLabels, bbox_to_anchor=(1.2, 1), borderaxespad=0, handlelength=0)  # noqa: E501
            thisAXM.legend(hLeg, legendLabels, bbox_to_anchor=(1.8, 1), borderaxespad=0, handlelength=0)  # noqa: E501
        thisAX.set_xlabel(f'{temp}')
        thisAX.set_xticklabels([])
        fig.supxlabel('Temperature (MeV)')
        ax[0].set_ylabel('Mass (GeV)')
        thisAXM.set_xlabel(f'{temp}')
        thisAXM.set_xticklabels([])
        figM.supxlabel('Temperature (MeV)')
        axM[0].set_ylabel('Mass (GeV)')
    # plt.show()
    if 'ylim' in params.keys():
        ax[0].set_ylim(params['ylim'])
        axM[0].set_ylim(params['ylim'])
    fig.savefig(pdfName)
    plt.close(fig)
    figM.savefig(pdfNameM)
    plt.close(figM)
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    main(sys.argv[1:])
