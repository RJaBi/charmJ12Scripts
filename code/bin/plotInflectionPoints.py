"""
Just plots the inflection points from the csv
"""

import pandas as pd  # type: ignore
import sys
import os
import toml
import gvar as gv  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl
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
    # refining ylims
    params = mo.refineXYLims(params, subDict=None)

    # Load the csv
    inflectDFList = []
    for csv in params['inflectCSV']:
        print(f"Loading the csv from {csv}")
        inflectDF = pd.read_csv(csv)
        # Removing white space in column names
        inflectDF = inflectDF.rename(columns=lambda x: x.strip())
        print(inflectDF.head())
        inflectDFList.append(inflectDF)
    # Set up where to save
    anaDir = os.path.join(params['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)
    PsiTpc = gv.gvar(params['Tpc'])
    fig, ax = plt.subplots(figsize=(16.6, 11.6))
    ax = GVP.myHSpan(PsiTpc, ax, alpha=0.8, lab=params['TpcLab'])
    xLabels = []
    for dd, df in enumerate(inflectDFList):
        # plt.gca().set_prop_cycle(None)
        print(dd, df)
        if params['TpcType'] == 'MeV':
            Tpc = gv.gvar(df['MeV'].values, df['MeVErr'].values)
            yScale = ' (MeV)'
        elif params['TpcType'] == 'TTpc':
            Tpc = gv.gvar(df['TTpc'].values, df['TTpcErr'].values)
            yScale = ''
        else:
            sys.exit(f'bad TpcType = {params["TpcType"]}')
        # xAxis = df['ana'].values
        xAxis = df['symb'].values
        leftScript = params['leftSuperScript'][dd]
        for xx, xV in enumerate(xAxis):
            xAxis[xx] = '${}^{' + leftScript + '}' + xV[1:] + '$'
        markers = GVP.markers
        colCount = 0
        for x, y, mark in zip(xAxis, Tpc, markers):
            if y < 0.8 * PsiTpc or y > 1.2 * PsiTpc:
                continue
            #if 'c' not in x:
            #    # skipping C=0
            #    # cycle the colour
            #    next_colour = next(ax._get_lines.prop_cycler)['color']  # noqa: F841
            #    continue
            xLabels.append(x)
            ax = GVP.plot_gvEbar(x, y, ax, ma=mark, ls='', col=GVP.colours[colCount])
            colCount = colCount + 1
            if colCount > len(GVP.colours) - 1:
                colCount = 0
    # finalise plot
    ax.set_xticklabels(xLabels, rotation=0)
    ax.set_ylabel('$T_{c}$' + yScale)
    ax.set_xlabel('Baryon')
    ax.legend(loc='best', ncol=2, fontsize=40)
    plt.savefig(os.path.join(anaDir, 'inflectionPoints.pdf'))
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    mpl.rcParams['lines.markersize'] = 28.0
    # For Poster/Presentation
    mpl.rcParams['ytick.labelsize'] = 32
    mpl.rcParams['xtick.labelsize'] = 36
    mpl.rcParams['font.size'] = 36
    mpl.rcParams['errorbar.capsize'] = 5  # making error bar caps slightly wider
    main(sys.argv[1:])
