
import gvar as gv  # type: ignore
import pandas as pd  # type: ignore
import toml
import os
import sys
# import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path
from typing import Any, Tuple, List
import numpy as np  # type: ignore
# Loading some functions from elsewhere in this repo
insertPath = os.path.join(Path(__file__).resolve().parent, '..', 'lib')
sys.path.insert(0, insertPath)
import myModules as mo  # type: ignore  # noqa: E402
# import myGVPlotter as GVP  # type: ignore  # noqa: E402


def cleanColumnNames(massDF: pd.DataFrame):  # -> Dict[str: str]:
    """
    returns clean strings to get the columns names of the dataframe
    as pandas apparently doesn't like column names with latex in it
    specialised for this csv
    """
    columns = {}
    for kk in massDF.columns:
        if 'Hadron' in kk:
            columns.update({'Hadron': kk})
        elif 'Operator' in kk:
            columns.update({'Operator': kk})
        elif 'tau' in kk:
            columns.update({'Nt': kk})
        elif '+' in kk:
            if 'Method' in kk:
                columns.update({'EP_Method': kk})
            else:
                columns.update({'EP': kk})
        elif '-' in kk:
            if 'Method' in kk:
                columns.update({'EM_Method': kk})
            else:
                columns.update({'EM': kk})
        else:
            columns.update({kk: kk})
    return columns


def getSysErr(thisAnaDir: str, EMethod: str, fTypes: List[str], parity: str) -> Tuple[Any, float, Any, List[str], Any]:  # noqa: E501
    """
    Loads the chosen/central value fit
    loads the other fittypes
    gets systematic error from spread of fits
    returns the central value, sys err on its own, central value with incorporated sys err
    and a list of the strings of the other fittypes and also their values
    """

    if parity == '+':
        fitIndex = 0
    elif parity == '-':
        fitIndex = 1
    else:
        sys.exit(f'bad parity = {parity}')
    # now load the chosen fit type
    EAveFile = os.path.join(thisAnaDir, f'fit_{EMethod}_Aved.gvar')
    E = gv.load(EAveFile)[fitIndex]
    EFTypes = [x for x in fTypes if x != EMethod]
    # Iterate over the other fit types, load those values in
    EAves = []
    for ftype in EFTypes:
        EAveFile = os.path.join(thisAnaDir, f'fit_{ftype}_Aved.gvar')
        EAves.append(gv.load(EAveFile)[fitIndex])
    # making it numpy
    EAves = np.asarray(EAves)  # type: ignore
    sysErr = gv.mean(np.max(np.fabs(EAves - E)))  # type: float
    ESys = gv.gvar(gv.mean(E), np.sqrt(gv.sdev(E)**2.0 + sysErr**2.0))
    return E, sysErr, ESys, EFTypes, EAves


def main(args: list):
    """
    Reads some parameters in from the command line
    Puts that toml file in
    then reads it. Does a bunch of analysis
    """
    params = mo.GetArgs(args)
    #  Printing the input toml back out - easier to read than printing as a dictionary
    toml.dump(params, f=sys.stdout)
    if 'massCSV' in params.keys():
        # Load the initial csv in which has the fit type we want in it
        massDF = pd.read_csv(params['massCSV'])
        print(massDF.head())
        # clean the column names cause pandas doesn't like latex
        columns = cleanColumnNames(massDF)
    # This is for the output
    csvDict = {}  # type: ignore
    csvDict.update({'anaDir': []})
    csvDict.update({'Nt': []})
    csvDict.update({'Operator': []})
    csvDict.update({'Hadron': []})
    csvDict.update({'EP': []})
    csvDict.update({'EPMethod': []})
    csvDict.update({'EP_Sys': []})
    csvDict.update({'EPSys': []})
    csvDict.update({'EPFtypes': []})
    csvDict.update({'EPFits': []})
    csvDict.update({'EM': []})
    csvDict.update({'EMMethod': []})
    csvDict.update({'EM_Sys': []})
    csvDict.update({'EMSys': []})
    csvDict.update({'EMFtypes': []})
    csvDict.update({'EMFits': []})
    # Iterate over the different hadron/temperature in the toml
    for ma in params['modelAverages']['mALabels']:
        Nt = params['modelAverages'][ma]['Nt']
        OP = params['modelAverages'][ma]['operator']
        EPfTypes = params['modelAverages'][ma]['EPfTypes']
        EMfTypes = params['modelAverages'][ma]['EMfTypes']
        if EPfTypes[0] == 'all':
            EPfTypes = params['modelAverages']['allFTypes']
        if EMfTypes[0] == 'all':
            EMfTypes = params['modelAverages']['allFTypes']
        if 'massCSV' in params.keys():
            # Cut the spreadsheet to this specific hadron/temperature
            # and get the methods chosen for EP and EM
            NTDF = massDF.loc[massDF[columns['Nt']] == Nt]
            OPNTDF = NTDF.loc[NTDF[columns['Operator']] == OP]
            EPMethod = OPNTDF[columns['EP_Method']].values[0]
            EMMethod = OPNTDF[columns['EM_Method']].values[0]
        else:
            # Or use method specified in the toml
            EPMethod = params['modelAverages'][ma]['EPMethod']
            EMMethod = params['modelAverages'][ma]['EMMethod']
        # This is where all the analysis stored for this hadron/temperature
        thisAnaDir = os.path.join(params['modelAverages'][ma]['anaDir'], f'{OP}_{Nt}x32')
        # cmd = 'ls ' + thisAnaDir
        # print(cmd)
        # os.system(cmd)
        # Loading the central value
        EP, EP_sysErr, EPSys, EPFTypes, EPFits = getSysErr(thisAnaDir, EPMethod, EPfTypes, '+')
        # print(EP, EP_sysErr, EPSys, EPFTypes, EPFits)
        EM, EM_sysErr, EMSys, EMFTypes, EMFits = getSysErr(thisAnaDir, EMMethod, EMfTypes, '-')
        # print(EM, EM_sysErr, EMSys, EMFTypes, EMFits)
        # Update the csvDict for saving
        csvDict['anaDir'].append(thisAnaDir)
        csvDict['Nt'].append(Nt)
        csvDict['Operator'].append(OP)
        if 'massCSV' in params.keys():
            # Use the label in the csv
            csvDict['Hadron'].append(OPNTDF[columns['Hadron']].values[0])
        else:
            # Use the label in the toml
            csvDict['Hadron'].append(params['modelAverages'][ma]['Hadron'])
        csvDict['EP'].append(EP)
        csvDict['EPMethod'].append(EPMethod)
        csvDict['EP_Sys'].append(EP_sysErr)
        csvDict['EPSys'].append(EPSys)
        csvDict['EPFtypes'].append(EPFTypes)
        csvDict['EPFits'].append(EPFits)
        csvDict['EM'].append(EM)
        csvDict['EMMethod'].append(EMMethod)
        csvDict['EM_Sys'].append(EM_sysErr)
        csvDict['EMSys'].append(EMSys)
        csvDict['EMFtypes'].append(EMFTypes)
        csvDict['EMFits'].append(EMFits)
    # Put the csvDict into a dataframe for saving
    csvSave = pd.DataFrame(csvDict)
    print(csvSave.head())
    if 'massCSV' in params.keys():
        outCSV = os.path.join(params['anaDir'], params['massCSV'].split('.csv')[0] + '_sysErr.csv')  # noqa: E501
    else:
        outCSV = os.path.join(params['anaDir'], 'mass_sysErr.csv')  # noqa: E501
    print(f'Saving csv to {outCSV}')
    csvSave.to_csv(outCSV)
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    main(sys.argv[1:])
