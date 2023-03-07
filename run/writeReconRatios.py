"""
Code to take the base toml file in example-SU6-base.toml
and make it into a full toml file
This one only does simple ratios of each correlator in the multiplet
"""
# Loading some functions from elsewhere in this repo

import os
import sys
import toml
import gvar as gv  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
modDir = os.path.join(Path(__file__).resolve().parent, '..', 'code', 'lib')
sys.path.insert(0, modDir)
import myModules as mo  # type: ignore  # noqa: E402


def main(args: list):
    params = mo.GetArgs(args)
    # Printing the input toml back out
    toml.dump(params, f=sys.stdout)

    cfDirBase = params['cfuns']['cfDirBase']
    cfListBase = params['cfuns']['cfListBase']
    # Loop over NT
    cfLabels = []
    cfuns = params['cfuns']

    symMathLabels = []
    symMathMaths = []
    labelsToAnalyse = []
    xMins = []
    for NT in params['cfuns']['NT']:
        # These two just hold the maths for this temperature
        maths = []
        mathLabels = []
        # and doing the replacces, etc
        cfDirNT = cfDirBase.replace('NT', NT)
        cfListNT = cfListBase.replace('NT', NT)
        # loop over operator/qqq
        for OP, qqq in zip(params['cfuns']['operators'], params['cfuns']['quarks']):
            OPUN = OP.replace('.', '_')
            # print('\n', OP, qqq)
            cfDir = cfDirNT.replace('OP', OPUN.replace('_', '.')).replace('QQQ', qqq)
            cfListOPQ = cfListNT.replace('OP', OPUN.replace('_', '.')).replace('QQQ', qqq)
            # Loop over the cshifts
            # add to the dictionary
            for cc, cs in enumerate(params['cfuns']['cshifts']):
                cfList = cfListOPQ.replace('sX', f's{cs}')
                cfLabels.append(f'{OPUN}_{qqq}_{NT}s{cs}')
                thisDict = {'cfDir': cfDir,
                            'cfList': cfList,
                            'latt': int(NT),
                            'ext': ''}
                cfuns.update({cfLabels[-1]: thisDict})
                # Now average over cshift if that's what we need to do
                if params['maths']['averageCShift']:
                    if cc == 0:
                        thisMathList = f'M:{OPUN}_{qqq}_{NT}s{cs}'
                        thisMathMath = f'({OPUN}_{qqq}_{NT}s{cs}+'
                    elif cc == len(params['cfuns']['cshifts']) - 1:
                        thisMathMath = thisMathMath + f'{OPUN}_{qqq}_{NT}s{cs})/{len(params["cfuns"]["cshifts"])}'  # noqa: E501
                        thisMath = thisMathList + f',{OPUN}_{qqq}_{NT}s{cs}:' + thisMathMath
                        maths.append(thisMath)
                        mathLabels.append(f'{OPUN}_{qqq}_{NT}')
                        if params['maths']['negParity']:
                            maths.append(f'flip:{OPUN}_{qqq}_{NT}:{OPUN}_{qqq}_{NT}')
                            mathLabels.append(f'F_{OPUN}_{qqq}_{NT}')
                    else:
                        thisMathList = thisMathList + f',{OPUN}_{qqq}_{NT}s{cs}'
                        thisMathMath = thisMathMath + f'{OPUN}_{qqq}_{NT}s{cs}+'

        # Put the previous maths in
        symMathLabels.extend(mathLabels)
        symMathMaths.extend(maths)
    # Now do the model correlators
    # So first need to load zero temp masses in from the csv
    df = pd.read_csv(params['massCSV'])
    # Remove spaces in column names
    df.columns = df.columns.str.replace(' ', '')
    print(df.columns)
    # Grab the bits of the dataframe with the zero-temperature masses
    T0NT = params['cfuns']['T0NT']
    df = df.query(f'Nt == {T0NT}')

    # Do the ratios I guess
    for NT in params['cfuns']['NT'][1:]:
        # loop over operator/qqq
        for OP, qqq in zip(params['cfuns']['operators'], params['cfuns']['quarks']):
            OPUN = OP.replace(".", "_")
            # Construct Recon correlators
            opName = f'{OPUN}_{qqq}'
            try:
                EP = str(df.query(f'Operator == "{opName}"')['EP'].values[0])
                EM = str(df.query(f'Operator == "{opName}"')['EM'].values[0])
                if 'massFact' in params.keys():
                    EP = gv.gvar(EP) * params['massFact'][0]
                    EM = gv.gvar(EM) * params['massFact'][1]
            except IndexError:
                print(opName, df['Operator'])
                sys.exit('operator not found')
            # toyZeroName = f'Recon_{OPUN}_{qqq}_{NT}_{T0NT}'
            # toyZero = f'toy:{params["cfuns"]["hadronType"]}:[1.0(0),1.0(0)]_[{EP},{EM}]_{NT}_0_{NT}'  # noqa: E501
            toyShortName = f'Recon_{OPUN}_{qqq}_{NT}_{NT}'
            toyShort = f'recon:{OPUN}_{qqq}_{T0NT}:baryon:{NT}:min'
            # toyShort = f'toy:{params["cfuns"]["hadronType"]}:[1.0(0),1.0(0)]_[{EP},{EM}]_{NT}_0_{NT}'  # noqa: E501
            symMathLabels.append(toyShortName)
            symMathMaths.append(toyShort)
            if params['maths']['negParity']:
                symMathLabels.append(f'F_{toyShortName}')
                symMathMaths.append(f'flip:{toyShortName}:{toyShortName}')
            # Construct ReconRatio of actual to model
            ratShort = f'M:{OPUN}_{qqq}_{NT},{toyShortName}:{toyShortName}/{OPUN}_{qqq}_{NT}'
            ratShortName = f'ReconRatio_{OPUN}_{qqq}_{NT}_{NT}'
            symMathLabels.append(ratShortName)
            symMathMaths.append(ratShort)
            if params['maths']['negParity']:
                symMathLabels.append(f'F_{ratShortName}')
                symMathMaths.append(f'flip:{ratShortName}:{ratShortName}')
            # Do the double ratio
            # But first pad the shorter of them up to T0NT
            padShort = f'ext:{ratShortName}:{T0NT}_min_end'
            padShortName = f'{ratShortName}_{T0NT}'
            symMathLabels.append(padShortName)
            symMathMaths.append(padShort)
            if params['maths']['negParity']:
                symMathLabels.append(f'F_{padShortName}')
                symMathMaths.append(f'ext:F_{ratShortName}:{T0NT}_min_end')
            if params['maths']['double']:
                # First construct the t0 values if haven't got them already
                if T0NT not in params['cfuns']['NT'][1:]:
                    T0toyShortName = f'Recon_{OPUN}_{qqq}_{T0NT}_{T0NT}'
                    T0toyShort = f'recon:{OPUN}_{qqq}_{T0NT}:baryon:{T0NT}:min'
                    # T0toyShort = f'toy:{params["cfuns"]["hadronType"]}:[1.0(0),1.0(0)]_[{EP},{EM}]_{T0NT}_0_{T0NT}'  # noqa: E501
                    symMathLabels.append(T0toyShortName)
                    symMathMaths.append(T0toyShort)
                    if params['maths']['negParity']:
                        symMathLabels.append(f'F_{T0toyShortName}')
                        symMathMaths.append(f'flip:{T0toyShortName}:{T0toyShortName}')
                    # Construct ReconRatio of actual to model
                    T0ratShort = f'M:{OPUN}_{qqq}_{T0NT},{T0toyShortName}:{T0toyShortName}/{OPUN}_{qqq}_{T0NT}'  # noqa: E501
                    T0ratShortName = f'ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}'
                    symMathLabels.append(T0ratShortName)
                    symMathMaths.append(T0ratShort)
                    if params['maths']['negParity']:
                        T0ratShort = f'M:F_{OPUN}_{qqq}_{T0NT},F_{T0toyShortName}:F_{T0toyShortName}/F_{OPUN}_{qqq}_{T0NT}'  # noqa: E501
                        T0ratShortName = f'F_ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}'
                        symMathLabels.append(T0ratShortName)
                        symMathMaths.append(T0ratShort)
                # Now can construct the double ratio
                # Construct the double ratio
                ratDouble = f'M:{padShortName},ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}:{padShortName}/ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}'  # noqa: E501
                ratDoubleName = f'reconDouble_{OPUN}_{qqq}_{NT}_{T0NT}'
                symMathLabels.append(ratDoubleName)
                symMathMaths.append(ratDouble)
                if params['maths']['negParity']:
                    ratDouble = f'M:F_{padShortName},F_ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}:F_{padShortName}/F_ReconRatio_{OPUN}_{qqq}_{T0NT}_{T0NT}'  # noqa: E501
                    ratDoubleName = f'F_reconDouble_{OPUN}_{qqq}_{NT}_{T0NT}'
                    symMathLabels.append(ratDoubleName)
                    symMathMaths.append(ratDouble)
            if params['maths']['double']:
                # analyse the double ratio
                if params['maths']['negParity']:
                    xMins.append(f'F_{toyShortName}')
                    # xMins.append(f'F_{OPUN}_{qqq}_{NT}')
                else:
                    # xMins.append(f'{OPUN}_{qqq}_{NT}')
                    xMins.append(toyShortName)
                labelsToAnalyse.append(ratDoubleName)
            if params['maths']['single']:
                # Just analyse the single ratios
                if params['maths']['negParity']:
                    labelsToAnalyse.append(f'F_{ratShortName}')
                    # xMins.append(f'F_{toyShortName}')
                    xMins.append(f'F_{OPUN}_{qqq}_{NT}')
                else:
                    xMins.append(f'{OPUN}_{qqq}_{NT}')
                    # xMins.append(toyShortName)
                    labelsToAnalyse.append(ratShortName)
            if params['maths']['G']:
                # Just the correlator and model correlator
                if params['maths']['negParity']:
                    # xMins.append(f'F_{toyShortName}')
                    # xMins.append(f'F_{OPUN}_{qqq}_{NT}')
                    # xMins.append(f'F_{OPUN}_{qqq}_{NT}')
                    labelsToAnalyse.append(f'F_{toyShortName}')
                    labelsToAnalyse.append(f'F_{OPUN}_{qqq}_{NT}')
                else:
                    # xMins.append(toyShortName)
                    # xMins.append(f'{OPUN}_{qqq}_{NT}')
                    # xMins.append(f'{OPUN}_{qqq}_{NT}')
                    labelsToAnalyse.append(toyShortName)
                    labelsToAnalyse.append(f'{OPUN}_{qqq}_{NT}')
    if len(labelsToAnalyse) == 0:
        # Why are you doing this?
        labelsToAnalyse = cfLabels + symMathLabels
    for lab, math in zip(symMathLabels, symMathMaths):
        if lab in labelsToAnalyse:
            print(lab, math)
    # done assembling
    params['cfuns'].update({'cfLabels': cfLabels})
    params['analysis'].update({
        'symMathLabels': symMathLabels,
        'symMathMaths': symMathMaths,
        'labelsToAnalyse': labelsToAnalyse})
    if len(xMins) > 0:
        params['analysis'].update({'xMins': xMins})
        # 'anaName': anaNames})

    # Making the folder to put the toml in if necessary
    outDir = os.path.join(*params["outToml"].split('/')[:-1])
    print(f'OUTDIR IS !!!!!!! {outDir}')
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    print(f'Writing toml to {params["outToml"]}_Ratios.toml')
    with open(f'{params["outToml"]}_Ratios.toml', 'w') as f:
        toml.dump(params, f=f)
    sys.exit('Done')


if __name__ == '__main__':
    main(sys.argv[1:])
