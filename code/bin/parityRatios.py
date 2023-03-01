
import os
import sys
import toml
import numpy as np
import gvar as gv  # type: ignore
import lsqfit as lsq  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import List, Any, Tuple, Dict
from pathlib import Path
modDir = os.path.join(Path(__file__).resolve().parent, '..', 'lib')
sys.path.insert(0, modDir)
import myModules as mo  # type: ignore  # noqa: E402
import myIO as io  # type: ignore  # noqa: E402
import myGVPlotter as GVP  # type: ignore  # noqa: E402
from myFits import arctan


def calcSummedRatio(GVD: np.ndarray, n0Arr: np.ndarray) -> np.ndarray:
    """
    Constructs
    R(n0) = (sum_n0^(1/2NT - 1){ GVD / sigma(GVD)**2.0}) / sum_n0^(1/2NT - 1){1/sigma(GVD)**2.0}
    for each R(n0)
    """
    stop = int(len(GVD) / 2) - 1
    R = np.empty(len(n0Arr), dtype='object')
    for nn, n0 in enumerate(n0Arr):
        num = np.sum(GVD[n0:stop] / (gv.sdev(GVD[n0:stop])**2.0))
        den = np.sum(1.0 / (gv.sdev(GVD[n0:stop])**2.0))
        R[nn] = num / den
    return R


def uniqueIndices(l: List) -> List:
    """
    Get the indices ofthe unique elements in l
    return that
    from
    https://stackoverflow.com/questions/56830995/find-the-indexes-of-unique-elements-of-a-list-in-python  # noqa: E501
    """
    seen = set()
    res = []
    for i, n in enumerate(l):
        if n not in seen:
            res.append(i)
            seen.add(n)
    return res


def gvIsClose(gv1, gv2, per=0.20):
    """
    returns True if gv1 is within per percent of gv2
    Both ways
    """
    gv1Max = gv.mean(gv1 + 0.2 * gv1)
    gv1Min = gv.mean(gv1 - 0.2 * gv1)
    gv2Max = gv.mean(gv2 + 0.2 * gv2)
    gv2Min = gv.mean(gv2 - 0.2 * gv2)
    close = True
    if gv1 > gv2Max:
        close = False
    elif gv1 < gv2Min:
        close = False
    elif gv2 > gv1Max:
        close = False
    elif gv2 < gv1Min:
        close = False
    return close


def getInflectionPoint(params, XPoints: np.ndarray, yData: np.ndarray, minSearch: float, tS: np.ndarray, extraMethods=['SCIPY', 'FIT']) -> Tuple[Any, gv.gvar]:  # noqa: E501
    """
    Calculates the inflection point from the spline
    Does a couple of checks that the answer is sensible
    by calculating the inflection point in multiple ways
    Returns the spline (gvar) and the inflection point
    """

    # Check what scale for Tpc we are in
    # if not present set to 1
    # i.e. are we in T or T/Tpc
    if 'TpcScale' in params['analysis'].keys():
        TpcScale = params['analysis']['TpcScale']
    else:
        TpcScale = 1.0
    # Calculate the spline
    gvSpline = gv.cspline.CSpline(XPoints, yData, alg='cspline', extrap_order=3)
    inflects = {}
    inflectTypes = []
    try:
        # Sometimes the gv method fails
        # The second derivative gives the inflection point
        def derivFunc(x):
            return gvSpline.D2(x) - 0
        # # for two different interval
        interval = gv.root.search(derivFunc, minSearch)
        myInterval = [interval[0], interval[1]]
        if interval[1] > 1.075 * TpcScale:
            print(interval, interval[1])
            myInterval[1] = 1.05 * TpcScale
        GVInflectVal = gv.root.refine(derivFunc, myInterval)
        # print(f'minSearch = {minSearch}, GV inflection Point = {GVInflectVal}')
        GV = True
        inflects.update({'GV': GVInflectVal})
        inflectTypes.append('GV')
    except ValueError:
        GV = False
    # Do the other methods
    for meth in extraMethods:
        if meth == 'SCIPY':
            ncon = 2000
            inflectEval = np.empty(ncon)
            bootIter = gv.bootstrap_iter(yData)
            splineCoeffs = np.empty([len(XPoints), ncon])  # type: ignore
            deriv2Coeffs = np.empty([len(XPoints) - 2, ncon])  # type: ignore
            for icon in range(0, ncon):
                RtnIcon = gv.mean(next(bootIter))
                spSpline = InterpolatedUnivariateSpline(XPoints, RtnIcon, k=3)
                # Also keep the coeffs
                splineCoeffs[:, icon] = spSpline.get_coeffs()
                deriv2 = spSpline.derivative(2)
                # Also keep the coeffs of the spline
                deriv2Coeffs[:, icon] = deriv2.get_coeffs()
                # Now evaluate it
                deriv2Eval = deriv2.__call__(tS)
                zeroArgMin = np.argmax(abs(deriv2Eval-0))
                zeroClosestVal = tS[zeroArgMin]
                # Putting in a shape 2 thing for the jackknife error code
                inflectEval[icon] = zeroClosestVal
            # Make it into a gvar
            scipyInflectVal = gv.dataset.avg_data(inflectEval, spread=True)
            inflects.update({'SCIPY': scipyInflectVal})
            inflectTypes.append('SCIPY')
        elif meth == 'FIT':
            # Try an arctan fit
            # Priors chosen to give approximately correct shape
            prior = {}
            if TpcScale == 1.0:
                prior['c0'] = gv.gvar(1.0, 0.7)
                prior['c1'] = gv.gvar(-0.5, 1.0)
                prior['c2'] = gv.gvar(10, 10)
                prior['Tpc'] = gv.gvar(1, 0.1)
            else:
                prior['c0'] = gv.gvar(1.0, 0.7)
                prior['c1'] = gv.gvar(-0.5, 1.0)
                prior['c2'] = gv.gvar(0.10, 0.10)
                prior['Tpc'] = gv.gvar(1 * TpcScale, 0.5 * TpcScale)
            try:
                # fit = lsq.nonlinear_fit(data=(XPoints, yData), fcn=arctan, prior=prior)  # noqa: E501
                # print(fit.format(maxline=True))
                # fitInflectVal = fit.p['Tpc']
                # Fit only the points of the spline around where expect Tpc to be
                nearX = mo.find_nearest(XPoints, TpcScale)
                # print(f'nearX = {nearX}')
                if nearX - 2 < 0:
                    xStart = 0
                    xEndMod = abs(nearX-2)
                else:
                    xStart = nearX - 2
                    xEndMod = 0
                if nearX + 3 + xEndMod > len(XPoints):  # type: ignore
                    xEnd = len(XPoints)  # type: ignore
                else:
                    xEnd = nearX + 3 + xEndMod
                fitXRange = XPoints[xStart: xEnd]
                fitYRange = yData[xStart: xEnd]
                fit = lsq.nonlinear_fit(data=(fitXRange, fitYRange), fcn=arctan, prior=prior)  # noqa: E501
                # print(fit.format(maxline=True))
                fitInflectVal = fit.p['Tpc']
                inflects.update({'FIT': fitInflectVal})
                inflectTypes.append('FIT')
            except ValueError:
                print('Value Error in fit. Probably residuals not finite')
                print('Using scipyInflectVal for fitInflectVal')
        else:
            sys.exit(f'Unsupported extra Inflection Point method {meth}')
    # Check if the other methods produce close values
    compVal = inflects[inflectTypes[0]]
    close = []
    print('compVal', compVal)
    for lab, val in inflects.items():
        if lab == inflectTypes[0]:
            continue
        print('    ', lab, val, gvIsClose(compVal, val, per=0.1))
        thisClose = gvIsClose(compVal, val, per=0.1)
        close.append(thisClose)
    if np.any(np.asarray(close)):
        finalInflectVal = compVal
    else:
        finalInflectVal = inflects['SCIPY']
    # print(finalInflectVal)
    return gvSpline, finalInflectVal

    

def main(args: list):
    """
    Reads some parameters in from the command line
    Puts that toml file in
    then reads it. Does a bunch of analysis
    """
    seed = gv.ranseed(9)  # noqa: F841
    params = mo.GetArgs(args)
    # Printing the input toml back out
    toml.dump(params, f=sys.stdout)
    # Setting x, limits to None if they were 'None' in the toml
    params = mo.refineXYLims(params)

    # Loading correlators
    if np.any(['resample' in x for x in params['analysis']['symMathMaths']]):
        resample = True
        cfDict, cfLabelsList, resampleDict = io.initCorrelators(params)
    else:
        resample = False
        cfDict, cfLabelsList = io.initCorrelators(params)
        
    # where we put the analysis
    anaDir = os.path.join(params['analysis']['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)

    # Check what scale for Tpc we are in
    # if not present set to 1
    # i.e. are we in T or T/Tpc
    if 'TpcScale' in params['analysis'].keys():
        TpcScale = params['analysis']['TpcScale']
    else:
        TpcScale = 1.0
        
    figRn0, axRn0 = plt.subplots(figsize=(16.6, 11.6))
    pdf = PdfPages(os.path.join(anaDir, 'SummedParityRatio.pdf'))
    colCount = 0
    markCount = 0
    allAnaOps = []
    RCP: Dict[str, Dict[str, np.ndarray]] = {}
    for taa, thisAna in enumerate(params['analysis']['labelsToAnalyse']):
        # thisRR: Dict[str, np.ndarray] = {}
        figTau, axTau = plt.subplots(figsize=(16.6, 11.6))
        figR, axR = plt.subplots(figsize=(16.6, 11.6))
        figRn0Ana, axRn0Ana = plt.subplots(figsize=(16.6, 11.6))
        thisAnaLab = params['cfuns']['hadrons'][taa]
        thisAnaOp = '_'.join(thisAna[0].split('_')[:-1])
        # print(thisAna, thisAnaLab, thisAnaOp)
        allAnaOps.append(thisAnaOp)
        # Also do the plot for each analysis separately
        thisAnaDir = os.path.join(anaDir, thisAnaOp)
        if not os.path.exists(thisAnaDir):
            os.makedirs(thisAnaDir)
        thisPdf = PdfPages(os.path.join(thisAnaDir, f'ParityRatioPlots_{thisAnaOp}.pdf'))
        RCP.update({thisAnaOp: {}})
        for nn, n0 in enumerate(params['analysis']['RCurves']):
            RCP[thisAnaOp].update({n0: np.empty(len(thisAna), dtype='object')})
        for aa, ana in enumerate(thisAna):
            if ana not in cfLabelsList:
                print('label ' + ana + ' is not in cfLabelsList:', cfLabelsList)
                sys.exit('Exiting')
            # Now do the analysis
            if resample:
                if ana in cfDict['data'].keys():
                    G = io.labelNorm(cfDict['data'][ana], params)
                    # Taking mean and cov
                    GVD = gv.dataset.avg_data(G)
                elif ana in resampleDict.keys():
                    GVD = io.labelNorm(resampleDict[ana], params)
            else:
                G = io.labelNorm(cfDict['data'][ana], params)
                # Taking mean and cov
                GVD = gv.dataset.avg_data(G)                
            # Get Mean
            GMean = gv.mean(GVD)
            # Now just plot the quantity (which is the mathcal(R) ratio)
            xPlot = int(len(GMean) / 2)
            tRange = np.asarray(range(0, len(GMean)))[:xPlot]
            axTau = GVP.plot_gvEbar(tRange, GVD[:xPlot], axTau, ma='d', ls='', lab=ana)
            # Now calculate the summed Ratio
            RRatio = calcSummedRatio(GVD, np.asarray(range(0, xPlot - 1)))
            axR = GVP.plot_gvEbar(range(0, xPlot - 1), RRatio, axR, ma='d', ls='', lab=ana)
            # and plot just the point for each temperature
            xPoint = params['analysis']['RXPoints'][aa]
            if colCount > len(GVP.colours) - 1:
                colCount = 0
            if markCount > len(mo.markers) - 1:
                markCount = 0
            n0ColCount = 0
            for nn, n0 in enumerate(params['analysis']['RCurves']):
                if n0ColCount > len(GVP.colours) - 1:
                    n0ColCount = 0
                axRn0Ana = GVP.plot_gvEbar(xPoint, RRatio[n0], axRn0Ana, ma=mo.markers[markCount], ls='', col=GVP.colours[n0ColCount], lab='$n_0=' + str(n0) + '$')  # noqa: E501
                RCP[thisAnaOp][n0][aa] = RRatio[n0]
                n0ColCount = n0ColCount + 1
                if nn == 0:
                    axRn0 = GVP.plot_gvEbar(xPoint, RRatio[n0], axRn0, ma=mo.markers[markCount], ls='', col=GVP.colours[colCount])  # noqa: E501  # lab='$' + thisAnaLab + '$'
        # Incrementing the colours counter
        colCount = colCount + 1
        markCount = markCount + 1
        axTau.set_ylabel('$\\mathcal{R}(\\tau)$')
        axTau.set_xlabel('$\\tau/a_\\tau$')
        axTau.set_ylim([-1, 1])
        axTau.legend(loc='best', ncol=2)
        axR.set_ylabel('$R(n_0)$')
        axR.set_xlabel('$n_0$')
        axR.set_ylim([-2, 2])
        axR.legend(loc='best', ncol=2)
        axRn0Ana.set_ylabel('$R(n_0)$')
        axRn0Ana.set_xlabel('Temperature (MeV)')
        # Removing duplicate labels
        handles, legendLabels = axRn0Ana.get_legend_handles_labels()
        uniqueInd = uniqueIndices(legendLabels)
        uniHand = [handles[i] for i in uniqueInd]
        uniLegLab = [legendLabels[i] for i in uniqueInd]
        axRn0Ana.legend(uniHand, uniLegLab, loc='best', ncol=2)
        # Saving the figures
        # axTau.set_title('axTau')
        thisPdf.savefig(figTau)
        # axR.set_title('axR')
        thisPdf.savefig(figR)
        # axRn0Ana.set_title('axRn0Ana')
        thisPdf.savefig(figRn0Ana)
        thisPdf.close()

    # Now outside the loop over different operators

    # So Let's do the inflection point fitting
    # Do it only to the first n0 for now
    XPoints = np.asarray(np.float64(params['analysis']['RXPoints']))
    tS = np.linspace(XPoints.min(), XPoints.max(), 5000)
    n0 = params['analysis']['RCurves'][0]

    colCount = 0
    markCount = 0
    inflectVals = {}
    inflectLabs = {}
    for taa, ana in enumerate(allAnaOps):
        thisAnaLab = params['cfuns']['hadrons'][taa]
        # find the inflection points
        #gvSpline, inflectVal = getInflectionPoint(params, XPoints, RCP[ana][n0], TpcScale * 0.8, tS)
        gvSpline, inflectVal = getInflectionPoint(params, XPoints, RCP[ana][n0], 155, tS)
        if colCount > len(GVP.colours) - 1:
            colCount = 0
        if markCount > len(mo.markers) - 1:
            markCount = 0
        axRn0 = GVP.myFill_between(tS, gvSpline(tS), axRn0, ls='--',colour=GVP.colours[colCount])  # noqa: E501
        if 'figTextYPos' in params['analysis'].keys():
            # Plotting some text at end of lines
            xt = XPoints[-1] + 0.2 * abs(XPoints[0] - XPoints[1])
            yt = params['analysis']['figTextYPos'][taa]
            text = axRn0.text(xt, yt, params['analysis']['figText'][taa])
            # Putting a marker in to the right of the text
            text.draw(figRn0.canvas.get_renderer())
            ex = text.get_window_extent()
            tex = axRn0.transData.inverted().transform_bbox(ex)
            if TpcScale == 1.0:
                xM = 455 / 167
            else:
                xM = 455
            yM = tex.y0 + 0.5 * tex.height
            axRn0.plot(xM, yM, marker=mo.markers[markCount], color=GVP.colours[colCount], linestyle='')
        
        if inflectVal < 0.8 * TpcScale or inflectVal > 1.2 * TpcScale:
            colCount = colCount + 1
            markCount = markCount + 1
            continue
        inflectVals.update({ana: inflectVal})
        inflectLabs.update({ana: thisAnaLab[1:-1]})  # removes the $
        # and plot
        axRn0 = GVP.myVSpan(inflectVal, axRn0, lab='$' + thisAnaLab + f'= {inflectVal}$ MeV', colour=GVP.colours[colCount])  # noqa: E501

        # incrementing counters
        colCount = colCount + 1
        markCount = markCount + 1
    # Finallising plot
    axRn0.set_ylabel('$R(n_0=' + f'{params["analysis"]["RCurves"][0]})$')
    axRn0.set_xlabel('Temperature (MeV)')
    # Removing duplicate labels
    handles, legendLabels = axRn0.get_legend_handles_labels()
    uniqueInd = uniqueIndices(legendLabels)
    uniHand = [handles[i] for i in uniqueInd]
    uniLegLab = [legendLabels[i] for i in uniqueInd]
    # axRn0.legend(uniHand, uniLegLab, loc='best', ncol=2)
    # Saving
    # axRn0.set_title('axRn0')
    pdf.savefig(figRn0)
    plt.close()
    pdf.close()
    # Save the data points we fit the spline to
    for taa, thisAna in enumerate(params['analysis']['labelsToAnalyse']):
        thisAnaOp = '_'.join(thisAna[0].split('_')[:-1])
        thisAnaDir = os.path.join(anaDir, thisAnaOp)
        with open(os.path.join(thisAnaDir, 'RCP.txt'), 'w') as f:
            f.write(gv.tabulate(RCP[thisAnaOp]))
    # Save the inflection points
    with open(os.path.join(anaDir, 'parityInflectionPoints.txt'), 'w') as f:
        f.write(gv.tabulate(gv.BufferDict(inflectVals)))
        f.write('\n')
    # Save them again
    saveFile = os.path.join(anaDir, 'Inflect_M_Sdev.csv')
    with open(saveFile, 'w') as f:
        f.write(f'ana, symb, MeV, MeVErr, TTpc, TTpcErr \n')
        for ana, temp in inflectVals.items():
            tempMeV = temp * TpcScale
            if TpcScale == 1.0:
                tempMeV = temp * TpcScale
                tempTpc = temp
            else:
                tempMeV = temp
                tempTpc = temp / TpcScale
            f.write(f'{ana}, {inflectLabs[f"{ana}"]}, {gv.mean(tempMeV)}, {gv.sdev(tempMeV)}, {gv.mean(tempTpc)}, {gv.sdev(tempTpc)} \n')  # noqa: E501
    sys.exit('Done')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    mpl.rcParams['lines.markersize'] = 10.0
    # For Poster/Presentation
    mpl.rcParams['ytick.labelsize'] = 32
    mpl.rcParams['xtick.labelsize'] = 32
    mpl.rcParams['font.size'] = 36
    main(sys.argv[1:])
