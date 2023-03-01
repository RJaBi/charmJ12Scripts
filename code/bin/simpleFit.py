import toml
import os
import sys
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
import matplotlib as mpl
import gvar as gv  # type: ignore
import lsqfit  # type: ignore
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path
# Loading some functions from elsewhere in this repo
insertPath = os.path.join(Path(__file__).resolve().parent, '..', 'lib')
sys.path.insert(0, insertPath)
import myModules as mo  # type: ignore  # noqa: E402
import myIO as io  # type: ignore  # noqa: E402
import myEffE as EP  # type: ignore  # noqa: E402
import myGVPlotter as GVP  # type: ignore  # noqa: E402
import myFits as FT  # type: ignore  # noqa: E402


def effEPlot(x: np.ndarray, y: np.ndarray, ax, hoz: List[np.ndarray] = [], ma='d', ls='', hozls=None, colour=None, hozColour=None, lab=None, hozLabs=None):  # noqa: E501
    """
    The plot of the effective mass with the lines
    """
    # Plot the effective mass
    ax = GVP.plot_gvEbar(x, y, ax, ma=ma, ls=ls, lab=lab)
    if hozls is None or len(hozls) != len(hoz):
        hozls = [None] * len(hoz)
    if hozColour is None or len(hozColour) != len(hoz):
        hozColour = [None] * len(hoz)
    if hozLabs is None or len(hozLabs) != len(hoz):
        hozLabs = [None] * len(hoz)
    # Plot the horizontal lines
    for hh, hl in enumerate(hoz):
        ax = GVP.myHSpan(hl, ax, ls=hozls[hh], colour=hozColour[hh], lab=hozLabs[hh])
    return ax


def make_prior(nexp, ABase=0.5, EBase=1):
    # make priors for fit parameters
    prior = gv.BufferDict()
    prior['A'] = [gv.gvar(ABase, 0.4) for i in range(nexp)]
    prior['E_P'] = [gv.gvar(i + EBase, 0.4) for i in range(nexp)]
    # prior['log(E_P)'] = [gv.log(gv.gvar((i - 1) + EBase, 0.4)) for i in range(nexp)]
    prior['B'] = [gv.gvar(ABase, 0.4) for i in range(nexp)]
    prior['E_M'] = [gv.gvar(i + EBase, 0.4) for i in range(nexp)]
    # prior['log(E_M)'] = [gv.log(gv.gvar((i - 1) + EBase, 0.4)) for i in range(nexp)]
    return prior


def make_con_prior(nexp, ABase=0.5, EBase=1):
    # make priors for fit parameters
    prior = gv.BufferDict()
    prior['A'] = [gv.gvar(ABase, 0.4) for i in range(nexp)]
    # prior['E_P'] = [gv.gvar(i + EBase, 0.4) for i in range(nexp)]
    prior['log(E_P)'] = [gv.log(gv.gvar(i + EBase, 0.4)) for i in range(nexp)]
    prior['B'] = [gv.gvar(ABase, 0.4) for i in range(nexp)]
    prior['log(Del_EM_EP)'] = [gv.log(gv.gvar(i + EBase, 0.4)) for i in range(nexp)]
    # prior['E_M'] = prior['E_P'] + prior['Del_EM_EP']
    # prior['log(E_M)'] = [gv.log(gv.gvar((i - 1) + EBase, 0.4)) for i in range(nexp)]
    return prior


def allFits(GVD: np.ndarray, fMin: int, fMax: int, Nt: int, fFunc) -> Tuple[List[Any], List[Tuple[int, int, int, int]]]:  # noqa: E501
    """
    Does all the fits
    """
    x = np.arange(0, Nt)
    # Getting fit windows
    fwin = FT.makeFitWins(fMin, fMax, Nt)
    # storage for fit results
    fitRes = []
    fitTuples = []
    # Iterate over fit windows
    for ls, le, rs, re in fwin:
        # print()
        try:
            # if True:
            yData = np.concatenate((GVD[ls:le], GVD[rs:re]))
            xData = np.concatenate((x[ls:le], x[rs:re]))
            # print(f'Doing fit for [{ls}, {le}] and [{rs}, {re}]')
            yData = np.concatenate((GVD[ls:le], GVD[rs:re]))
            xData = np.concatenate((x[ls:le], x[rs:re]))
            # Iterate over number of exponentials
            # p0 = {'A': [0.5], 'B':[0.5], 'E_P':[0.2], 'E_M':[0.2]}
            p0 = None
            count = 0
            # nexpFinal = 0
            # broke = False
            # for nexp in range(1, 6):
            for nexp in range(1, 4):
                # if nexp > 1:
                #    continue
                EBase = 0.2
                yDataDict = {
                    'yData': yData,
                    'E_M': [gv.gvar(i + EBase, 0.4) for i in range(nexp)]
                }
                if count > 0:
                    prevFit: lsqfit.nonlinear_fit = fit  # type: ignore  # noqa: F821
                prior = make_con_prior(nexp, EBase=EBase)
                # fit = lsqfit.nonlinear_fit(data=(xData, yDataDict), fcn=fFunc, prior=prior, p0=p0)
                # print(fit.format(maxline=True))
                # for k, v in fit.p.items():
                #     print(k , v)
                # print(fit.p)
                # print(fit.p['Del_EM_EP'])
                # print(fit.p['E_P']+fit.p['Del_EM_EP'])
                # fit.p.update({'E_M': fit.p['E_P']+fit.p['Del_EM_EP']})
                # sys.exit()
                # if True:
                try:
                    if p0 is None:
                        # Use variance fit to get initial point
                        # For fit
                        yVarDataDict = {
                            'yData': gv.gvar(gv.mean(yData), gv.sdev(yData)),
                            'E_M': [gv.gvar(i + EBase, 0.4) for i in range(nexp)]
                        }
                        p0Var = gv.mean(prior)
                        fitVar = lsqfit.nonlinear_fit(data=(xData, yVarDataDict), fcn=fFunc, prior=prior, p0=p0Var)  # noqa: E501
                        p0 = fitVar.pmean
                    fit = lsqfit.nonlinear_fit(data=(xData, yDataDict), fcn=fFunc, prior=prior, p0=p0)  # noqa: E501
                    fit.p.update({'E_M': fit.p['E_P']+fit.p['Del_EM_EP']})
                    if not np.isfinite(fit.logGBF):
                        continue
                # if True:
                except:  # noqa: E722
                    # nexpFinal = nexp
                    continue
                if fit.chi2 / fit.dof < 1.:
                    p0 = fit.pmean          # starting point for next fit (opt.)
                else:
                    p0 = None
                if count > 0:
                    lGBFDiff = abs(prevFit.logGBF - fit.logGBF)
                    if lGBFDiff <= 1:
                        # print(prevFit.pmean, fit.pmean)
                        fit = prevFit
                        # nexpFinal = nexp - 1
                        # broke = True
                        break
                count = count + 1

            # print(f'nexp = {nexpFinal}, broke = {broke}')
            fitRes.append(fit)
            # append afterwards in case none of the fits worked
            fitTuples.append((ls, le, rs, re))
            # if False:
        except:  # noqa: E722
            print('fit failed')
            continue
        # sys.exit()
    return fitRes, fitTuples


def allConstFits(GVD: np.ndarray, fMin: int, fMax: int, Nt: int) -> Tuple[List[Any], List[Tuple[int, int, int, int]]]:  # noqa: E501
    """
    Does all the fits
    """
    x = np.arange(0, Nt)
    # Getting fit windows
    fwin = FT.makeFitWins(fMin, fMax, Nt)
    # storage for fit results
    fitRes = []
    fitTuples = []
    # Iterate over fit windows
    for ls, le, rs, re in fwin:
        # print()
        print(f'Doing fit for [{ls}, {le}] and [{rs}, {re}]')
        yF, keepF = mo.removeZero(GVD[ls: le])
        yB, keepB = mo.removeZero(GVD[rs: re])
        xF = x[ls: le][keepF]
        xB = x[rs: re][keepB]
        # Packaging the data for lsqfit
        yData = {
            'F': yF,
            'B': yB}
        xData = {
            'F': xF,
            'B': xB}
        # Making the prior
        prior = {
            'c0_F': gv.gvar(0.2, 0.8),
            'c0_B': gv.gvar(0.2, 0.8)}
        fit = lsqfit.nonlinear_fit(data=(xData, yData), fcn=effMassFcn_lsq, prior=prior)  # noqa: E501
        fitRes.append(fit)
        # append afterwards in case none of the fits worked
        fitTuples.append((ls, le, rs, re))
    return fitRes, fitTuples


def fitDictList(fitDict: Dict[Any, Any], fitTuples: List[Tuple[int, int, int, int]]) -> List[Any]:
    """
    Transforms a dictionary of fits to a list
    Order is order in fitTuples
    """
    fitList = []
    # print('keys', fitDict.keys())
    # print('ft', fitTuples)
    # sys.exit()
    for k in fitTuples:
        fitList.append(fitDict[k])
    return fitList


def fitListDict(fitList: List[Any], fitTuples: List[Tuple[int, int, int, int]]) -> Dict[Any, Any]:
    """
    Transforms a list of fits to a dict
    Order is order in fitTuples
    """
    fitDict = {}
    for ii, k in enumerate(fitTuples):
        fitDict.update({k:
                        {'p': fitList[ii].p,
                         'chi2': fitList[ii].chi2,
                         'dof': fitList[ii].dof,
                         'logGBF': fitList[ii].logGBF,
                         'prior': fitList[ii].prior,
                         'Q': fitList[ii].Q}})  # the Q is the p-value
    return fitDict


# effective mass for single sided
def effMassFcn(x, p):
    return p['c0'] * np.ones(x.shape)


# effetive mass for both sides
def effMassFcn_lsq(x, p):
    ans = {}
    for k in ['F', 'B']:
        ans[k] = p[f'c0_{k}'] * np.ones(x[k].shape)
    return ans


def pValueModelAve(Q: np.ndarray, fitParams: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Does the p-value based model averaging of 1901.07519 (FORM FROM 2003.12130)
    """
    aved = []
    statErr = []
    sysErr = []
    weights = []
    for ii, fp in enumerate(fitParams):
        errFit2 = np.float64(gv.sdev(fp))**(-2.0)
        wf = Q * errFit2 / np.sum(Q * errFit2)
        aved.append(np.sum(wf * fp))
        statErr.append(np.sum(wf * gv.sdev(fp)**2.0))
        sysErr.append(np.sum(wf * (gv.mean(fp) - gv.mean(aved[-1]))**2.0))
        # Incoporating the systematic uncertainty due to choice of
        # fit windos in
        cov = gv.evalcov(aved[-1])
        # newCov = cov + statErr[-1]**2 + sysErr[-1]**2
        # They are already std squared
        # newCov = cov + statErr[-1] + sysErr[-1]
        newCov = cov + sysErr[-1]
        aved[-1] = gv.gvar(gv.mean(aved[-1]), newCov)
        weights.append(wf)
    # return the model averaged values
    return np.asarray(aved), np.asarray(weights)


def AICModelAve(fitParams: List[np.ndarray], fPNum: np.ndarray, dof: np.ndarray, Nt: int, chi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # noqa: E501
    """
    Does model averaging according to
    2008.01069, "Bayesian model averaging for analysis of lattice field theory results"
    """
    # i.e. assuming equal probabilty for all models
    pM = np.ones(len(fitParams[0])) * 1.0/float(len(fitParams[0]))  # 1.0/len(fits['data'])
    # number of fit parameters
    k = fPNum
    # the number of data points not fitted to
    nCut = np.ones(len(fitParams[0])) * Nt - dof
    # Calculate the AIC Akaike information criterion (AIC)
    AIC = -2.0 * np.log(pM) + chi2 + 2.0 * k + 2.0 * nCut
    # If the AIC is inf, or non-finite, set it to zero
    expAIC = np.exp(-0.5*AIC)
    expAICIsFinite = np.isfinite(expAIC)
    finiteExpAIC = np.copy(expAIC)
    finiteExpAIC[np.invert(expAICIsFinite)] = 0.0
    # hence calculate probability of model given data
    # prMD = np.exp(-0.5*AIC)/(np.sum(np.exp(-0.5*AIC)))
    prMD = finiteExpAIC/(np.sum(finiteExpAIC))
    # calculate the model averages and
    # systematic errors associated with
    # model seelction
    aved = []
    sysErr1 = []
    sysErr2 = []
    # statErr = []
    for ii, fp in enumerate(fitParams):
        aved.append(np.sum(fp * prMD))
        # statErr.append(np.sum(prMD * gv.sdev(fp)**2.0))
        sysErr1.append(np.sum(prMD * gv.mean(fp)**2.0))
        sysErr2.append(gv.mean(aved[-1])**2.0)
    sysErr1 = np.asarray(sysErr1)  # type: ignore
    sysErr2 = np.asarray(sysErr2)  # type: ignore
    statCov = gv.evalcov(aved)
    # Adding the systematic errors
    newCov = statCov + np.diag(sysErr1) - np.diag(sysErr2)  # + np.diag(statErr)
    aved = gv.gvar(gv.mean(aved), newCov)  # type: ignore
    return aved, prMD  # type: ignore


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
    cfDict, cfLabelsList = io.initCorrelators(params)

    # Now that we've got everything we want.
    # Time to do analysis.

    massdt = np.float64(params['analysis']['massdt'])

    # where we put the analysis
    anaDir = os.path.join(params['analysis']['anaDir'])
    print('Analysis output directory is ', anaDir)
    if not os.path.exists(anaDir):
        os.makedirs(anaDir)
    # The fit window we are considering
    fMin = params['analysis']['fitMin']
    fMax = params['analysis']['fitMax']
    for ana in params['analysis']['labelsToAnalyse']:
        if ana not in cfLabelsList:
            print('label ' + ana + ' is not in cfLabelsList:', cfLabelsList)
            sys.exit('Exiting')
        # Where we will save this bit of analysis
        thisAnaDir = os.path.join(anaDir, ana)
        if not os.path.exists(thisAnaDir):
            os.makedirs(thisAnaDir)
        pdf = PdfPages(os.path.join(thisAnaDir, 'mAve.pdf'))
        # Now do the analysis
        G = io.labelNorm(cfDict['data'][ana], params)
        print(f'Data has shape {G.shape}')
        # Binning the data such that have around 400 bins
        ncon = G.shape[0]
        nana = 400
        if nana > ncon:
            nana = ncon
        binsize = int(np.floor(ncon/nana))
        print(f'Binning data with binsize = {binsize}')
        binned = np.asarray(gv.dataset.bin_data(G, binsize=binsize))
        print(f'Data now has shape {binned.shape}')
        # Take the gv data set averaging and correct cov matrix
        GVD = gv.dataset.avg_data(binned)
        if not params['analysis']['covFit']:
            GVD = gv.gvar(gv.mean(GVD), gv.sdev(GVD))
        print(f'The gvar dataset has shape {GVD.shape}')
        Nt = np.shape(binned)[1]
        x = np.arange(0, Nt)
        # Setting x-axis and actually fMax
        x = range(0, Nt)  # type: ignore
        if fMax > len(x)/2:
            thisfMax = int(np.floor(len(x)/2)) - int(np.ceil(Nt*0.05))
        else:
            thisfMax = fMax
        # make easier to skip first points
        thisfMin = fMin
        effEFMin = thisfMin
        if effEFMin < int(np.ceil(Nt*0.1)):
            effEFMin = int(np.ceil(Nt*0.1))

        # The fit function is here
        # So that it can get Nt implicitly
        # Cause I can't find a way to pass through to lsqfit
        def f(x, p):
            # print(Nt)
            # function used to fit x, y data
            A = p['A']  # array of a[i]s
            E_P = p['E_P']  # array of E[i]s
            B = p['B']  # array of a[i]s
            E_M = p['E_M']  # array of E[i]s
            pos = sum(ai * np.exp(-Ei * x) for ai, Ei in zip(A, E_P))
            neg = sum(ai * np.exp(-Ei * (Nt - x)) for ai, Ei in zip(B, E_M))
            return pos + neg

        # has E_M > E_P
        def fCon(x, p):
            # A constrained fit function
            # which has EM >= EP
            # print(Nt)
            # function used to fit x, y data
            A = p['A']  # array of a[i]s
            E_P = p['E_P']  # array of E[i]s
            B = p['B']  # array of a[i]s
            Del_EM_EP = p['Del_EM_EP']  # array of E[i]s
            E_M = E_P + Del_EM_EP
            pos = sum(ai * np.exp(-Ei * x) for ai, Ei in zip(A, E_P))
            neg = sum(ai * np.exp(-Ei * (Nt - x)) for ai, Ei in zip(B, E_M))
            ans = {'yData': pos + neg,
                   'E_M': E_M}
            return ans
        # Have now defined the fit function
        doFits = params['analysis']['redoFits']
        fitPath = os.path.join(thisAnaDir, 'fit')
        # and also if the file doesn't exist
        print(f'fitPath = {fitPath}')
        if not os.path.exists(fitPath+'.1.gvar') or doFits or not os.path.exists(fitPath+'Tuples.pickle'):  # noqa: E501
            fitRes, fitTuples = allFits(GVD, thisfMin, thisfMax, Nt, fCon)
            print(f'Saving fits to {fitPath}.N.gvar')
            cut = 50
            count = 0
            fcount = 0
            for ii in range(0, len(fitRes), cut):
                if ii >= len(fitRes):
                    end = len(fitRes)+1
                else:
                    end = ii + cut
                # print(ii, count, len(fitRes), end)
                # transform into dict before saving
                fitDict = fitListDict(fitRes[count:end], fitTuples[count:end])
                gv.dump(fitDict, f'{fitPath}.{fcount}.gvar', add_dependencies=True)
                fcount = fcount + 1
                # incremneting counters
                count = count + cut
            print(f'Saving fitwindows to {fitPath}Tuples.pickle')
            pickle.dump(fitTuples, open(f'{fitPath}Tuples.pickle', 'wb'))
        if True:
            # ALWAYS RELOAD FROM THE FIT FILES TO GET RIGHT STRUCTURE
            print(f'from fitwindows from {fitPath}Tuples.pickle')
            fitTuples = pickle.load(open(f'{fitPath}Tuples.pickle', 'rb'))
            print(f'Loading fits from {fitPath}.N.gvar')
            cut = 50
            count = 0
            fcount = 0
            fitRes = []
            for ii in range(0, len(fitTuples), cut):
                fitDict = gv.load(f'{fitPath}.{fcount}.gvar')
                # Transform back into list after loading
                if ii >= len(fitTuples):
                    end = len(fitRes) + 1
                else:
                    end = ii + cut
                fitList = fitDictList(fitDict, fitTuples[count:end])
                fitRes.extend(fitList)
                fcount = fcount + 1
                count = count + cut
        print('After fits')
        # Calculate effective mass
        effE = EP.getEffE(params, massdt, GVD)
        # Remove nan like this cause its a gvar
        for ii, ee in enumerate(effE):
            if np.isnan(gv.mean(ee)) or np.isnan(gv.sdev(ee)):
                # print('is nan', ii)
                effE[ii] = gv.gvar(0, 0)
        # fit effective mass
        print(f'effEFMin = {effEFMin}, thisfMax = {thisfMax}')
        eFitRes, eFitTuples = allConstFits(effE, effEFMin, thisfMax, Nt)
        print(len(eFitRes), len(eFitTuples))
        # Grab the bits we care about
        Q = []
        eEPF = []
        eEMF = []
        wins = []
        chi2 = []
        dof = []
        fParams = []
        for ii, fit in enumerate(eFitRes):
            # Know that tehse are the prior values
            if fit.p['c0_F'] != gv.gvar(0.2, 0.8):
                if fit.p['c0_B'] != gv.gvar(0.2, 0.8):
                    Q.append(fit.Q)
                    eEPF.append(fit.p['c0_F'])
                    eEMF.append(fit.p['c0_B'])
                    wins.append('-'.join(map(str, np.add(eFitTuples[ii], 0))))
                    # The 2 is the middle one
                    wins[-1] = mo.replace_nth(wins[-1], '-', ':', 2)
                    chi2.append(fit.chi2)
                    dof.append(fit.dof)
                    fParams.append(len(fit.p))
        Q = np.asarray(Q)  # type: ignore
        eEPF = np.asarray(eEPF)  # type: ignore
        eEMF = np.asarray(eEMF)  # type: ignore
        chi2 = np.asarray(chi2)  # type: ignore
        dof = np.asarray(dof)  # type: ignore
        fParams = np.asarray(fParams)  # type: ignore
        # Take model averages
        AIC_plat, prMD_plat = AICModelAve([np.asarray(eEPF), np.asarray(eEMF)], np.asarray(fParams), np.asarray(dof), Nt, np.asarray(chi2))  # noqa: E501
        print('AIC_plat', AIC_plat)
        # Save these
        pickle.dump(prMD_plat, open(f'{fitPath}_AIC_plat_prMD.pickle', 'wb'))
        gv.dump(AIC_plat, f'{fitPath}_AIC_plat_Aved.gvar', add_dependencies=True)
        pValAved_plat, wf = pValueModelAve(Q, [eEPF, eEMF])  # type: ignore
        print('pValAved_plat', pValAved_plat)
        pickle.dump(wf, open(f'{fitPath}_pVal_plat_wf.pickle', 'wb'))
        gv.dump(pValAved_plat, f'{fitPath}_pVal_plat_Aved.gvar', add_dependencies=True)
        # Just plot the effective mass fits
        # Pos parity E+
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1.errorbar(wins, y=gv.mean(eEPF), yerr=gv.sdev(eEPF), marker='d', linestyle='')
        # startTicks = list(set(startTicks))
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1 = FT.plotWinLines(Sax1, wins)
        # Sax1.set_xlim([wins[0], None])
        Sax1.set_xlim([-1, len(wins)+1])
        # Sax1.xaxis.set_ticklabels([])
        Sax1.set_ylabel('$a_\\tau\,E_{plat}^{+}$', fontsize=36)  # noqa: W605
        # Sax1.legend(loc='best', ncol=2)
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        pdf.savefig(Sfig)
        plt.close(Sfig)
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1.errorbar(wins, y=gv.mean(eEMF), yerr=gv.sdev(eEMF), marker='d', linestyle='')
        Sax1 = FT.plotWinLines(Sax1, wins)
        Sax1.set_xlim([-1, len(wins)+1])
        # Sax1.set_xlim([wins[0], None])
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1.set_ylabel('$a_\\tau\,E_{plat}^{-}$', fontsize=36)  # noqa: W605
        # Sax1.legend(loc='best', ncol=2)
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        pdf.savefig(Sfig)
        plt.close(Sfig)
        # Neg parity E-
        # Plot the effective mass model averaging
        # # P-Val E+
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1.errorbar(wins, y=gv.mean(eEPF), yerr=gv.sdev(eEPF), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved_plat[0], ax1, ls='--', colour='olive', lab='$E^{+}_{PVal-plat} = ' + f'{pValAved_plat[0]}$')  # noqa: E501
        ax1 = FT.plotWinLines(ax1, wins)
        # finalising
        ax1.set_xlim([-1, len(wins)+1])
        # ax1.set_xlim([wins[0], None])
        ax2.plot(wins, wf[0, :], marker='+', linestyle='')
        # ax2.set_xticklabels(wins, rotation=45, ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylim(params['analysis']['EFitLim'])
        ax1.set_ylabel('$a_\\tau\,E_{plat}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('$\\tau_{min}$', fontsize=36)
        ax2.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        # finalising the second plot
        # Plot the effective mass model averaging
        # # P-Val E-
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1.errorbar(wins, y=gv.mean(eEMF), yerr=gv.sdev(eEMF), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved_plat[1], ax1, ls=':', colour='brown', lab='$E^{-}_{PVal-plat} = ' + f'{pValAved_plat[1]}$')  # noqa: E501
        ax1 = FT.plotWinLines(ax1, wins)
        # finalising
        # ax1.set_xlim([wins[1], None])
        ax1.set_xlim([-1, len(wins)+1])
        ax2.plot(wins, wf[1, :], marker='+', linestyle='')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylabel('$a_\\tau\,E_{plat}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('$\\tau_{min}$', fontsize=36)
        ax2.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        # Plot the effective mass model averaging
        # # AIC-Val E+
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1.errorbar(wins, y=gv.mean(eEPF), yerr=gv.sdev(eEPF), marker='d', linestyle='')
        ax1 = GVP.myHSpan(AIC_plat[0], ax1, ls='--', colour='pink', lab='$E^{+}_{AIC-plat} = ' + f'{AIC_plat[0]}$')  # noqa: E501
        ax1 = FT.plotWinLines(ax1, wins)
        # finalising
        # ax1.set_xlim([wins[0], None])
        ax1.set_xlim([-1, len(wins)+1])
        ax2.plot(wins, prMD_plat, marker='+', linestyle='')  # type: ignore  # noqa: F821
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylim(params['analysis']['EFitLim'])
        ax1.set_ylabel('$a_\\tau\,E_{plat}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        pdf.savefig(fig)
        plt.close(fig)
        # Plot the effective mass model averaging
        # # P-Val E-
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1.errorbar(wins, y=gv.mean(eEMF), yerr=gv.sdev(eEMF), marker='d', linestyle='')
        ax1 = GVP.myHSpan(AIC_plat[1], ax1, ls=':', colour='cyan', lab='$E^{-}_{AIC-plat} = ' + f'{AIC_plat[1]}$')  # noqa: E501
        ax1 = FT.plotWinLines(ax1, wins)
        # finalising
        # ax1.set_xlim([wins[1], None])
        ax1.set_xlim([-1, len(wins)+1])
        ax2.plot(wins, prMD_plat, marker='+', linestyle='')  # type: ignore  # noqa: F821
        # ax2.set_xticklabels(wins, rotation=45, ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylim(params['analysis']['EFitLim'])
        ax1.set_ylabel('$a_\\tau\,E_{plat}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        pdf.savefig(fig)
        plt.close(fig)
        # # Now do the correlator fits
        # Grabbing data from fits
        fEP = []
        fEM = []
        fDE = []
        chi2 = []
        Q = []
        dof = []
        fParams = []
        # Sorting the x-axis out
        wins = []
        NDevP = 5
        NDevM = 5
        # Setting the 'cuts' on the fits
        # Using the pVal Plateau fit
        eEP = pValAved_plat[0]
        eEM = pValAved_plat[1]
        if gv.sdev(eEP) > 0.1 * gv.mean(eEP):
            eEUP = gv.sdev(eEP)
        else:
            eEUP = gv.mean(eEP)
        if gv.sdev(eEM) > 0.1 * gv.mean(eEM):
            eEUM = gv.sdev(eEM)
        else:
            eEUM = gv.mean(eEM)
        for ii, fit in enumerate(fitRes):
            # Here keeping only fits which are somewhat reasonable
            iiEP = fit['p']['E_P'][0]
            setEP = False
            # This loops doesnt do fits if their uncertainty larger
            # than value
            # passes onto next 'exponential'
            # for jjEP in fit['p']['E_P']:
            #    if gv.sdev(jjEP) > abs(gv.mean(jjEP)):
            #        continue
            #    if gv.mean(jjEP) < 0.3:
            #        continue
            #    if not setEP:
            #        iiEP = jjEP
            #        setEP = True
            # if fitTuples[ii][0] == 8 and fitTuples[ii][1] == 13:
            #    print(iiEP)
            #    print(setEP)
            #    print(fit['p']['E_P'])
            #    sys.exit()
            if not setEP:
                # if fitTuples[ii][0] == 4 and fitTuples[ii][1] == 13:
                #    print('looking at 4-13')
                # sys.exit()
                # iiA = fit['p']['A'][0]
                for jj, jjA in enumerate(fit['p']['A']):
                    if jj >= 2:
                        continue
                    if gv.sdev(fit['p']['E_P'][jj]) > gv.mean(fit['p']['E_P'][jj]):
                        continue
                    if gv.sdev(fit['p']['A'][jj]) > gv.mean(fit['p']['A'][jj]):
                        continue
                    if gv.mean(jjA) <= 0.55 and gv.mean(jjA) >= 0.45:
                        if gv.sdev(jjA) <= 0.41 and gv.sdev(jjA) >= 0.39:
                            print('skipping', jj, jjA)
                            continue
                        # if jjA == gv.gvar('0.50(40)'):
                        # This was the prior value
                        # continue
                    # if jjA > iiA * 0.8:
                    if not setEP:
                        if jjA > 0.1:
                            # iiA = jjA
                            iiEP = fit['p']['E_P'][jj]
                            setEP = True
            # if fitTuples[ii][0] == 36 and fitTuples[ii][1] == 45:
            # if iiEP > 0.8:
            #     if fitTuples[ii][0] <=10:
            #         continue
            #     print(fitTuples[ii])
            #     print(iiEP)
            #     print(setEP)
            #     print(fit['p']['E_P'])
            #     print('E    ', fit['p']['E_P'], fit['p']['E_M'])
            #     print('AB   ', fit['p']['A'], fit['p']['B'])
            #     print(gv.sdev(fit['p']['A'][2]), gv.sdev(fit['p']['A'][2]) == 0.40)
            #     print('    ', iiEP)
            # sys.exit()
            iiEM = fit['p']['E_M'][0]
            if gv.mean(iiEP) <= gv.mean(eEP) + eEUP * NDevP and gv.mean(iiEP) >= gv.mean(eEP) - eEUP * NDevP:  # type: ignore  # noqa: E501
                if gv.mean(iiEM) <= gv.mean(eEM) + eEUM * NDevM and gv.mean(iiEM) >= gv.mean(eEM) - eEUM * NDevM:  # type: ignore  # noqa: E501
                    # if gv.mean(iiEP) > 1.0 or gv.mean(iiEM) > 1.0:
                    #    continue
                    fEP.append(iiEP)  # type: ignore
                    fEM.append(iiEM)  # type: ignore
                    fDE.append(fit['p']['Del_EM_EP'][0])  # type: ignore
                    chi2.append(fit['chi2'])  # type: ignore
                    Q.append(fit['Q'])  # type: ignore
                    dof.append(fit['dof'])  # type: ignore
                    fParams.append(len(fit['p']))  # type: ignore
                    """
                    if fParams[-1] > 4:
                        print(fitTuples[ii])
                        print('E    ', fit['p']['E_P'], fit['p']['E_M'])
                        print('AB   ', fit['p']['A'], fit['p']['B'])
                        print('    ', iiEP, iiEM)
                    """
                    wins.append('-'.join(map(str, np.add(fitTuples[ii], 0))))
                    # The 2 is the middle one
                    wins[-1] = mo.replace_nth(wins[-1], '-', ':', 2)
        # convert into numpy arrays
        fEP = np.asarray(fEP)  # type: ignore
        fEM = np.asarray(fEM)  # type: ignore
        fDE = np.asarray(fDE)  # type: ignore
        chi2 = np.asarray(chi2)  # type: ignore
        Q = np.asarray(Q)  # type: ignore
        dof = np.asarray(dof)  # type: ignore
        fParams = np.asarray(fParams)  # type: ignore
        # OPrinting (temp)
        # for ii in range(0, len(fDE)):
        #    print(fEP[ii], fDE[ii], fEM[ii], fDE[ii] + fEP[ii])
        print(f'Averaging {len(fEP)} fits')
        # model averaging sing p value method
        pValAved, wf = pValueModelAve(Q, [fEP, fEM])  # type: ignore
        print(pValAved, wf.shape)
        # and similarly for the AIC method
        AICValAved, prMD = AICModelAve([fEP, fEM], fParams, dof, Nt, chi2)  # type: ignore
        print(AICValAved, prMD.shape)
        # Now save the model averaged values
        pickle.dump(wf, open(f'{fitPath}_pVal_wf.pickle', 'wb'))
        gv.dump(pValAved, f'{fitPath}_pVal_Aved.gvar', add_dependencies=True)
        # now the AIC
        pickle.dump(prMD, open(f'{fitPath}_AIC_prMD.pickle', 'wb'))
        gv.dump(AICValAved, f'{fitPath}_AIC_Aved.gvar', add_dependencies=True)
        # Now do some plots
        # Just the fit values. No model averages
        # E+
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEP), Sax1, 'd', '')
        # Sax1.set_xlim([wins[0], None])
        Sax1.set_xlim([-1, len(wins)+1])
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        # Sax1.legend(loc='best', ncol=2)
        pdf.savefig(Sfig)
        plt.close(Sfig)
        # E-
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEM), Sax1, 'd', '')
        # Sax1.set_xlim([wins[0], None])
        Sax1.set_xlim([-1, len(wins)+1])
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        # Sax1.legend(loc='best', ncol=2)
        pdf.savefig(Sfig)
        plt.close(Sfig)
        # First do the model averaged plots with all fit windows
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        fig_t, (ax1_t, ax2_t) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEP), ax1, 'd', '')
        ax1_t = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEM), ax1_t, 'd', '')
        ax1 = GVP.myHSpan(pValAved[0], ax1, colour='tab:red', ls='--', lab='$E^{+}_{pval} = ' + f'{pValAved[0]}$')  # noqa: E501
        ax1_t = GVP.myHSpan(pValAved[1], ax1_t, ls=':', colour='tab:green', lab='$E^{-}_{pval} = ' + f'{pValAved[1]}$')  # noqa: E501
        ax1 = GVP.myHSpan(AICValAved[0], ax1, ls='--', colour='tab:orange', lab='$E^{+}_{AIC} = ' + f'{AICValAved[0]}$')  # noqa: E501
        ax1_t = GVP.myHSpan(AICValAved[1], ax1_t, ls=':', colour='tab:purple', lab='$E^{-}_{AIC} = ' + f'{AICValAved[1]}$')  # noqa: E501
        # Putting a vertical line where fit window changes
        ax1 = FT.plotWinLines(ax1, wins)
        ax1_t = FT.plotWinLines(ax1_t, wins)
        # finalising
        # ax1.set_xlim([wins[0], None])
        ax1.set_xlim([-1, len(wins)+1])
        # ax1_t.set_xlim([wins[0], None])
        ax1_t.set_xlim([-1, len(wins)+1])
        ax2.plot(wins, prMD, marker='d', linestyle='')
        ax2.plot(wins, wf[0, :], marker='+', linestyle='')
        # ax2.set_xticklabels(wins, rotation=45, ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax2_t.plot(wins, prMD, marker='d', linestyle='')
        ax2_t.plot(wins, wf[1, :], marker='+', linestyle='')
        # ax2_t.set_xticklabels(wins, rotation=45, ha='right')
        ax1_t.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2_t.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1_t.set_xticklabels(wins, rotation=90)
        for tickLab in ax1_t.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2_t.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylim(params['analysis']['EFitLim'])
        ax1_t.set_ylim(params['analysis']['EFitLim'])
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        ax1_t.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2_t.set_xlabel('fit window')
        ax2_t.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        ax1_t.legend(loc='best', ncol=2)
        pdf.savefig(fig)
        plt.close(fig)
        pdf.savefig(fig_t)
        plt.close(fig_t)
        print(f'averaged len(wins), {len(wins)} fits')
        # eff mass plot
        fig, ax1 = plt.subplots(figsize=(16.6, 11.6))
        hoz = [pValAved[0], pValAved[1], AICValAved[0], AICValAved[1], eEP, eEM, AIC_plat[0], AIC_plat[1]]  # noqa: E501
        hozls = ['--', ':', '--', ':', '--', ':', '--', ':']
        hozLabs = ['$E^{+}_{pval} = ' + f'{pValAved[0]}$',
                   '$E^{-}_{pval} = ' + f'{pValAved[1]}$',
                   '$E^{+}_{AIC} = ' + f'{AICValAved[0]}$',
                   '$E^{-}_{AIC} = ' + f'{AICValAved[1]}$',
                   '$E^{+}_{PVal-plat} = ' + f'{eEP}$',
                   '$E^{-}_{PVal-plat} = ' + f'{eEM}$',
                   '$E^{+}_{AIC-plat} = ' + f'{AIC_plat[0]}$',
                   '$E^{-}_{AIC-plat} = ' + f'{AIC_plat[1]}$']
        hozCols = ['tab:red', 'tab:green',
                   'tab:orange', 'tab:purple',
                   'tab:olive', 'tab:brown',
                   'tab:pink', 'tab:cyan']
        ax1 = effEPlot(x, effE, ax1, hoz, ma='d', lab=None, ls='', hozColour=hozCols, hozLabs=hozLabs, hozls=hozls)  # noqa: E501
        # Finalising
        ax1.legend(loc='best', ncol=2)
        ax1.set_ylabel('$a_\\tau\,E^{' + f'{ana.replace("_", ".").replace("x", "X")}' + '}(\\tau)$', fontsize=36)  # noqa: W605, E501
        ax1.set_xlabel('$\\tau / a_\\tau$', fontsize=36)  # noqa: W605
        ax1.set_ylim(params['analysis']['effEyLim'])
        # effEYLim = ax1.get_ylim()
        pdf.savefig(fig)
        plt.close(fig)

        # First do the model averaged plots with top 50 fit windows
        # AIC method first
        nFits = 50
        # Sort according to the most probable windows
        sortOrder = np.flip(prMD.argsort())
        sortPrMD = prMD[sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEP = fEP[sortOrder][0: nFits]
        sortEM = fEM[sortOrder][0: nFits]
        fig, (ax1, ax3, ax2) = plt.subplots(3, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [2, 2, 0.5]})  # noqa: E501
        # Plot the fit values
        ax1.errorbar(sortWins, y=gv.mean(sortEP), yerr=gv.sdev(sortEP), marker='d', linestyle='')
        ax3.errorbar(sortWins, y=gv.mean(sortEM), yerr=gv.sdev(sortEM), marker='d', linestyle='')
        # Plot the resulting fits
        ax1 = GVP.myHSpan(AICValAved[0], ax1, ls='--', colour='tab:orange', lab='$E^{+}_{AIC} = ' + f'{AICValAved[0]}$')  # noqa: E501
        ax3 = GVP.myHSpan(AICValAved[1], ax3, ls=':', colour='tab:purple',  lab='$E^{-}_{AIC} = ' + f'{AICValAved[1]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0], None])
        ax1.set_xlim([-1, len(sortWins) + 1])
        ax2.plot(sortWins, sortPrMD, marker='d', linestyle='')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        # ax1.set_ylim(params['analysis']['EFitLim'])
        ax1.legend(loc='best', ncol=2)
        ax3.legend(loc='best', ncol=2)
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax3.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        print(f'averaged len(wins), {len(wins)} fits')
        # Now the two p-valued
        # These may have different weights
        # E+ PVal
        sortOrder = np.flip(wf[0, :].argsort())
        sortWfP = wf[0, sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEP = fEP[sortOrder][0: nFits]
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1.errorbar(sortWins, y=gv.mean(sortEP), yerr=gv.sdev(sortEP), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved[0], ax1, colour='tab:red', ls='--',  lab='$E^{+}_{pval} = ' + f'{pValAved[0]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0], None])
        ax1.set_xlim([-1, len(sortWins) + 1])
        ax1.legend(loc='best', ncol=2)
        ax2.plot(sortWins, sortWfP, marker='+', linestyle='')
        # ax2.set_xticklabels(sortWins, rotation=45, ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        # E+ PVal
        sortOrder = np.flip(wf[1, :].argsort())
        sortWfP = wf[1, sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEM = fEM[sortOrder][0: nFits]
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})  # noqa: E501
        ax1.errorbar(sortWins, y=gv.mean(sortEM), yerr=gv.sdev(sortEM), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved[1], ax1, colour='tab:green', ls=':',  lab='$E^{-}_{pval} = ' + f'{pValAved[1]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0], None])
        ax1.set_xlim([-1, len(sortWins) + 1])
        ax1.legend(loc='best', ncol=2)
        ax2.plot(sortWins, sortWfP, marker='+', linestyle='')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        #########################################################################################
        # average only the most probable say 30% of fits
        # Based on AIC average
        # fitPer = 0.01
        fitPer = 0.3
        nFits = int(np.ceil(fitPer * len(prMD)))
        if nFits < 10:
            nFits = 10
        print(f'nFits is {nFits}')
        sortOrder = np.flip(wf[0, :].argsort())
        sortWins = np.asarray(wins)[sortOrder][5: nFits+5]
        # Grabbing data from fits
        fEP = []
        fEM = []
        chi2 = []
        Q = []
        dof = []
        fParams = []
        # Sorting the x-axis out
        wins = []
        for ii, fit in enumerate(fitRes):
            thisWin = '-'.join(map(str, np.add(fitTuples[ii], 0)))
            # The 2 is the middle one
            thisWin = mo.replace_nth(thisWin, '-', ':', 2)
            # This is to match earlier
            iiEP = fit['p']['E_P'][0]
            setEP = False
            # This loops doesnt do fits if their uncertainty larger
            # than value
            # passes onto next 'exponential'
            # for jjEP in fit['p']['E_P']:
            #    if gv.sdev(jjEP) > abs(gv.mean(jjEP)):
            #        continue
            #    if gv.mean(jjEP) < 0.3:
            #        continue
            #    if not setEP:
            #        iiEP = jjEP
            #        setEP = True
            # if fitTuples[ii][0] == 8 and fitTuples[ii][1] == 13:
            #    print(iiEP)
            #    print(setEP)
            #    print(fit['p']['E_P'])
            #    sys.exit()
            if not setEP:
                # if fitTuples[ii][0] == 4 and fitTuples[ii][1] == 13:
                #    print('looking at 4-13')
                # sys.exit()
                # iiA = fit['p']['A'][0]
                for jj, jjA in enumerate(fit['p']['A']):
                    # print(jj)
                    if jj >= 2:
                        continue
                    if gv.sdev(fit['p']['E_P'][jj]) > gv.mean(fit['p']['E_P'][jj]):
                        continue
                    if gv.sdev(fit['p']['A'][jj]) > gv.mean(fit['p']['A'][jj]):
                        continue
                    if gv.mean(jjA) <= 0.55 and gv.mean(jjA) >= 0.45:
                        if gv.sdev(jjA) <= 0.41 and gv.sdev(jjA) >= 0.39:
                            # print('skipping', jj, jjA)
                            continue
                        # if jjA == gv.gvar('0.50(40)'):
                        #  This was the prior value
                        # continue
                    # if jjA > iiA * 0.8:
                    if not setEP:
                        if jjA > 0.1:
                            # iiA = jjA
                            iiEP = fit['p']['E_P'][jj]
                            setEP = True

            iiEM = fit['p']['E_M'][0]
            if thisWin in sortWins:
                # print(thisWin, iiEP)
                fEP.append(iiEP)  # type: ignore
                fEM.append(iiEM)  # type: ignore
                chi2.append(fit['chi2'])  # type: ignore
                Q.append(fit['Q'])  # type: ignore
                dof.append(fit['dof'])  # type: ignore
                fParams.append(len(fit['p']))  # type: ignore
                wins.append(thisWin)
        # print(wins)
        # convert into numpy arrays
        fEP = np.asarray(fEP)  # type: ignore
        fEM = np.asarray(fEM)  # type: ignore
        chi2 = np.asarray(chi2)  # type: ignore
        Q = np.asarray(Q)  # type: ignore
        dof = np.asarray(dof)  # type: ignore
        fParams = np.asarray(fParams)  # type: ignore
        print(f'Averaging {len(fEP)} fits')
        # model averaging sing p value method
        pValAved, wf = pValueModelAve(Q, [fEP, fEM])  # type: ignore
        print('after aic', pValAved, wf.shape)
        # and similarly for the AIC method
        AICValAved, prMD = AICModelAve([fEP, fEM], fParams, dof, Nt, chi2)  # type: ignore
        print('after aic', AICValAved, prMD.shape)
        # Save these fits
        pickle.dump(wf, open(f'{fitPath}_AICP_pVal_wf.pickle', 'wb'))
        gv.dump(pValAved, f'{fitPath}_AICP_pVal_Aved.gvar', add_dependencies=True)
        # now the AIC
        pickle.dump(prMD, open(f'{fitPath}_AICP_AIC_prMD.pickle', 'wb'))
        gv.dump(AICValAved, f'{fitPath}_AICP_AIC_Aved.gvar', add_dependencies=True)

        # Now do some plots
        # E +
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEP), Sax1, 'd', '')
        # Sax1.set_xlim([wins[0], None])
        Sax1.set_xlim([-1, len(wins) + 1])
        Sax1 = FT.plotWinLines(Sax1, wins)
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1.set_ylabel('$a_\\tau\,E_{Rfit}^{+}$', fontsize=36)  # noqa: W605
        Sax1 = GVP.myHSpan(pValAved[0], Sax1, colour='tab:red', ls='--', lab='$E^{+}_{pval} = ' + f'{pValAved[0]}$')  # noqa: E501
        Sax1 = GVP.myHSpan(AICValAved[0], Sax1, ls='--', colour='tab:orange', lab='$E^{+}_{AIC} = ' + f'{AICValAved[0]}$')  # noqa: E501
        Sax1.set_ylim(params['analysis']['EFitLim'])
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        Sax1.legend(loc='best', ncol=2)
        pdf.savefig(Sfig)
        plt.close(Sfig)
        # E-
        Sfig, Sax1 = plt.subplots(figsize=(16.6, 11.6))
        Sax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEM), Sax1, 'd', '')
        Sax1 = FT.plotWinLines(Sax1, wins)
        # Sax1.set_xlim([wins[0], None])
        Sax1.set_xlim([-1, len(wins)+1])
        Sax1.xaxis.set_ticklabels(FT.getWinTicks(Sax1, wins), fontsize=16)
        Sax1 = GVP.myHSpan(pValAved[1], Sax1, ls=':', colour='tab:green', lab='$E^{-}_{pval} = ' + f'{pValAved[1]}$')  # noqa: E501
        Sax1 = GVP.myHSpan(AICValAved[1], Sax1, ls=':', colour='tab:purple', lab='$E^{-}_{AIC} = ' + f'{AICValAved[1]}$')  # noqa: E501
        Sax1.set_ylabel('$a_\\tau\,E_{Rfit}^{-}$', fontsize=36)  # noqa: W605
        Sax1.set_xlabel('$\\tau_{min}$', fontsize=36)
        Sax1.legend(loc='best', ncol=2)
        # Sax1.set_title('test')
        pdf.savefig(Sfig)
        plt.close(Sfig)
        # First do the model averaged plots with all fit windows
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        fig_t, (ax1_t, ax2_t) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})  # noqa: E501
        ax1 = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEP), ax1, 'd', '')
        ax1_t = GVP.plot_gvEbar(np.asarray(wins), np.asarray(fEM), ax1_t, 'd', '')
        ax1 = GVP.myHSpan(pValAved[0], ax1, colour='tab:red', ls='--', lab='$E^{+}_{pval} = ' + f'{pValAved[0]}$')  # noqa: E501
        ax1_t = GVP.myHSpan(pValAved[1], ax1_t, ls=':', colour='tab:green', lab='$E^{-}_{pval} = ' + f'{pValAved[1]}$')  # noqa: E501
        ax1 = GVP.myHSpan(AICValAved[0], ax1, ls='--', colour='tab:orange', lab='$E^{+}_{AIC} = ' + f'{AICValAved[0]}$')  # noqa: E501
        ax1_t = GVP.myHSpan(AICValAved[1], ax1_t, ls=':', colour='tab:purple', lab='$E^{-}_{AIC} = ' + f'{AICValAved[1]}$')  # noqa: E501
        # Putting a vertical line where fit window changes
        ax1 = FT.plotWinLines(ax1, wins)
        ax1_t = FT.plotWinLines(ax1_t, wins)
        # finalising
        # ax1.set_xlim([wins[0], None])
        ax1.set_xlim([-1, len(wins)+1])
        # ax1_t.set_xlim([wins[0], None])
        ax1_t.set_xlim([-1, len(wins)+1])
        ax2.plot(wins, prMD, marker='d', linestyle='')
        ax2.plot(wins, wf[0, :], marker='+', linestyle='')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax2.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(wins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax2_t.plot(wins, prMD, marker='d', linestyle='')
        ax2_t.plot(wins, wf[1, :], marker='+', linestyle='')
        ax2_t.set_xticklabels(wins, rotation=90)  # , ha='right')
        ax1_t.set_xticklabels(wins, rotation=90)
        for tickLab in ax1_t.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2_t.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        # ax1.set_ylim(params['analysis']['EFitLim'])
        # ax1_t.set_ylim(params['analysis']['EFitLim'])
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        ax1_t.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2_t.set_xlabel('fit window')
        ax2_t.set_ylabel('weight')
        ax1.legend(loc='best', ncol=2)
        ax1_t.legend(loc='best', ncol=2)
        pdf.savefig(fig)
        plt.close(fig)
        pdf.savefig(fig_t)
        plt.close(fig_t)
        print(f'averaged len(wins), {len(wins)} fits')
        # eff mass plot
        fig, ax1 = plt.subplots(figsize=(16.6, 11.6))
        hoz = [pValAved[0], pValAved[1], AICValAved[0], AICValAved[1]]  # , eEP, eEM, AIC_plat[0], AIC_plat[1]]  # noqa: E501
        hozls = ['--', ':', '--', ':']  # , '--', ':', '--', ':']
        hozLabs = ['$E^{+}_{pval} = ' + f'{pValAved[0]}$',
                   '$E^{-}_{pval} = ' + f'{pValAved[1]}$',
                   '$E^{+}_{AIC} = ' + f'{AICValAved[0]}$',
                   '$E^{-}_{AIC} = ' + f'{AICValAved[1]}$',
                   # '$E^{+}_{PVal-plat} = ' + f'{eEP}$',
                   # '$E^{-}_{PVal-plat} = ' + f'{eEM}$',
                   # '$E^{+}_{AIC-plat} = ' + f'{AIC_plat[0]}$',
                   # '$E^{-}_{AIC-plat} = ' + f'{AIC_plat[1]}$'
        ]  # noqa: E124
        hozCols = ['tab:red', 'tab:green',
                   'tab:orange', 'tab:purple',
                   # 'tab:olive', 'tab:brown',
                   # 'tab:pink', 'tab:cyan'
        ]  # noqa: E124
        ax1 = effEPlot(x, effE, ax1, hoz, ma='d', lab=None, ls='', hozColour=hozCols, hozLabs=hozLabs, hozls=hozls)  # noqa: E501
        # Finalising
        ax1.legend(loc='best', ncol=2, fontsize=30)
        #ax1.set_ylabel('$a_\\tau\,E^{' + f'{ana.replace("_", ".").replace("x", "X")}' + '}(\\tau)$', fontsize=36)  # noqa: W605, E501
        ax1.set_ylabel('$a\,m_{eff}(\\tau)$')
        ax1.set_xlabel('$\\tau / a_\\tau$', fontsize=36)  # noqa: W605
        ax1.set_ylim(params['analysis']['effEyLim'])
        # effEYLim = ax1.get_ylim()
        pdf.savefig(fig)
        plt.close(fig)

        # First do the model averaged plots with top 50 fit windows
        # AIC method first
        nFits = 50
        # Sort according to the most probable windows
        sortOrder = np.flip(prMD.argsort())
        sortPrMD = prMD[sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEP = fEP[sortOrder][0: nFits]
        sortEM = fEM[sortOrder][0: nFits]
        fig, (ax1, ax3, ax2) = plt.subplots(3, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [2, 2, 0.5]})  # noqa: E501
        # Plot the fit values
        ax1.errorbar(sortWins, y=gv.mean(sortEP), yerr=gv.sdev(sortEP), marker='d', linestyle='')
        ax3.errorbar(sortWins, y=gv.mean(sortEM), yerr=gv.sdev(sortEM), marker='d', linestyle='')
        # Plot the resulting fits
        ax1 = GVP.myHSpan(AICValAved[0], ax1, ls='--', colour='tab:orange', lab='$E^{+}_{AIC} = ' + f'{AICValAved[0]}$')  # noqa: E501
        ax3 = GVP.myHSpan(AICValAved[1], ax3, ls=':', colour='tab:purple',  lab='$E^{-}_{AIC} = ' + f'{AICValAved[1]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0], None])
        ax1.set_xlim([-1, len(sortWins) + 1])
        ax2.plot(sortWins, sortPrMD, marker='d', linestyle='')
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        # ax2.set_xticklabels(sortWins, rotation=45, ha='right')
        # ax1.set_ylim(params['analysis']['EFitLim'])
        ax1.legend(loc='best', ncol=2)
        ax3.legend(loc='best', ncol=2)
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax3.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        print(f'averaged len(wins), {len(wins)} fits')
        # Now the two p-valued
        # These may have different weights
        # E+ PVal
        sortOrder = np.flip(wf[0, :].argsort())
        sortWfP = wf[0, sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEP = fEP[sortOrder][0: nFits]
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})  # noqa: E501
        ax1.errorbar(sortWins, y=gv.mean(sortEP), yerr=gv.sdev(sortEP), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved[0], ax1, colour='tab:red', ls='--',  lab='$E^{+}_{pval} = ' + f'{pValAved[0]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0], None])
        ax1.set_xlim([-1, len(sortWins) + 1])
        ax1.legend(loc='best', ncol=2)
        ax2.plot(sortWins, sortWfP, marker='+', linestyle='')
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        # ax2.set_xticklabels(sortWins, rotation=45, ha='right')
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{+}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        # E+ PVal
        sortOrder = np.flip(wf[1, :].argsort())
        sortWfP = wf[1, sortOrder][0: nFits]
        sortWins = np.asarray(wins)[sortOrder][0: nFits]
        sortEM = fEM[sortOrder][0: nFits]
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16.6, 11.6), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})  # noqa: E501
        ax1.errorbar(sortWins, y=gv.mean(sortEM), yerr=gv.sdev(sortEM), marker='d', linestyle='')
        ax1 = GVP.myHSpan(pValAved[1], ax1, colour='tab:green', ls=':',  lab='$E^{-}_{pval} = ' + f'{pValAved[1]}$')  # noqa: E501
        # finalising
        # ax1.set_xlim([sortWins[0] - 1, None])
        # ax1.set_xlim([-1, len(sortWins) + 1])
        ax1.tick_params(labelbottom=False, labeltop=True, top='on', direction='out')
        # ax1.set_xlim([-1, None])
        ax1.legend(loc='best', ncol=2)
        ax2.plot(sortWins, sortWfP, marker='+', linestyle='')
        # xTickLabels = range(1,len(sortWins) + 1)
        ax2.set_xticklabels(sortWins, rotation=90)  # , ha='right')
        ax1.set_xticklabels(sortWins, rotation=90)
        for tickLab in ax1.get_xaxis().get_ticklabels()[1::2]:  # Getting every odd xtick label
            tickLab.set_visible(False)
        for tickLab in ax2.get_xaxis().get_ticklabels()[::2]:  # Getting every 2nd xtick label
            tickLab.set_visible(False)
        ax1.set_ylabel('$a_\\tau\,E_{fit}^{-}$', fontsize=36)  # noqa: W605
        ax2.set_xlabel('fit window')
        ax2.set_ylabel('weight')
        pdf.savefig(fig)
        plt.close(fig)
        # Close the pdf
        print(f'Closing the pdf {os.path.join(thisAnaDir, "mAve.pdf")}')
        pdf.close()
    # Outsideloop
    sys.exit('Finished')


if __name__ == '__main__':
    mo.initBigPlotSettings()
    # mpl.rcParams['lines.markersize'] = 16.0
    # For Poster/Presentation
    # mpl.rcParams['ytick.labelsize'] = 32
    # mpl.rcParams['xtick.labelsize'] = 32
    mpl.rcParams['font.size'] = 36
    main(sys.argv[1:])
