
import sys
from typing import Any, List, Dict
import numpy as np  # type: ignore
from scipy import linalg  # type: ignore
import gvar as gv  # type: ignore
import os
import sympy as sp  # type: ignore
import opt_einsum  # type: ignore
from myModules import clean_filename


gvarSeed = 100


def strToLambda(string):
    """
    Converts a string to a lambda function
    Returns the function + variables
    String has format
    outVar:var1,,,...,_N: f(var1,,,...,_N)
    where f(...) must be an appropriate maths function
    such as x**2 + 2*x + 1
    """
    # print(string)
    # Splitting the string into variables and functions
    outVar, allVar, func = string.split(':')
    allVar = allVar.split(',')
    # initialising symbolic variables with keys
    varDict = {}
    for v in allVar:
        varDict.update({v: sp.symbols(v)})
    # now putting the variables into a tuple
    symVars = []
    for k, v in varDict.items():
        symVars.append(v)
    # reverse to make ordering logical (i.e. 1st variables is first now)
    symVars = tuple(reversed(symVars))
    # print(func, type(func))
    # print(symVars, type(symVars[0]))
    # Making the lambda function
    # print(symVars, func)
    f = lambda *symVars: eval(func)  # noqa: E731
    # Dumping the variables into global namespace
    # So can make symbolic lambda function
    globals().update(varDict)
    # Making symbolic lambda function
    lam_f = sp.lambdify(symVars, f(*symVars))
    # print(lam_f(1, 2, 3))
    # print(allVar, func)
    return lam_f, allVar, outVar


def sortEvals(w: np.ndarray, vl: np.ndarray, vr: np.ndarray) -> np.ndarray:
    """
    Sorts the eigenvectors according to the eigenvalues w
    returns w, vl, vr but sorted
    I.e. sorted so max eigenvalue is first
    """
    sys.exit('Eigenvector sorting is broken')
    # Makes them basically all the same. Clearly wrong
    argsort = np.flip(np.argsort(w))
    wSorted = np.take_along_axis(w, argsort, axis=0)
    argsortExpand = np.expand_dims(argsort, axis=1)
    vlSorted = np.take_along_axis(vl, argsortExpand, axis=0)
    vrSorted = np.take_along_axis(vr, argsortExpand, axis=0)
    return wSorted, vlSorted, vrSorted


def performVarAnalysis(GList: List[np.ndarray], n: int, t0: int, dt: int, src_t0: int) -> List[np.ndarray]:  # noqa: E501
    """
    Performs a variational analysis on the correlators in GList
    Assembles them into a matrix.
    """
    if len(GList) != n*n:
        sys.exit(f'Invalid size of correlation matrix, n = {n}, len = {len(GList)}')
    count = 0
    G = np.zeros(list(np.shape(GList[0])) + [n, n])
    for ii in range(0, n):
        for jj in range(0, n):
            G[..., ii, jj] = GList[count]
            count = count + 1
    # Symmeterising
    for it in range(0, np.shape(G)[1]):
        for nc in range(0, np.shape(G)[0]):
            G[nc, it, :, :] = 0.5 * (G[nc, it, :, :] + G[nc, it, :, :].T)
    # U,U* trick
    G[:, :, :, :] = 0.5 * (G[:, :, :, :] + np.conjugate(G[:, :, :, :]))
    # Normalising at src t0
    G_norm = np.zeros(n)
    for ix in range(0, n):
        G_norm[ix] = np.mean(G[:, src_t0, ix, ix], axis=0)
    for nc in range(0, np.shape(G)[0]):
        for ix in range(0, n):
            G[nc, :, ix, :] = G[nc, :, ix, :] / np.sqrt(abs(G_norm[ix]))
            G[nc, :, :, ix] = G[nc, :, :, ix] / np.sqrt(abs(G_norm[ix]))
    # Solving the generalised eigenvalue equation
    A = np.mean(G[:, t0+dt, :, :], axis=0)
    B = np.mean(G[:, t0, :, :], axis=0)
    w, vl, vr = linalg.eig(A, B, left=True, right=True)
    w = np.real(w)
    # Sorting eigenvectors
    # Already sorted by eigenvalue
    # w, vl, vr = sortEvals(w, vl, vr)
    # projecting
    GProj = []
    blah = np.einsum('ia, ntij,ja->nta', vl, G, vr)
    for ii in range(0, n):
        GProj.append(blah[:, :, ii])
    return GProj


def performVarAnalysisGvar(GList: List[np.ndarray], n: int, t0: int, dt: int, src_t0: int, saveDir: str, newVec=True) -> List[np.ndarray]:  # noqa: E501
    """
    Performs a variational analysis on the correlators in GList
    Assembles them into a matrix.
    Saves the eigenvectors
    """
    if len(GList) != n*n:
        sys.exit(f'Invalid size of correlation matrix, n = {n}, len = {len(GList)}')
    count = 0
    G = np.zeros(list(np.shape(GList[0])) + [n, n], dtype='object')
    for ii in range(0, n):
        for jj in range(0, n):
            G[..., ii, jj] = GList[count]
            count = count + 1
    # Symmeterising
    for it in range(0, np.shape(G)[1]):
        G[it, :, :] = 0.5 * (G[it, :, :] + G[it, :, :].T)
    # U,U* trick
    # G[:, :, :] = 0.5 * (G[:, :, :] + np.conjugate(G[:, :, :]))
    # Normalising at src t0
    G_norm = np.zeros(n, dtype='object')
    for ix in range(0, n):
        G_norm[ix] = G[src_t0, ix, ix]
    for ix in range(0, n):
        G[:, ix, :] = G[:, ix, :] / np.sqrt(np.fabs(G_norm[ix]))
        G[:, :, ix] = G[:, :, ix] / np.sqrt(np.fabs(G_norm[ix]))
    if newVec:
        print('Solving GEVP')
        # Solving the generalised eigenvalue equation
        A = gv.mean(G[t0+dt, :, :])
        B = gv.mean(G[t0, :, :])
        w, vl, vr = linalg.eig(A, B, left=True, right=True)
        w = np.real(w)
        # Sorting eigenvectors
        # Already sorted by eigenvalue
        # w, vl, vr = sortEvals(w, vl, vr)
        # Saving the eigenvectors
        np.savetxt(os.path.join(saveDir, 'vr.txt'), vr, header='right eigenvectors')
        np.savetxt(os.path.join(saveDir, 'vl.txt'), vl, header='left eigenvectors')
        print(saveDir)
    else:
        print('Loading GEVP')
        vr = np.loadtxt(os.path.join(saveDir, 'vr.txt'))
        vl = np.loadtxt(os.path.join(saveDir, 'vl.txt'))
    # projecting
    GProj = []
    # print(vl, vr, G.shape)
    blah = opt_einsum.contract('ia, tij,ja->ta', vl, G, vr, backend='object')
    for ii in range(0, n):
        GProj.append(blah[:, ii])
    return GProj


def labelNorm(G: np.ndarray, params):
    """
    normallise on the label-to-analyse level
    i.e. normallises whatever correlator is fed in
    G is 2dim. [ncon, NT]
    """
    if 'norm' not in params['cfuns'].keys():
        return G
    if params['cfuns']['norm'] == 'labelSrcAve':
        G[:, :] = G[:, :] / np.mean(G[:, int(params['analysis']['src_t0'])])
    else:
        print(f'Not normallising on label level as norm == {params["cfuns"]["norm"]}')
    return G


# def loadSingleMeson(cfFile: str) -> np.ndarray():
def loadSingleMeson(cfFile):
    """
    Loads a single meson correlator in gencf format
    Returns real component only.
    """
    myDtype = '>c16'
    return np.real(np.fromfile(cfFile, dtype=myDtype))


def setParityMask(proj: str) -> np.ndarray:
    """
    returns the appropriate mask in column major order
    """
    if proj == '+':
        # (1,1), (2,2)
        maskFlat = np.array([False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True])  # noqa: E501
    elif proj == '-':
        # (3,3), (4,4)
        maskFlat = np.array([True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False])  # noqa: E501
    elif proj == '11':
        # (1,1)
        # Positive background Field, positive parity
        maskFlat = np.array([False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])  # noqa: E501
    elif proj == '33':
        # (3,3)?
        # Positive background Field, negative parity
        sys.exit(f'proj {proj} Not implemented')
    elif proj == '22':
        # (2,2)
        # negative background Field, positive parity
        maskFlat = np.array([True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True])  # noqa: E501
    elif proj == '44':
        # (4,4)
        # negative background Field, Negative parity
        sys.exit(f'proj {proj} Not implemented')
    else:
        sys.exit(f'proj {proj} Not implemented')
    # Put the mask into matrix form
    return maskFlat.reshape([4, 4], order='F')


def loadGencfMomBaryon(cfFile: str, nt: int, numMom: int, proj: str = '+', mom: int = 0):
    """
    Loads a single momenta of a gencf baryon with no lorentz indices
    """
    ns = 4
    # nt = 64
    # Load the data into a dim 1 vector
    myDtype = '>c16'
    data_read = np.real(np.fromfile(cfFile, dtype=myDtype))
    # Reshape the data into a dim 4 matrix
    # Use column major ordering (i.e. fortran)
    data_baryon = data_read.reshape([ns, ns, nt, numMom, 1, 1], order='F')
    mask = setParityMask(proj)
    invMask = ~mask  # swap trues for false, false for true
    mCount = invMask.sum()

    """
    # Extend the mask to one for each time slice didn't work
    mask_nt = np.asarray([mask] * nt, order='F').reshape([ns, ns, nt], order='F')
    data_masked = np.ma.array(data_baryon[:, :, :, mom, 0, 0], mask=mask_nt)
    """

    # so do it timeslice per timeslice
    # Flatten teh data
    data = np.empty(nt)
    for tt in range(0, nt):
        if mCount > 1:
            # Sum over appropriate terms
            if mCount == 2:
                data[tt] = np.sum(np.ma.array(data_baryon[:, :, tt, mom, 0, 0], mask=mask).compressed())  # noqa: E501
            else:
                sys.exit(f'bad number of terms {mCount} kept via mask for proj {proj}')
        else:
            data[tt] = np.ma.array(data_baryon[:, :, tt, mom, 0, 0], mask=mask).compressed()
    return data


def loadOQCDMeson(cfFile: str, gOff: int = 5):
    """
    Loads an openqcd meson which does not have momenta
    gOff 5 is the standard Pseudoscalar or pion operator
    """
    data = np.genfromtxt(cfFile, delimiter=' ')
    cf = np.empty(data.shape[0])
    for jj in range(0, data.shape[0]):    # iterate over timeslices
        # this gets for the correct gamma matrices
        cf[jj] = data[jj, 2*gOff: 2*gOff + 1]
    return np.real(cf)


def loadSingleBaryon(cfFile):
    """
    Loads a single meson correlator in gencf format
    Returns real component only.
    """
    myDtype = 'float'
    return np.loadtxt(cfFile, dtype=myDtype)[:, 0]


def initCorrelators(params):
    """
    Loads all the correlators in, does the maths, etc
    """
    # Sets the random seed for gvar
    gv.ranseed(gvarSeed)
    # The labels for the correlators we're gonna load
    cfLabelsList = params['cfuns']['cfLabels']
    # We will store the loaded data, as well as the params for that data
    cfDict: Dict[str, Dict[str, Any]] = {'data': {},
                                         'params': {}}
    # Load the correlators in
    for cfLL in cfLabelsList:
        # load list of correlators

        if 'ext' in params['cfuns'][cfLL].keys():
            ext = params['cfuns'][cfLL]['ext']
        else:
            ext = ''

        with open(params['cfuns'][cfLL]['cfList']) as f:
            files = [fi.strip()+ext for fi in f.readlines()]
            cfDir = params['cfuns'][cfLL]['cfDir']
            cfList = [os.path.join(cfDir, cf) for cf in files]
        # Sort cfList based on the filename
        cfName = files[0]
        if 'icfg' in cfName:
            sortFiles = [fi.split('icfg')[1][0:9] for fi in files]
            sortOrder = np.asarray(sortFiles).argsort()
            cfList = np.asarray(cfList)[sortOrder]
        # Optionally skipping some configs
        # Uses a simple string in the correlator name
        if 'cfSkip' in params['cfuns'].keys():
            cfSkip = params['cfuns']['cfSkip']
        else:
            cfSkip = []
        print('pre', len(cfList))
        if len(cfSkip) > 0:
            for sk in cfSkip:
                for cf in cfList:
                    if sk in cf:
                        # string in
                        print(f'Removing correlator {cf} due to skip element {sk}')
                        cfList.remove(cf)
        print('post', len(cfList))

        # Optionally not using all the cfuns we wanted loaded
        if 'cfCut' in params['cfuns'].keys():
            cfCut = params['cfuns']['cfCut']
        else:
            cfCut = len(cfList)
        print(f'cfCut is {cfCut}')
        cfList = cfList[:cfCut]
        # Set the loading type
        if 'hadronType' not in params['cfuns'].keys():
            hType = 'meson'
            cfList = [cf+'.2cf' for cf in cfList]
        else:
            if params['cfuns']['hadronType'] == 'meson':
                hType = 'meson'
                cfList = [cf+'.2cf' for cf in cfList]
            elif params['cfuns']['hadronType'] == 'baryon':
                hType = 'baryon'
            elif params['cfuns']['hadronType'] == 'baryon-gencf':
                hType = 'baryon-gencf'
                mom = params['cfuns'][cfLL]['momSelect']
                nMom = params['cfuns'][cfLL]['momNumber']
                proj = params['cfuns'][cfLL]['proj']
            elif params['cfuns']['hadronType'] == 'OQCDmeson':
                hType = 'OQCDmeson'
                gam = params['cfuns'][cfLL]['gam']
            else:
                sys.exit(f'Unsupported hadron type {params["cfuns"]["hadronType"]}')

        Nt = params['cfuns'][cfLL]['latt']
        # load the correlators
        G = np.empty([len(cfList), Nt])
        for ii, cf in enumerate(cfList):
            # print(ii, cf)
            # G[ii, :] = loadOQCDMeson(cf, gam)
            # G[ii, :] = loadSingleMeson(cf)
            try:
                if hType == 'baryon':
                    G[ii, :] = loadSingleBaryon(cf)
                elif hType == 'meson':
                    G[ii, :] = loadSingleMeson(cf)
                elif hType == 'OQCDmeson':
                    G[ii, :] = loadOQCDMeson(cf, gam)
                elif hType == 'baryon-gencf':
                    G[ii, :] = loadGencfMomBaryon(cf, Nt, nMom, proj, mom)
                else:
                    sys.exit(f'invalid hType {hType}')
            except:  # noqa: E722
                sys.exit(f'Loading correlator {cf} failed. Exiting')
            if 'norm' in params['cfuns'].keys():
                # Normallise each correlator on config level according to method in norm
                if params['cfuns']['norm'] == 'src':
                    sys.exit('Do not use "src" averaging on configuration level')
                    print(f'normallising correlators at t = {int(params["analysis"]["src_t0"])}')
                    G[ii, :] = G[ii, :]/G[ii, int(params['analysis']['src_t0'])]
                elif params['cfuns']['norm'] == 'ave':
                    print('normallising correlators by average')
                    G[ii, :] = G[ii, :]/np.mean(G[ii, :])
                elif params['cfuns']['norm'] == 'sum':
                    print('normallising correlators by sum')
                    G[ii, :] = G[ii, :]/np.sum(G[ii, :])
                elif params['cfuns']['norm'] == 'srcAve':
                    G[ii, :] = G[ii, :]  # This happens after have loaded all the correlators
                elif params['cfuns']['norm'] == '':  # No normallisation
                    G[ii, :] = G[ii, :]
                elif params['cfuns']['norm'] == 'labelSrcAve':
                    G[ii, :] = G[ii, :]  # This happens after have loaded all the correlators and done maths  # noqa: E501
                else:
                    sys.exit(f'Unsupported normallise method {params["cfuns"]["norm"]}. Supported are src, ave and sum, srcAve')  # noqa: E501
        if 'norm' in params['cfuns'].keys():
            # Normallise by average
            if params['cfuns']['norm'] == 'srcAve':
                print('Normallising correlators by average accross correlators at src location')
                G[:, :] = G[:, :]/np.mean(G[:, int(params['analysis']['src_t0'])])
        # And update the storage dict
        cfDict['data'].update({cfLL: G})
        cfDict['params'].update({cfLL: params['cfuns'][cfLL]})
        print(f'Loaded correlators for label {cfLL} successfully')

    # Now do the maths we want to do
    smLabels = params['analysis']['symMathLabels']
    smMaths = params['analysis']['symMathMaths']
    if len(smLabels) != len(smMaths):
        print('smLabels', 'smMaths')
        if len(smLabels) < len(smMaths):
            for ii in range(0, len(smMaths)):
                try:
                    print(smLabels[ii], smMaths[ii])
                except IndexError:
                    print(smMaths[ii])
        else:
            for ii in range(0, len(smLabels)):
                try:
                    print(smLabels[ii], smMaths[ii])
                except IndexError:
                    print(smLabels[ii])
        sys.exit('symMath labels and Maths must be matched pair of lists')
    # Determining if we have any resampling (jackknife) to do
    resample = False
    if np.any(['resample' in x for x in smMaths]):
        resample = True
        resampleDict = {}        
        sys.exit('symMath labels and Maths must be matched pair of lists')
    for lab, math in zip(smLabels, smMaths):
        print('Doing math for ', lab, math)
        if 'var' in math:
            # I.e. the maths looks like
            # 'var:G11,G12,G21,G22:t0INTdtINT'
            # and the label looks like
            # "Name1, name2", i.e. of length n

            # Getting the variational parameters
            varParams = math.split(':')[-1]
            t0 = varParams.split('dt')[0]
            t0 = t0[2:]
            dt = varParams.split('dt')[1]
            src_t0 = params['analysis']['src_t0']
            print(f'Doing variational analysis with t0={t0} and dt={dt}')
            # GList must be organised in i.e. order
            # GList = [G11,G12,G13,G21,G22,G23,G32,G31,G33]

            allVar = math.split(':')[1]
            allVar = allVar.split(',')
            # Putting into the list
            GList = []
            for v in allVar:
                if v not in cfLabelsList:
                    print('label '+v+' is not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                GList.append(cfDict['data'][v])
            n = int(np.sqrt(len(GList)))
            GProj = performVarAnalysis(GList, n, int(t0), int(dt), int(src_t0))
            count = 0
            outVar = lab.split(',')

            for o in outVar:
                if o in cfLabelsList:
                    print(f'label {o} is already in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                # put into the correlator dictionary
                cfDict['data'].update({o: GProj[count]})
                # Just setting the params based on first of the variables, i.e. G11
                cfDict['params'].update({o: cfDict['params'][allVar[0]]})
                print('TODO: Check that maths is compatible. I.e. that from same ensemble, same configs. Using i.e ILAFF or manual checks')  # noqa: E501
                # And update the list of labels for 'correlators' we have
                cfLabelsList.append(o)
                count = count + 1
        elif 'flip' in math:
            # I.e. the maths looks like
            # 'flip:G11:G11'
            # and the label looks like
            # "Name1", i.e. of length 1
            # This works only for a single correlator

            # Grabbing that single correlator data
            allVar = math.split(':')[1]
            allVar = allVar.split(',')
            # Putting into the list
            GList = []
            for v in allVar:
                if v not in cfLabelsList:
                    print('label '+v+' is not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                GList.append(cfDict['data'][v])
            n = len(GList)
            if n > 1:
                sys.exit(f'Flip only works for single label. Multiple {allVar} selected')
            corrName = allVar[0]
            G = np.copy(GList[0])
            print(f'Reversing correlator, i.e. G(t) -> G(Nt - t) for {corrName}')
            # The below loop is equivalent to the flip
            # NT = np.shape(G)[-1]
            # GR = np.empty(G.shape)
            # for ii in range(1, nT+1):
            #     GR[..., ii-1] = G[..., NT - ii]
            # Flip only along time-axis to maintain cfg index
            GR = np.flip(G, axis=-1)
            print(GR.shape, G.shape, GList[0].shape)
            # Renormalising for new 'src' location
            if params['cfuns']['norm'] == 'src':
                for ii in range(0, GR.shape[0]):
                    # print(f'normallising correlators at t = {int(params["analysis"]["src_t0"])}')
                    GR[ii, :] = GR[ii, :]/GR[ii, int(params['analysis']['src_t0'])]

            # Put into the correlator dictionary
            cfDict['data'].update({lab: GR})
            cfDict['params'].update({lab: cfDict['params'][allVar[0]]})
            cfLabelsList.append(lab)
        elif 'resample' in math:
            # I.e. the maths looks like
            # 'resample:G11,G22:G11/G22'
            # Then it calculates mean + resamples for G11, G22
            func, aV, outVar = strToLambda(math)
            myVar = {}
            myVarJack = []
            for var in aV:
                # print(var, np.mean(cfDict['data'][var], axis=0)[0:5])
                # sys.exit()
                if var not in cfLabelsList:
                    print('label '+var+' not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                # Grab the data
                myVar.update({var: cfDict['data'][var]})
                myVarJack.append(mo.doJack(cfDict['data'][var], order=2))
            # Package into a tuple so can unpack
            myVarJackTuple = tuple(myVarJack)
            # and call the function
            dataJack = func(*myVarJackTuple)
            dataGV = gv.gvar(dataJack[0, 0, ...], mo.jackCov(dataJack[:, 0, ...]))
            del myVarJack
            gc.collect()
            # Put it in the resampleDictionary for later
            resampleDict.update({lab: dataGV})
            # and update the rest of the lists
            cfDict['params'].update({lab: cfDict['params'][aV[0]]})
            # And update the list of labels for 'correlators' we have
            cfLabelsList.append(lab)
        else:
            func, aV, outVar = strToLambda(math)
            myVar = []
            for var in aV:
                # print(var, np.mean(cfDict['data'][var], axis=0)[0:5])
                # sys.exit()
                if var not in cfLabelsList:
                    print('label '+var+' not in cfLabelsList:')
                    print(*[f'{x}  \n' for x in cfLabelsList])
                    sys.exit('Exiting')
                if 'R' in lab:
                    print('rat', lab, var, cfDict['data'][var].shape, np.mean(cfDict['data'][var], axis=0)[0:3])  # noqa: E501
                # Here we set the variables to be appropriate values
                myVar.append(cfDict['data'][var])
            # Now make it into a tuple so can unpack
            myVarTuple = tuple(myVar)
            # and call the function and put into correlator dictionary with label
            cfDict['data'].update({lab: func(*myVarTuple)})
            # print(lab, np.mean(cfDict['data'][lab], axis=0)[0:5])
            cfDict['params'].update({lab: cfDict['params'][aV[0]]})
            print('TODO: Check that maths is compatible. I.e. that from same ensemble, same configs. Using i.e ILAFF or manual checks')  # noqa: E501
            # And update the list of labels for 'correlators' we have
            cfLabelsList.append(lab)
    if resample:
        return cfDict, cfLabelsList, resampleDict
    else:
        return cfDict, cfLabelsList


def preNorm(params, cfDataDict):
    """
    Normallises correlators before gvar
    Normallise each correlator on config level according to ensemble average ofmethod in norm
    """
    if 'norm' in params['cfuns'].keys() and 'pre' in params['cfuns']['norm']:
        for k, G in cfDataDict.items():
            if params['cfuns']['norm'].split(':')[-1] == 'ave':
                # print('normallising correlators by average across all time slices')
                cfDataDict[k] = G[:, :] / np.mean(np.mean(G[:, :], axis=0), axis=1)
            elif params['cfuns']['norm'].split(':')[-1] == 'sum':
                # print('normallising correlators by sum across time slices')
                cfDataDict[k] = G[:, :] / np.sum(np.mean(G[:, :], axis=1))
            elif params['cfuns']['norm'].split(':')[-1] == 'src':
                # print(f'normallising correlators at t0 = {params["analysis"]["src_t0"]}')
                cfDataDict[k] = G[:, :] / np.mean(G[:, int(params['analysis']['src_t0'])])
            elif params['cfuns']['norm'].split(':')[-1] == '':  # No normallisation
                cfDataDict[k] = G
            else:
                sys.exit(f'Unsupported pre-gvar normallise method {params["cfuns"]["norm"]}. Supported are pre:ave, pre:src, pre:sum')  # noqa: E501
    return cfDataDict


def postNorm(params, GVD):
    """
    Normallises correlators after gvar
    Normallise each correlator according to method in norm
    """
    if 'norm' in params['cfuns'].keys() and 'post' in params['cfuns']['norm']:
        for k, G in GVD.items():
            if params['cfuns']['norm'].split(':')[-1] == 'ave':
                # print('normallising correlators by average across all time slices')
                GVD[k] = G / np.mean(G)
            elif params['cfuns']['norm'].split(':')[-1] == 'sum':
                # print('normallising correlators by sum across time slices')
                GVD[k] = G / np.sum(G)
            elif params['cfuns']['norm'].split(':')[-1] == 'src':
                # print(f'normallising correlators at t0 = {params["analysis"]["src_t0"]}')
                GVD[k] = G / G[int(params['analysis']['src_t0'])]
            elif params['cfuns']['norm'].split(':')[-1] == '':  # No normallisation
                GVD[k] = G
            else:
                sys.exit(f'Unsupported post-gvar normallise method {params["cfuns"]["norm"]}. Supported are post:ave, post:src, post:sum')  # noqa: E501
    return GVD


def GReconBaryon(N: int, N0: int, GZero: np.ndarray) -> np.ndarray:
    """
    Constructs the reconstructed correlator at higher temperature
    for baryon
    """
    m = int(N0/N)
    GR = np.zeros(N, dtype=object)
    for tt in range(0, N):
        for nn in range(0, m - 1 + 1):
            GR[tt] = GR[tt] + ((-1) ** nn) * GZero[tt + nn * N]
    return GR


def GReconMeson(N: int, N0: int, GZero: np.ndarray) -> np.ndarray:
    """
    Constructs the reconstructed correlator at higher temperature
    for meson
    Appendix B of 2209.14681
    """
    m = int(N0/N)
    GR = np.zeros(N, dtype=object)
    for tt in range(0, N):
        for nn in range(0, m - 1 + 1):
            GR[tt] = GR[tt] + GZero[tt + nn * N]
    return GR


def autoPadLength(N: int, N0: int, fac: int) -> int:
    """
    Works out the amount of data points to pad with
    such that N0/N = fac
    """
    # Handles cases where N is only just smaller
    pad = fac * N - N0
    # else much smaller
    if pad < 0:
        pad = 0
        padded = False
        while not padded:
            pad = pad + 1
            if ((N0 + pad) / N) % fac == 0:
                padded = True
    return pad


def extCorr(G: np.ndarray, NT: int, extType: str, where: str) -> np.ndarray:
    """
    A function to pad the correlator with specified value
    at specified values
    """
    if NT <= len(G):
        # Dont need to pad
        GR = G
    else:
        # Need to pad
        if extType == 'zero':
            # Pad with zeros
            pad = gv.gvar(['0(0)'] * (NT - len(G)))
        elif extType == 'min':
            # Pad with minimum value of correlator
            pad = gv.gvar([np.min(G)] * (NT - len(G)))
        elif extType == 'fit':
            sys.exit('Not yet implemented. Will do linear fit to log at where')
        else:
            sys.exit(f'Bad extType = {extType}. Valid are zero, min')
        # where?
        if where == 'end':
            GR = np.concatenate((G, pad))
        elif where == 'NT2':
            mid = int(len(G) / 2)
            leftG = G[:mid]
            rightG = G[mid:]
            GR = np.concatenate((leftG, pad, rightG))
        elif where == 'min':
            mid = np.argmin(G)  # type: ignore
            leftG = G[:mid]
            rightG = G[mid:]
            GR = np.concatenate((leftG, pad, rightG))
        else:
            sys.exit(f'Unsupported extend location {where}')
    return GR


def initCorrelatorsGvar(params):
    """
    Loads correlators, converts to gv dataset, does the maths
    """
    # Sets the random seed for gvar
    gv.ranseed(gvarSeed)
    # The labels for the correlators we're gonna load
    cfLabelsList = params['cfuns']['cfLabels']
    # We will store the loaded data, as well as the params for that data
    cfParamsDict: Dict[str, Dict[str, Any]] = {}
    cfDataDict: Dict[str, np.ndarray] = {}
    # Load the correlators in
    for cfLL in cfLabelsList:
        # A quick way to use the same list
        # adds an extension to the correlator name
        # i.e. for smearing
        if 'ext' in params['cfuns'][cfLL].keys():
            ext = params['cfuns'][cfLL]['ext']
        else:
            ext = ''
        # load list of correlators
        with open(params['cfuns'][cfLL]['cfList']) as f:
            files = [fi.strip()+ext for fi in f.readlines()]
        cfDir = params['cfuns'][cfLL]['cfDir']
        cfList = [os.path.join(cfDir, cf) for cf in files]
        # Sort cfList based on the filename
        cfName = files[0]
        if 'icfg' in cfName:
            sortFiles = [fi.split('icfg')[1][0:9] for fi in files]
            sortOrder = np.asarray(sortFiles).argsort()
            cfList = np.asarray(cfList)[sortOrder]
        # Optionally skipping some configs
        # Uses a simple string in the correlator name
        if 'cfSkip' in params['cfuns'].keys():
            cfSkip = params['cfuns']['cfSkip']
        else:
            cfSkip = []
        print('pre', len(cfList))
        if len(cfSkip) > 0:
            for sk in cfSkip:
                for cf in cfList:
                    if sk in cf:
                        # string in
                        print(f'Removing correlator {cf} due to skip element {sk}')
                        cfList.remove(cf)
        print('post', len(cfList))
        # Optionally not using all the cfuns we wanted loaded
        if 'cfCut' in params['cfuns'].keys():
            cfCut = params['cfuns']['cfCut']
        else:
            cfCut = len(cfList)
        print(f'cfCut is {cfCut}')
        cfList = cfList[:cfCut]
        # Set the loading type
        if 'hadronType' not in params['cfuns'].keys():
            hType = 'meson'
            cfList = [cf+'.2cf' for cf in cfList]
        else:
            if params['cfuns']['hadronType'] == 'meson':
                hType = 'meson'
                cfList = [cf+'.2cf' for cf in cfList]
            elif params['cfuns']['hadronType'] == 'baryon':
                hType = 'baryon'
            elif params['cfuns']['hadronType'] == 'baryon-gencf':
                hType = 'baryon-gencf'
                mom = params['cfuns'][cfLL]['momSelect']
                nMom = params['cfuns'][cfLL]['momNumber']
                proj = params['cfuns'][cfLL]['proj']
            elif params['cfuns']['hadronType'] == 'OQCDmeson':
                hType = 'OQCDmeson'
                gam = params['cfuns'][cfLL]['gam']
            else:
                sys.exit(f'Unsupported hadron type {params["cfuns"]["hadronType"]}')
        # Have set the type
        Nt = params['cfuns'][cfLL]['latt']
        # load the correlators
        G = np.empty([len(cfList), Nt])
        for ii, cf in enumerate(cfList):
            # print(ii, cf)
            # G[ii, :] = loadSingleBaryon(cf)
            try:
                if hType == 'baryon':
                    G[ii, :] = loadSingleBaryon(cf)
                elif hType == 'meson':
                    G[ii, :] = loadSingleMeson(cf)
                elif hType == 'OQCDmeson':
                    G[ii, :] = loadOQCDMeson(cf, gam)
                elif hType == 'baryon-gencf':
                    G[ii, :] = loadGencfMomBaryon(cf, Nt, nMom, proj, mom)
                else:
                    sys.exit(f'invalid hType {hType}')
            except:  # noqa: E722
                sys.exit(f'Loading correlator {cf} failed. Exiting')
        # And update the storage dicts
        cfDataDict.update({cfLL: G})
        cfParamsDict.update({cfLL: params['cfuns'][cfLL]})
        print(f'Loaded correlators for label {cfLL} successfully')

    # Optionally normallise
    cfDataDict = preNorm(params, cfDataDict)
    # Now convert to gvar
    GVD = gv.dataset.avg_data(cfDataDict)
    # Now will do the maths on the gvar as well as the original for now
    # Now do the maths we want to do
    smLabels = params['analysis']['symMathLabels']
    smMaths = params['analysis']['symMathMaths']
    if len(smLabels) != len(smMaths):
        sys.exit('symMath labels and Maths must be matched pair of lists')
    for lab, math in zip(smLabels, smMaths):
        print('Doing math for ', lab, math)
        if 'var' in math:
            # I.e. the maths looks like
            # 'var:G11,G12,G21,G22:t0INTdtINT'
            # OR
            # 'var:G11,G12,G21,G22:path:t0INTdtINT'
            # where path is the full path to the eigenvectors to use
            # and the label looks like
            # "Name1, name2", i.e. of length n
            # Getting the variational parameters
            varParams = math.split(':')[-1]
            if '/' not in math:
                t0 = varParams.split('dt')[0]
                t0 = t0[2:]
                dt = varParams.split('dt')[1]
                src_t0 = params['analysis']['src_t0']
                print(f'Doing variational analysis with t0={t0} and dt={dt}')
                newVec = True
                vrDir = os.path.join(params['analysis']['anaDir'], clean_filename(lab))
                if not os.path.exists(vrDir):
                    os.makedirs(vrDir)
            else:
                vrDir = math.split(':')[-2]
                newVec = False
                t0 = varParams.split('dt')[0]
                t0 = t0[2:]
                dt = varParams.split('dt')[1]
                src_t0 = params['analysis']['src_t0']
            # GList must be organised in i.e. order
            # GList = [G11,G12,G13,G21,G22,G23,G32,G31,G33]
            allVar = math.split(':')[1]
            allVar = allVar.split(',')
            # Putting into the list
            GList = []
            for v in allVar:
                if v not in cfLabelsList:
                    print('label '+v+' is not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                GList.append(GVD[v])
            n = int(np.sqrt(len(GList)))
            GProj = performVarAnalysisGvar(GList, n, int(t0), int(dt), int(src_t0), vrDir, newVec=newVec)  # noqa: E501
            count = 0
            outVar = lab.split(',')
            for o in outVar:
                if o in cfLabelsList:
                    print(f'label {o} is already in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                # put into the correlator dictionary
                GVD.update({o: GProj[count]})
                # Just setting the params based on first of the variables, i.e. G11
                cfParamsDict.update({o: cfParamsDict[allVar[0]]})
                # And update the list of labels for 'correlators' we have
                cfLabelsList.append(o)
                count = count + 1
        elif 'flip' in math:
            # I.e. the maths looks like
            # 'flip:G11:G11'
            # and the label looks like
            # "Name1", i.e. of length 1
            # This works only for a single correlator
            # Grabbing that single correlator data
            allVar = math.split(':')[1]
            allVar = allVar.split(',')
            # Putting into the list
            GList = []
            for v in allVar:
                if v not in cfLabelsList:
                    print('label '+v+' is not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                GList.append(GVD[v])
            n = len(GList)
            if n > 1:
                sys.exit(f'Flip only works for single label. Multiple {allVar} selected')
            corrName = allVar[0]
            G = np.copy(GList[0])
            print(f'Reversing correlator, i.e. G(t) -> G(Nt - t) for {corrName}')
            # The below loop is equivalent to the flip
            # NT = np.shape(G)[-1]
            # GR = np.empty(G.shape)
            # for ii in range(1, nT+1):
            #     GR[..., ii-1] = G[..., NT - ii]
            # Flip only along time-axis to maintain cfg index
            GR = np.flip(G, axis=-1)
            GVD.update({lab: GR})
            cfParamsDict.update({lab: cfParamsDict[allVar[0]]})
            cfLabelsList.append(lab)
        elif 'toy:' in math[0:8]:
            print('Doing toy math')
            # A toy correlator
            # format:
            # toy:TOYTYPE:[A(AErr),B(BErr)]_[C(CErr),D(DErr)]_NT_xmin_xmax
            # Where the lists can be however long you like
            # NT is more accurately the temperature really
            # xmin, xmax are ints that dictate the range of tau values
            # i.e. if start at 0, Nt = 16 for 16 total
            splits = math.split(':')
            toyType = splits[1]
            toyA = gv.gvar(splits[2].split('_')[0][1:-1].split(','))
            toyE = gv.gvar(splits[2].split('_')[1][1:-1].split(','))
            p = {'A': toyA, 'E': toyE}
            NT = float(splits[2].split('_')[2])
            xMin = int(splits[2].split('_')[3])
            xMax = int(splits[2].split('_')[4])
            tauRange = np.asarray(range(xMin, xMax))
            if len(toyA) != len(toyE):
                sys.exit(f'A coefficients {toyA} and E coefficients {toyE} must have same length')
            if toyType == 'baryon':
                toyFunc = constructBaryonToyFunc(tauRange, p, NT)
            elif toyType == 'meson':
                toyFunc = constructMesonToyFunc(tauRange, p, NT)
            else:
                sys.exit(f'bad ToyFunc {toyType}. Try meson or baryon')
            # Need to update the cfParamsDict
            # Don't think it ever gets used though...
            cfParamsDict.update({lab: {'p': p,
                                       'NT': NT,
                                       'xMin': xMin,
                                       'xMax': xMax,
                                       'toyType': toyType}})
            GVD.update({lab: toyFunc(tauRange, p)})
            cfLabelsList.append(lab)
        elif 'ext:' in math[0:8]:
            # I.e. the maths looks like
            # 'ext:G11:NT_extType_where'
            # and the label looks like
            # "Name1", i.e. of length 1
            # This works only for a single correlator
            # Grabbing that single correlator data
            # Does not extend if already longer
            # extType defines what to extend with
            # where defines where to extend (middle or end)
            allVar = math.split(':')[1]
            allVar = allVar.split(',')
            # Putting into the list
            GList = []
            for v in allVar:
                if v not in cfLabelsList:
                    print('label '+v+' is not in cfLabelsList:', cfLabelsList)
                    sys.exit('Exiting')
                GList.append(GVD[v])
            n = len(GList)
            if n > 1:
                sys.exit(f'Flip only works for single label. Multiple {allVar} selected')
            G = np.copy(GList[0])
            NT = int(math.split(':')[2].split('_')[0])
            extType = math.split(':')[2].split('_')[1]
            where = math.split(':')[2].split('_')[2]
            GR = extCorr(G, NT, extType, where)
            # Put into the correlator dictionary
            GVD.update({lab: GR})
            cfParamsDict.update({lab: cfParamsDict[allVar[0]]})
            cfLabelsList.append(lab)
        elif 'recon' in math[0:8]:
            """
            Construct a meson or baryon reconstructed correlator
            I.e. for mesons, see Appendix B of 2209.14681
            A similar relation applies for mesons
            This is effectively a resummation of the correlator
            form:
            recon:GLABEL:[meson/baryon]:NT:extType
            The new correlator will have length NT
            extType is what to pad with, i.e. zero or min
            """
            reconType = math.split(':')[2]
            # The correlator to be 'recon/resummed'
            # i.e. the zero temp correlator
            var = math.split(':')[1]
            # The length of the new recon correlator
            NT = int(math.split(':')[3])
            # The type to pad with
            extType = math.split(':')[4]
            # Get the correlator to be resummed
            if var not in cfLabelsList:
                print('label '+var+' not in cfLabelsList:')
                print(*[f'{x}  \n' for x in cfLabelsList])
                sys.exit('Exiting')
            else:
                G = GVD[var]
            # The length of the original correlator
            N0 = len(G)
            if reconType == 'baryon':
                div = int(np.floor(N0 / NT))
                # Must be odd multiple
                if div % 2 == 0:
                    fac = div + 1
                else:
                    fac = div
                pad = autoPadLength(NT, N0, fac)
                Gext = extCorr(G, N0 + pad, extType, 'min')
                # Also add the extended correlator to the dataset
                GVD.update({f'{lab}_ext': Gext})
                cfParamsDict.update({f'{lab}_ext': cfParamsDict[var]})
                # And update the list of labels for 'correlators' we have
                cfLabelsList.append(f'{lab}_ext')
                # Now construct the recon
                test = GReconBaryon(NT, len(Gext), Gext)
            elif reconType == 'meson':
                if N0 % NT != 0:
                    pad = autoPadLength(NT, N0, 1)
                    Gext = extCorr(G, N0 + pad, extType, 'NT2')
                else:
                    Gext = G
                # Also add the extended correlator to the dataset
                GVD.update({f'{lab}_ext': Gext})
                cfParamsDict.update({f'{lab}_ext': cfParamsDict[var]})
                # And update the list of labels for 'correlators' we have
                cfLabelsList.append(f'{lab}_ext')
                # Now construct the recon
                test = GReconMeson(NT, len(Gext), Gext)
            else:
                sys.exit(f'bad reconType {reconType} in {math}')
            GVD.update({lab: test})
            cfParamsDict.update({lab: cfParamsDict[var]})
            # And update the list of labels for 'correlators' we have
            cfLabelsList.append(lab)
        else:
            func, aV, outVar = strToLambda(math)
            myGVVar = []
            for var in aV:
                if var not in cfLabelsList:
                    print('label '+var+' not in cfLabelsList:')
                    print(*[f'{x}  \n' for x in cfLabelsList])
                    sys.exit('Exiting')
                # Here we set the variables to be appropriate values
                myGVVar.append(GVD[var])
            # Now make it into a tuple so can unpack
            myGVVarTuple = tuple(myGVVar)
            test = func(*myGVVarTuple)
            # and update the dictionaries
            GVD.update({lab: test})
            cfParamsDict.update({lab: cfParamsDict[aV[0]]})
            # And update the list of labels for 'correlators' we have
            cfLabelsList.append(lab)
    # Optionally normallise
    GVD = postNorm(params, GVD)
    return {'data': GVD, 'params': cfParamsDict}, cfLabelsList


def constructBaryonToyFunc(x, p, NT):
    """
    Constructs the lsqfit form of the baryontoyfunc at this NT
    """
    def baryonToyFuncNT(x, p):
        """
        A single forward and a single back exponential
        as suitable for a baryon
        in form suitable for lsqfit
        p[A], p[E] are alternating pos/neg amplitues/energies
        """
        ans = 0
        count = 0
        for A, E in zip(p['A'], p['E']):
            if count % 2 == 0:
                # forward
                term = A * np.exp(-E * x) / (1 + np.exp(-E * NT))
            else:
                # backward
                term = A * np.exp(E * x) / (1 + np.exp(E * NT))
            ans = ans + term
            count = count + 1
        return ans
    return baryonToyFuncNT


def constructMesonToyFunc(x, p, NT):
    """
    Constructs the lsqfit form of the meson toyfunc at this NT
    """
    def mesonToyFuncNT(x, p):
        """
        A single forward and a single back exponential
        as suitable for a meson
        in form suitable for lsqfit
        p[A], p[E] are amplitudes and energies
        """
        ans = 0
        for A, E in zip(p['A'], p['E']):
            # forward
            term = A * np.exp(-E * x) / (1 - np.exp(-E * NT))
            # backward
            term = term + - A * np.exp(E * x) / (1 - np.exp(E * NT))
            ans = ans + term
        return ans
    return mesonToyFuncNT


def writeGVDCSV(outName: str, GVDD, keys):
    """
    Writes all the keys variables + values in GVDD to a csv
    format tau, lab1 Val, lab1 Err, lab2 Val, lab2 Err...
    ASSUMES ALL HAVE SAME LENGTH
    """
    print(keys)
    header = ['tau']
    for k in keys:
        header = header + [k, f'{k}_Err']
    headerString = ','.join(header)
    lines = []
    for tau in range(0, len(GVDD[keys[0]])):
        thisLine = [str(tau)]
        for k in keys:
            v = GVDD[k]
            thisLine = thisLine + [str(gv.mean(v[tau])), str(gv.sdev(v[tau]))]
        lines.append(','.join(thisLine))
    with open(outName, 'w') as f:
        f.write(headerString + '\n')
        for ll in lines:
            f.write(ll + '\n')


def getNT(GVDD):
    """
    gets the length of the entries in GVD
    """
    NTs = []
    for k, v in GVDD.items():
        NTs.append(len(v))
    return list(set(NTs))


def sepGVDNT(GVDD):
    """
    Puts all the entries of differing lengths in different dictionaries
    """
    NTs = getNT(GVDD)
    GAll = {}
    for NT in NTs:
        GAll.update({NT: {}})
        for k, v in GVDD.items():
            if len(v) == NT:
                GAll[NT].update({k: v})
    # and return
    return GAll


def saveCSV(anaDir: str, suffix: str, GVD, keys=['all']):
    """
    Saves the csv with the data labels in keys
    if keys[0] == 'all', then just save all
    This enables saving only derived quantities
    """
    # Separate out the different elements into a dictionary where each
    # member of the dictionary has same length
    NTGVDD = sepGVDNT(GVD['data'])
    # and now save each
    for NT, d in NTGVDD.items():
        if keys[0] == 'all':
            keysD = list(d.keys())
        else:
            keysD = [k for k in keys if k in d.keys()]
        if len(keysD) == 0:
            continue
        outFile = os.path.join(anaDir, f'NT_{NT}_{suffix}.csv')
        writeGVDCSV(outFile, d, keys=keysD)

