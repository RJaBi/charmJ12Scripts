
import sys
from typing import Any, List, Dict
import numpy as np  # type: ignore
from scipy import linalg  # type: ignore
import gc
import os
import sympy as sp  # type: ignore
import gvar as gv  # type: ignore
import myModules as mo

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
    print(func, type(func))
    print(symVars, type(symVars[0]))
    # Making the lambda function
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


def labelNorm(G: np.ndarray, params):
    """
    normallise on the label-to-analyse level
    i.e. normallises whatever correlator is fed in
    G is 2dim. [ncon, NT]
    """
    if 'norm' in params['cfuns'].keys():
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
                    sys.exit('Do not use "src" averaging')
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
                    print('label '+var+' not in cfLabelsList:', cfLabelsList)
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
