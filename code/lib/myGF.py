"""
Functions concerning gaugefield parameter setting and tuning
"""

from typing import Tuple, Any, MutableMapping
import numpy as np
import gvar as gv  # type: ignore
import os
import pickle


def calcNu(xi0_g: float, xi0_q: float) -> float:
    """
    Calculates the nu parameter.
    i.e. nu = xi0_g/xi0_q
    i.e. ratio of gluonic and fermionic anistropies
    """
    return xi0_g / xi0_q


def calcKappa(m0: float, nu: float, xi0_g: float) -> float:
    """
    Calculates anistropic kappa
    As in the openqcd-fastsum anistropy.pdf documentations
    Eqn 13
    Also in hep-lat/0006019 apparently.
    """
    return 1.0 / (2.0 * (m0 + 1 + 3.0 * (nu / xi0_g)))


def calcM0(kappa: float, nu: float, xi0_g: float) -> float:
    """
    Calculates m0 from anistropic kappa
    i.e. the opposite
    As in the openqcd-fastsum anistropy.pdf documentations
    Eqn 13
    Also in hep-lat/0006019 apparently.
    """
    return (1.0 / (2 * kappa)) - 1 - 3.0 * (nu / xi0_g)


def jackCov(jack1: np.ndarray) -> np.ndarray:
    """
    Calculates covariance matrix from jackknife subensembles
    The data is of form [0:ncon,...]
    where 0 is the ensemble average value
    Returns matrix C[ti,]
    """
    ncon = float(jack1.shape[0] - 1)
    return np.cov(jack1, rowvar=False) * (ncon - 1)


def loadData(params: MutableMapping[str, Any], ana: int, y=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data in from the pickle files into a numpy array
    """
    if y:
        keys = params['modelAverages']['mALabels'][ana]
    else:
        keys = params['modelAverages']['xValues'][ana]
    # Removing duplicates which 100% shouldn't exist. Just in case and then resorting
    # keys = sorted(list(set(keys)))
    # print(keys)
    for ii, k in enumerate(keys):
        # Loading the pickle file
        pickleFile = os.path.join(params['modelAverages'][k]['pickleDir'], 'mAve.pickle')
        kd = pickle.load(open(pickleFile, 'rb'))
        # Rescaling the model average jackknife values such that their uncertainty
        # reflects that of the sigModelAve (which includes a sytematic due to fit window selecton)
        # this does not effect the mean value
        mAve = kd['modelAve']
        CVJackErr = np.diag(jackCov(mAve))**0.5
        for ff in range(1, mAve.shape[0]):
            mAve[ff, :] = mAve[ff, :] - (mAve[ff, :] - mAve[0, :]) * (kd['sigModelAve']/CVJackErr)
        # if this is the first key, make the containers for the data
        # if k == keys[0]:
        if ii == 0:
            data = np.empty(list(mAve.shape) + [len(keys)])
            sigModelAve = np.empty([mAve.shape[1], len(keys)])
        # If it's not, just do the assignment
        data[..., ii] = mAve
        sigModelAve[..., ii] = kd['sigModelAve']

    return data, sigModelAve


def loadDataGvar(params: MutableMapping[str, Any], ana: int, y=True) -> gv.gvar:
    """
    Loads the data in from the pickle files into Gvar
    """
    if y:
        keys = params['modelAverages']['mALabels'][ana]
    else:
        keys = params['modelAverages']['xValues'][ana]
    # Removing duplicates which 100% shouldn't exist. Just in case and then resorting
    # keys = sorted(list(set(keys)))
    # print(keys)
    for ii, k in enumerate(keys):
        # Loading the pickle file
        pickleFile = os.path.join(params['modelAverages'][k]['pickleDir'], 'mAve.pickle')
        kd = pickle.load(open(pickleFile, 'rb'))
        # Rescaling the model average jackknife values such that their uncertainty
        # reflects that of the sigModelAve (which includes a sytematic due to fit window selecton)
        # this does not effect the mean value
        mAve = kd['modelAve']
        CVJackErr = np.diag(jackCov(mAve))**0.5
        for ff in range(1, mAve.shape[0]):
            mAve[ff, :] = mAve[ff, :] - (mAve[ff, :] - mAve[0, :]) * (kd['sigModelAve']/CVJackErr)
        # if this is the first key, make the containers for the data
        # if k == keys[0]:
        if ii == 0:
            data = []
        # If it's not, just do the assignment
        data.append(gv.gvar(mAve[0, :], jackCov(mAve))[params['modelAverages']['massParam']])
    return np.asarray(data)


def makeGVar(params: MutableMapping[str, Any], ana: int, data: np.ndarray) -> gv.gvar:
    """
    Transforms the data into gv.gvars
    """
    # Package the y data into gvars
    dataList: gv.gvar = []
    for ii, k in enumerate(params['modelAverages']['mALabels'][ana]):
        dataCov = jackCov(data[..., ii])
        dataList.append(gv.gvar(data[0, :, ii], dataCov))
    return np.asarray(dataList)
