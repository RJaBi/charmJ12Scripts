from scipy.optimize import minimize  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Any
import sys


def effE_centre(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
    """
    The centre finite difference for effective energy
    """
    # Sometimes it fails for GVARs in a weird way (ZeroDivisionError)
    # This is i.e. when z is very large (10**22)
    # zero is my number for effE failed anyway
    z = (np.roll(GJ2, -int(massdt), axis=len(GJ2.shape)-1) + np.roll(GJ2, int(massdt), axis=len(GJ2.shape)-1))/(2.0*GJ2)  # noqa: E501
    try:
        return (1.0/massdt) * np.arccosh(z)
    except ZeroDivisionError:
        # List to accomodate gvar's easily
        out = []
        if len(z.shape) == 1:
            for nt, v in enumerate(z):
                try:
                    out.append((1.0/massdt) * np.arccosh(v))
                except ZeroDivisionError:
                    out.append(0)
        else:
            # Here this should work for jackknife subensembles
            for nt in range(0, np.shape(z)[-1]):
                v = z[..., nt]
                try:
                    out.append((1.0/massdt) * np.arccosh(v))
                except ZeroDivisionError:
                    out.append(np.zeros(np.shape(v)))
        return np.asarray(out)


def effE_forward(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
    """
    The standard forward finite difference effective mass
    """
    return (1.0/massdt) * np.log(GJ2 / np.roll(GJ2, -int(massdt), axis=len(GJ2.shape)-1))


def effE_solve(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
    """
    The mesonic correlator solve for mass
    """
    GRatio = GJ2/np.roll(GJ2, -1, axis=len(GJ2.shape) - 1)
    # initial values for solver
    p0EJ2 = (1.0/massdt) * np.arccosh((np.roll(GJ2, -int(massdt), axis=len(GJ2.shape) - 1) + np.roll(GJ2, int(massdt), axis=len(GJ2.shape) - 1)) / 2.0 * GJ2)  # noqa: E501
    np.nan_to_num(p0EJ2, copy=False)

    def coshFunc(mEff, t, nT):
        """
        The cosh function that should describe the ratio of correlators
        Eqn6.57 of Gattringer,Lang
        """
        return np.cosh(mEff*(t - nT/2))/np.cosh(mEff * (t + 1 - nT/2))
    # Bounds chosen arbitrarily
    bounds = [(-1, 3)]
    effEJ2 = np.zeros(np.shape(GJ2))
    itert = np.nditer(effEJ2[:, 0, :], flags=['multi_index'])
    # Looping over all 1st order and time
    # Looping over 2nd order as well is far too slow
    print('Setting 2nd order jackkife values to last 1st order jackknife value')
    for ii in itert:
        inds = itert.multi_index
        jackIn = inds[0]
        tIn = inds[-1]
        p0 = p0EJ2[jackIn, 0, tIn]

        def minFunc(x):
            ans = abs(GRatio[jackIn, 0, tIn] - coshFunc(x, tIn, np.shape(GJ2)[-1]))
            return ans

        dd = minimize(minFunc, p0, bounds=bounds)
        effEJ2[jackIn, :, tIn] = dd.fun[0]

    return effEJ2


def effE_barAna(GJ2: np.ndarray) -> np.ndarray:
    """
    Implements the 4 point effective mass for a baryon with periodic BC
    Get Pos Parity soln
    """
    nT = np.shape(GJ2)[-1]
    effE = np.empty(np.shape(GJ2))
    for nn in range(0, nT):
        g0 = GJ2[:, :, nn]
        g1N = 1 + nn
        if g1N >= nT:
            g1N = g1N - nT
        g1 = GJ2[:, :, g1N]
        g2N = 2 + nn
        if g2N >= nT:
            g2N = g2N - nT
        g2 = GJ2[:, :, g2N]
        g3N = 3 + nn
        if g3N >= nT:
            g3N = g3N - nT
        g3 = GJ2[:, :, g3N]
        xSol = (g1 * g2 - g0 * g3 + ((g1 * g2 - g0 * g3)**2.0 - 4.0 * (g1**2.0 - g0 * g2) * (g2**2.0 - g1*g3))**0.5) / (2.0 * (g1**2.0 - g0 * g2))  # noqa: E501
        effE[:, :, nn] = xSol
    return effE


def getEffE(params: Dict[str, Any], massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
    """
    Functionalising the calculation of eff mass for use in toher programs
    """
    if 'effEMethod' not in params['analysis'].keys():
        effEMethod = 'forward'
    else:
        effEMethod = params['analysis']['effEMethod']
    if effEMethod == 'centre':
        print('Using centre finite difference for effective energy')
        # Noisier
        effEJ2 = effE_centre(massdt, GJ2)
    elif effEMethod == 'solve':
        print('Solving for effective energy')
        effEJ2 = effE_solve(massdt, GJ2)
    elif effEMethod == 'forward':
        print('Using forward finite difference for effective energy')
        effEJ2 = effE_forward(massdt, GJ2)
    elif effEMethod == 'baryonPBC':
        print('Using four points baryon Periodic BC for eff energy')
        effEJ2 = effE_barAna(GJ2)
    else:
        sys.exit(f'Unsupported effective energy method {effEMethod}')
    # Reomving nans (replace with 0.0)
    np.nan_to_num(effEJ2, copy=False)
    return effEJ2
