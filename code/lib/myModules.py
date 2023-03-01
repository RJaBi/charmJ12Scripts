
from typing import Dict, Any, MutableMapping, Tuple, List
import matplotlib as mpl  # type: ignore
import numpy as np  # type: ignore
import sys
import toml
import unicodedata
import string


def GetArgs(args: list) -> MutableMapping[str, Any]:
    """
    Tests the args for appropriateness,
    loads the parameters in from the parameter file specified in args
    """
    print(args)
    if len(args) > 1:
        sys.exit('invalid number of arguments presented')
    if 'toml' in args[0]:
        print('load toml from '+args[0])
        params = toml.load(args[0])
    else:
        sys.exit('Not a toml file '+args[0])
    return params


def initBigPlotSettings():
    """
    Initialise a bunch of plot settings that make the plots look nicer
    """
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['errorbar.capsize'] = 3  # restoring the caps on error bars
    print(mpl.rcParams['lines.linewidth'])
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markeredgewidth'] = 0.5
    mpl.rcParams['figure.max_open_warning'] = 50
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.serif'] = ['Computer Modern']
    mpl.rcParams['lines.markersize'] = 5.0
    mpl.rcParams['figure.figsize'] = (16.6, 11.6)
    print('Updated Plot Settings')


def refineXYLims(params, subDict='analysis') -> Dict[str, Any]:
    """
    Toml does not support None types.
    Matplotlib variables set to None will just use the default value
    I.e. if i set the x limit, ax.set_xlim([None, None]), matplotlib will
    choose whatever limit it thinks is appropriate.
    Hence if the toml has an xlimit of i.e. ['None',], then this will set it to
    [None,]. Will work for both indices.

    This will look at all keys with 'Lim' in them under 'analysis'
    """
    if subDict is None:
        myDict = params
    else:
        myDict = params[subDict]
    for k, v in myDict.items():
        if 'Lim' in k:
            for ii, x in enumerate(v):
                if x == 'None':
                    v[ii] = None
    # Now that have updated, return.
    return params


markers = ['o', 's', 'p', 'P', '*', 'X', 'd', '<', '>', '$\\boxtimes$', '$\\bigoplus$', '$\\bigotimes$']  # noqa: E501


def removeZero(data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Removes any occasions where the data is zero
    returns the removed, and also the indices which were kept
    works fine for gvar
    """
    keep = []
    for ii, dat in enumerate(data):
        if dat != 0:
            keep.append(ii)
    newData = data[keep]
    return newData, keep


def replaceZero(data: np.ndarray, rep: float) -> np.ndarray:
    """
    relaces any occasions where the data is zero with rep
    """
    keep = []
    for ii, dat in enumerate(data):
        if dat != 0:
            keep.append(ii)
    newData = np.ones(np.shape(data)) * rep
    newData[keep] = data[keep]
    return newData


def clean_filename(filename, whitelist=None, replace=' ', char_limit=255):
    """
    Modified from
    Url: https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    """
    if whitelist is None:
        whitelist = "-_.() %s%s" % (string.ascii_letters, string.digits)
    else:
        whitelist = whitelist + list("-_.() %s%s" % (string.ascii_letters, string.digits))
    # replace spaces
    for r in replace:
        filename = filename.replace(r, '_')
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    # keep only whitelisted chars
    # test = [c for c in cleaned_filename if c in whitelist]
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename) > char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))  # noqa: E501
    return cleaned_filename[:char_limit]


def replace_nth(s, sub, repl, n=1):
    """
    Taken from
    https://stackoverflow.com/questions/46705546/python-replace-every-nth-occurrence-of-string
    Replaces the nth 'sub' by 'repl' in s
    """
    chunks = s.split(sub)
    size = len(chunks)
    rows = size // n + (0 if size % n == 0 else 1)
    return repl.join([
        sub.join([chunks[i * n + j] for j in range(n if (i + 1) * n < size else size - i * n)])
        for i in range(rows)
    ])


def doJack(data: np.ndarray, order=1):
    """
    Does jackknife resampling
    i.e. generates jackknife subensembles
    At specific order, assuming data is the original, unjackknifed data
    Assumes config is the leftmost index of data
    """
    if order == 1:
        print('consider not doing this seems broken with jackCov')
        jack = np.empty(data.shape)
        for ii in range(0, jack.shape[0]):
            jack[ii, ...] = (np.sum(data, axis=0) - data[ii, ...]) / (data.shape[0] - 1)
    elif order == 2:
        ncon = data.shape[0]
        jack = np.empty([ncon + 1, ncon + 1] + list(data.shape[1:]))
        print(jack.shape)
        jack[0, 0, ...] = doJack(data, 0)    # Mean
        jack[1:, 0, ...] = doJack(data, 1)    # 1st Order Jacks
        jack[0, 1:, ...] = data    # Just the data
        for ii in range(1, ncon+1):
            for jj in range(1, ncon+1):
                jack[ii, jj, ...] = (np.sum(data, axis=0) - data[ii-1] - data[jj-1])/(ncon-2)
    elif order == 0:
        # Just take the average
        jack = np.mean(data, axis=0)
    else:
        sys.exit('higher order jackknfies not implemented yet')

    return jack


def jackCov(jack1: np.ndarray) -> np.ndarray:
    """
    estimates elements of covariance matrix via the jackknife method
    The data is of form [0:ncon,...]
    where 0 is the ensemble average value
    Returns matrix C[ti,]
    """
    ncon = float(jack1.shape[0] - 1)
    return np.cov(jack1, rowvar=False) * (ncon - 1)


def find_nearest(array, value):
    """
    Finds the index of the point in the array which is closest to value
    """
    array = np.asarray(array)
    idx = (np.fabs(array - value)).argmin()
    return idx
