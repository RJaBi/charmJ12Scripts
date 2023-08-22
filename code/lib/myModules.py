
from typing import Dict, Any, MutableMapping, Tuple, List
import matplotlib as mpl  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import warnings
import sys
import toml
import unicodedata
import string


def GetArgs(args: list) -> MutableMapping[str, Any]:
    """
    Tests the args for appropriateness,
    loads the parameters in from the parameter file specified in args
    """
    # print(args)
    if len(args) > 1:
        sys.exit('invalid number of arguments presented')
    if 'toml' in args[0]:
        # print('load toml from '+args[0])
        params = toml.load(args[0])
    else:
        sys.exit('Not a toml file '+args[0])
    return params


def GertPlotSettings():
    """
    Updates more plot settings!
    """
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['ytick.labelsize'] = 28
    mpl.rcParams['xtick.labelsize'] = 28
    mpl.rcParams['axes.labelsize'] = 40
    # Increase tick size
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    # Set yticks on both sides, pointing in only
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['ytick.direction'] = 'in'
    # Make minor ticks visible
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    # print('Updated Gert Plot Settings')


def initBigPlotSettings():
    """
    Initialise a bunch of plot settings that make the plots look nicer
    """
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['errorbar.capsize'] = 3  # restoring the caps on error bars
    # print(mpl.rcParams['lines.linewidth'])
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markeredgewidth'] = 0.5
    mpl.rcParams['figure.max_open_warning'] = 50
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.serif'] = ['Computer Modern']
    # FUCKING LATEX
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['lines.markersize'] = 5.0
    mpl.rcParams['figure.figsize'] = (16.6, 11.6)
    # reduce pdf compression (zero is none)
    mpl.rcParams['pdf.compression'] = 1
    # make type 42 types for easier editing in pdf editor
    mpl.rcParams['pdf.fonttype'] = 42
    GertPlotSettings()
    # print('Updated Plot Settings')
    # stop showing all warnings
    # this does not work on LOCATOR warnings :(
    # warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
    warnings.filterwarnings("ignore")


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
                else:
                    v[ii] = float(x)
    # Now that have updated, return.
    return params


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

# #######################################################
# Old jackknife resampling code
# Still works i gues


def doJack(data: np.ndarray, order=2):
    """
    Does jackknife resampling
    i.e. generates jackknife subensembles
    At specific order, assuming data is the original, unjackknifed data
    Assumes config is the leftmost index of data
    """
    if order == 1:
        # print('consider not doing this seems broken with jackCov')
        jack = np.empty(data.shape)
        for ii in range(0, jack.shape[0]):
            jack[ii, ...] = (np.sum(data, axis=0) - data[ii, ...]) / (data.shape[0] - 1)
    elif order == 2:
        ncon = data.shape[0]
        jack = np.empty([ncon + 1, ncon + 1] + list(data.shape[1:]))
        # print(jack.shape)
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
    The data is of form [0:ncon,...]
    where 0 is the ensemble average value
    Returns matrix C[ti,]
    """
    ncon = float(jack1.shape[0] - 1)
    return np.cov(jack1, rowvar=False) * (ncon - 1)


def jackErrNDARRAY(c: np.ndarray):
    """
    Takes jacknife error for a two dimensional numpy array
    The first dimension is the jackknife
    """
    jackErr = np.empty(c.shape[1])
    ncon = float(c.shape[0]) - 1.0
    for xx in range(0, len(jackErr)):
        # avg = c[0,]
        thisXX = c[1:, xx]
        avg = np.sum(thisXX)/(ncon)
        sumTerm = np.sum((thisXX - avg)**2.0)
        err = np.sqrt(sumTerm * (ncon-1)/(ncon))
        jackErr[xx] = err
    # Return
    return jackErr


def find_nearest(array, value):
    """
    Finds the index of the point in the array which is closest to value
    """
    array = np.asarray(array)
    idx = (np.fabs(array - value)).argmin()
    return idx


def getLine_XYYERR(ax, ll: int):
    """
    Gets the x, y, yerr points from the axis for
    the specified line ll
    returns 3 ndarrays and the label
    """
    thisX = ax.lines[ll].get_xdata()
    thisY = ax.lines[ll].get_ydata()
    yErrGot = False
    # Iterate over x,y points
    for nn in range(0, len(ax.lines[ll].get_xdata())):
        thisYErr = np.empty(np.shape(thisX))
        # This is a big messy process of checks
        # to handle the case where we have mixed errorbars and pints
        # and to handle when have same x, y but different errors
        ccDone = []
        for cc in range(0, len(ax.collections)):
            if cc in ccDone:
                # Skipping if have already done it
                continue
            coll = ax.collections[cc]
            if not isinstance(coll, mpl.collections.LineCollection):
                ccDone.append(cc)
                continue
            segs = np.asarray(coll.get_segments())
            if len(segs.shape) != 3:
                continue
            # segs is a list of 2x2 matrices for each x, y point
            # i.e. [npoints, 2, 2]
            # seg[:, 1] is the y points at end of errorbars
            # seg[:, 0] is the x points at centre of error bars
            # hence only need 1 of them
            xSegs = segs[:, 0, 0]
            # check these correspond to correct x points
            # the str comparison is to fix cases where
            # the thisX is a string not a number
            if len(xSegs) != len(thisX):
                ccDone.append(cc)
                continue
            if (xSegs == thisX).all() or type(thisX[0] == str):
                # Now check that the y data is correct
                ySegs = segs[:, :, 1]
                # Iterate over each point to check
                # assumes that error is symmetric
                # which it should be
                for point in range(0, np.shape(segs)[0]):
                    yPoint = thisY[point]
                    # take difference (error), add to lower point
                    yErr = abs(ySegs[point][1] - ySegs[point][0]) / 2
                    ySegPoint = ySegs[point][0] + yErr
                    if yPoint == ySegPoint:
                        # then we have correct ydata too
                        thisYErr[point] = yErr
                        ccDone.append(cc)
                        yErrGot = True
        # So now have the x, y, yerr data
        # now need to try for the label
        # It is not always possible
        # i.e. if some lines do not have legends
        # can not match definitevely
        lenLines = len(ax.lines)
        try:
            lenLeg = len(ax.get_legend().texts)
        except AttributeError:
            # i.e. there is no legend probably
            lenLeg = -5
        if lenLines == lenLeg:
            # can match
            legLabel = ax.get_legend().texts[ll].get_text()
        else:
            legLabel = f'Series_Axis{ll}_Line{nn}'
    if yErrGot:
        return thisX, thisY, thisYErr, legLabel
    else:
        return None, None, None, None


def pdfSaveXY(pdf, fig, allAX, tight=False):
    """
    Saves the x,y, yerr data from the figure/axis and then
    saves the figure to the pdf.
    returns the pdf
    """
    xData = []
    yData = []
    yErrData = []
    labels = []
    if isinstance(allAX, np.ndarray) or isinstance(allAX, tuple):
        axArray = np.asarray(allAX).reshape((-1))
        # print('multi', len(axArray))
        for ax in axArray:
            # Loop over number of lines
            for ll in range(0, len(ax.lines)):
                thisX, thisY, thisYErr, legLabel = getLine_XYYERR(ax, ll)
                if thisX is None:
                    continue
                xData.append(thisX)
                yData.append(thisY)
                yErrData.append(thisYErr)
                labels.append(legLabel)
    else:
        # Loop over number of line
        # print('single', len(allAX.lines))
        for ll in range(0, len(allAX.lines)):
            thisX, thisY, thisYErr, legLabel = getLine_XYYERR(allAX, ll)
            # print(thisX, thisY, thisYErr, legLabel)
            # print(ll, legLabel, thisX)
            if thisX is None:
                continue
            xData.append(thisX)
            yData.append(thisY)
            yErrData.append(thisYErr)
            labels.append(legLabel)
    # Get the page number of the pdf, i.e. the pagecount
    page = pdf.get_pagecount()
    # Get the filename of the pdf
    fh = pdf._file.fh.name
    csvName = fh.replace('.pdf', f'_Page{page}.csv')
    # Cast to numpy arrays
    """
    try:
        xArData = np.asarray(xData)
        yArData = np.asarray(yData)
        yErrArData = np.asarray(yErrData)
        labels = np.asarray(labels)
        # Trim off 1 dimension ranks
        if len(np.shape(xArData)) != 1:
            xData = np.asarray(xArData)[..., 0]
        if len(np.shape(yArData)) != 1:
            yData = np.asarray(yArData)[..., 0]
        if len(np.shape(yErrArData)) != 1:
            yErrData = np.asarray(yErrArData)[..., 0]
        # construct the pandas dataframe
        DF = pd.DataFrame.from_records(np.asarray([xData, yData, yErrData, labels]).T, columns=['xData', 'yData', 'yErrData', 'label'])  # noqa: E501
    except ValueError:
    """
    # print('Not converting to numpy and trimming off 1 dim ranks')
    # This is as the xData is inhomogenous
    # construct the pandas dataframe
    DF = pd.DataFrame.from_records([xData, yData, yErrData, labels]).T
    DF.columns = ['xData', 'yData', 'yErrData', 'label']

    # Save to a csv
    DF.to_csv(csvName, index=False)
    # save the figure
    if tight:
        fig.tight_layout()
    pdf.savefig(fig)
    return pdf
