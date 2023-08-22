# Python

### Here I keep generic python helper functions
### That apply to multiply smaller programs
---
### myModules
*A couple of very generic functions used by virtually every file*
 * GetArgs(args: list) -> MutableMapping[str, Any]
   * * Loads the toml file in the arguments
 * initBigPlotSettings()
   * * Initialise a bunch of plot settings that make the plots look nicer
 * GertPlotSettings()
   * * Initialise a bunch of plot settings that make the plots look nicer after discussion with Gert Aarts
 * refineXYLims(params, subDict: 'analysis') -> Dict[str, Any]
   * * Toml doesn't support None type. This converts any string 'None' value in the params[subDict] dictionary to be None type. just does params if subDict=None
 * removeZero(data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
   * * Removes any occasions where the data is zero
   * * returns the removed and also indices which were kept
 * replaceZero(data: np.ndarray, rep: float)
   * * Similarly but replaces the zero with the float
 * clean\_filename(filename, whitelist=None, replace=' ', char_limit=255):
   * * Removes special characters from the string in filenamexo
 * replace_nth(s, sub, repl, n=1):
   * * Replaces the nth 'sub' by 'repl' in string s
 * doJack(data: np.ndarray, order=2):
   * * Does jackknife resampling
   * * possible weird behaviour for order 1 on their own
 * jackCov(jack1: np.ndarray) -> np.ndarray:
   * * calculates the covariance matrix from 1st order jackknifes
 * jackErrNDARRAY(c: np.ndarray):
   * * Calculates the uncertainty from the jackknif method
   * * agrees with diag**0.5 of above
 * find_nearest(array, value):
   * * finds the index of the point in the array which is closest to value
 * pdfSaveXY(pdf, fig, allAX, tight=False):
   * * Saves the x,y, yerr data from the figure/axis and then
   * * saves the figure to the pdf
   * * returns the pdf
 * getLine_XYYERR(ax, ll: int):
   * * gets the x,y, yerr data from the axis

### myIO
*More specialised code for loading correlation functions*
 * gvarSeed
   * * a seed to use for setting the gvar seed
 * strToLambda(string)
   * * Converts a specifically formatted string to a lambda function
   * * I.e.
   * * * outVar:var1,,,...,_N: f(var1,,,...,_N)where f(...) must be an appropriate maths function
 * sortEvals(w: np.ndarray, vl: np.ndarray, vr: np.ndarray) -> np.ndarray:
   * * Sorts eigenvectors by eigenvalue w (max eigval first)
   * * possibly broken
   * * not currently used as numpy routines return sorted anyway
 * performVarAnalysis(GList: List[np.ndarray], n: int, t0: int, dt: int, src_t0: int) -> List[np.ndarray]
   * * Performs a variational analysis on the correlators in GList
   * * Assumes order is G11, G12, G21, G22, etc
   * * Uses the correlator at t0 + dt to define the reoccurance relation
   * * Normallise G by G = 0.5 * (G + np.conjugate(G))
* performVarAnalysisGvar(GList: List[np.ndarray], n: int, t0: int, dt: int, src_t0: int) -> List[np.ndarray]
   * * Performs a variational analysis on the correlators in GList
   * * Works on gvar variables
   * * Assumes order is G11, G12, G21, G22, etc
   * * Uses the correlator at t0 + dt to define the reoccurance relation
   * * DOES NOT Normallise G by G = 0.5 * (G + np.conjugate(G))
* labelNorm(G: np.ndarray, params):
   * * Specifically for normallising the correlator data [ncon, Nt] at the labelToAnalyse level
   * * Currently only supports by mean at src t0
 * loadSingleMeson(cfFile):
   * * Loads a single meson correlator in gencf format
   * * * i.e. Nt * double precision complex in big endian. Nt=0 first
 * setParityMask(proj: str) -> np.ndarray:
   * * Sets the appropriate parity mask for the 4x4 dirac matrix in column major order
 * loadGencfMomBaryon(cfFile: str, nt: int, numMom: int, proj: str = '+', mom: int = 0):
   * * Loads a single momenta of a gencf baryon with no lorentz indices
   * * A gencf baryon is complex, double precision, big endian, with a 4x4 dirac matrix at each time step, in column major ordering
 * loadOQCDMeson(cfFile: str, gOff: int = 5):
   * * Loads an openqcd meson correlator which does not have momenta
   * * gOff 5 is gamma 5
   * * gOff 2,3,4 are the vector components
 * loadSingleBaryon(cfFile):
   * * loads a single momenta only openqcd-hadspec baryon file
 * initCorrelators(params):
   * * Loads all correlators in params['cfuns'] in
   * * Does the maths, etc
 * preNorm(params, cfDataDict):
   * * Normalllises correlators on config level according to ensemble average values
 * postNorm(params, GVD):
   * * Normallises correlators once converted to gvar format
 * GReconBaryon(N: int, N0: int, GZero: np.ndarray) -> np.ndarray:
   * * constructs the reconstructed correlator at higher temperature for baryons/fermions
 * GReconMeson(N: int, N0: int, GZero: np.ndarray) -> np.ndarray:
   * * constructs the reconstructed correlator at higher temperature for mesons/bosons
 * autoPadLength(N: int, N0: int, fac: int) -> int:
   * * works out the amount of data points to pad with such that
   * * N0/N = fac
 * extCorr(G: np.ndarray, NT: int, extType: str, where: str) -> np.ndarray:
   * * extends correlator G to NT points
   * * padding depending upon extType (zero or min, sub)
   * * padding at 'end', midpoint ('NT2') or minimum of G ('min')
 * initCorrelatorsGvar(params):
   * * Loads all correlators in params['cfuns'] in
   * * converts to gvar
   * * does the maths, etc
   * * See the function for math types
   * * Does not support the 'resample' maths
 * constructBaryonToyFunc(x, p, NT):
   * * constructs a function in lsqfit format to calculate a baryon
   * * p[A], p[E] are alternating pos/neg amplitues/energies
 * constructMesonToyFunc(x, p, NT):
   * * constructs a function in lsqfit format to calculate a baryon
   * * p[A], p[E] are alternating pos/neg amplitues/energies
 * writeGVDCSV(outName: str, GVDD, keys):
   * * Writes all the variables listed in keys to a csv
   * * format tau, lab1 Val, lab1 Err, lab2 Val, lab2 Err...
   * * ASSUMES ALL HAVE SAME LENGTH
 * getNT(GVDD):
   * * Gets the length of the entries in GVDD
 * sepGVDNT(GVDD):
   * * Puts all the dict entries of differening lengths in different dictionaries
 * saveCSV(anaDir: str, suffix: str, GVD, keys=['all']):
   * * Combines the above functions to save a csv for each length (NT)
 * def getFactorPlus(N: int, N0: int, add: int=0) -> int:
   * * returns the number such that (N0 + add) / N is an odd integer
   * * acts recursively, increasing add
   

### myEffE
*Taking an effective mass (energy) of correlators*

 * effE_centre(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
   * * The centre finite difference for effective energy
   * * i.e. an arccosh
 * effE_forward(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
   * * the foward finite difference effective mass
   * * (1.0/massdt) * log(G(t)/G(t+massdt))
 * effE_solve(massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
   * * Solves the meson correlator (cosh) for the mass
 * effE_barAna(GJ2: np.ndarray) -> np.ndarray:
   * * Implements the 4 point effective mass for a baryon with periodic BC
   * * Gets the positive parity solution
 * getEffE(params: Dict[str, Any], massdt: np.float64, GJ2: np.ndarray) -> np.ndarray:
   * * Calls the above function based on the effEMethod in params['analysis']
   * * changes any nans to zeros

### myGVPlotter
*Functions for plotting gvar's using matplotlib*
 * markers
   * * defines a list of marker types for matplotlib
 * colours
   * * defines a list of colours for matplotlib
 * plot_gvEbar(x: np.ndarray, y: np.ndarray, ax, ma=None, ls=None, lab=None, col=None, alpha=1, fillstype='full', mac=None):
   * * Makes plotting an errorbar plot of a gvar easier
 * myHSpan(y: np.ndarray, ax, ls='--', colour=None, lab=None, alpha=1.0):
   * * Horizontal spans of gvars
 * myVSpan(y: np.ndarray, ax, ls='--', colour=None, lab=None, alpha=1.0):
   * * Vertical spans of gvars
 * myFill_between(x: np.ndarray, y: np.ndarray, ax, ma=None, ls=None, lab=None, alpha=0.15, colour=None):
   * * Fill_between for gvar

### myFits
*Functions used in multiple fitting programs*
 * makeFitWins(fitMin: int, fitMax: int, Nt: int, twoSided=True):
   * * Makes all fit windows from fitMin to fitMax
 * plotWinLines(ax, wins: List[str]):
   * * Plots a vertical line where the start of the fit window changes
   * * Assumes wins is sorted
 * getWinTicks(ax, wins: List[str]) -> List[str]:
   * * returns a list of ticklabels where only the start t0 is labelled
 * arctan(x, p):
   * * an arctan fit as suggested in 2007.04188 suitable for lsqfit
 * linFunc(x: np.ndarray, p: Dict[str, Any])
   * * A linear fit function suitable for lsqfit
 * sqrtLinFunc(x: np.ndarray, p: Dict[str, Any])
   * * a sqrt(linear) fit function suitable for lsqfit
   

### myGF
* Functions concerning gaugefield parameter setting and tuning
 * calcNu(xi0_g: float, xi0_q: float) -> float
   * * nu = xi0g/xi0_q
 * calcKappa(m0: float, nu: float, xi0_g: float) -> float
   * * Calculates anisotropic kappa value from m0
 * calcM0(kappa: float, nu: float, xi0_g: float) -> float:
   * * Calculates m0 from anisotropic kappa value
 * makeGVar(params: MutableMapping[str, Any], ana: int, data: np.ndarray) -> gv.gvar:
   * * Transforms the loaded fit parameters into gvar variables
   * * The loaded data was a 2nd order jackknife of the fit param
 * loadData(params: MutableMapping[str, Any], ana: int, y=True) -> Tuple[np.ndarray, np.ndarray]
   * * Loads fit values from pickle files. Full jack subensembles
   * * Breaks for different number of configs for each fit loaded
 * loadData(params: MutableMapping[str, Any], ana: int, y=True) -> Tuple[np.ndarray, np.ndarray]
   * * Loads fit values from pickle files. Uses Full jack subensembles and puts into gvar immediately
 * jackCov(jack1: np.ndarray) -> np.ndarray:
   * * Calculates covariance matrix from jackknife subensembles
   * * identical to that in modules
