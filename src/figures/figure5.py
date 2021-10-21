"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import resids, residsSeq, doseResponsePlot, R2_Plot_Mods

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 4), (1, 4))

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    resids(xOptimalMultnoGC, True, False, True)
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)
    modelDFmulnoGCnoIL13 = modelDFmulnoGC.loc[(modelDFmulnoGC.Ligand != "hIL13")]

    xOptimalSeqnoGC = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    residsSeq(xOptimalSeqnoGC, True, False, True)
    modelDFSeqnoGC = residsSeq(xOptimalSeqnoGC, True, False)

    doseResponsePlot(ax[0:3], modelDFmulnoGCnoIL13, allCells=False)
    R2_Plot_Mods(modelDFmulnoGC, modelDFSeqnoGC, ax=ax[3], training=False)
    plt.prism()

    return f
