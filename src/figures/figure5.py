"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import resids, R2_Plot_Cells, R2_Plot_Ligs, affFit, doseResponsePlot

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 10), (3, 3))

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    resids(xOptimalMultnoGC, True, False, True)
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)
    modelDFmulnoGC = modelDFmulnoGC.loc[(modelDFmulnoGC.Ligand != "hIL13")]

    R2_Plot_Cells(modelDFmulnoGC, ax[0:2], False, mice=True, training=True)
    R2_Plot_Ligs(modelDFmulnoGC, ax[2], training=True)
    R2_Plot_Cells(modelDFmulnoGC, ax[3:4], False, mice=False, training=False)
    R2_Plot_Ligs(modelDFmulnoGC, ax[4], training=False)
    affFit(ax[5], gcFit=False)
    doseResponsePlot(ax[6:9], modelDFmulnoGC, allCells=False)
    plt.prism()

    return f
