"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .figureCommon import getSetup
from src.MBmodel import resids, R2_Plot_Cells, R2_Plot_Ligs, affFit, ABtest


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 6), (2, 3))

    xOptimalMult = pd.read_csv("src/data/CurrentFit.csv").x.values
    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    modelDFmul = resids(xOptimalMult, True, True)
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)

    R2_Plot_Ligs(modelDFmulnoGC, ax[0])
    R2_Plot_Cells(modelDFmulnoGC, ax[1:3], False)
    R2_Plot_Ligs(modelDFmul, ax[3])
    R2_Plot_Cells(modelDFmul, [ax[4]], False, False)
    affFit(ax[5], gcFit=True)
    plt.prism()

    ax[0].set(title=ax[0].get_title() + "Using Measured Macrophage γc")
    ax[1].set(title=ax[1].get_title() + " Using Measured Macrophage γc")
    ax[3].set(title=ax[3].get_title() + " Using Optimized Macrophage γc")
    ax[4].set(title=ax[4].get_title() + " Using Optimized Macrophage γc")
    ax[5].set(title=ax[5].get_title() + " Using Optimized Macrophage γc")

    return f

# calc at different valencie
