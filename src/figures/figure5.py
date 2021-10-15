"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import resids, R2_Plot_Cells, affFit

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((6, 6), (2, 2))

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    resids(xOptimalMultnoGC, True, False, True)
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)

    R2_Plot_Cells(modelDFmulnoGC, ax[0:2], False, mice=True, training=True)
    R2_Plot_Cells(modelDFmulnoGC, ax[2:3], False, mice=False, training=False)

    affFit(ax[3], gcFit=False)
    plt.prism()

    return f

# calc at different valencie
