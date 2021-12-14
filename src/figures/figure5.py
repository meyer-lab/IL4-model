"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import resids, residsSeq, doseResponsePlot, R2_Plot_Cells, R2_Plot_RecS, R2_Plot_CV, affDemo, EC50PCA, EmaxPCA

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((12, 6), (2, 5))
    EC50PCA(ax[0:2], True)

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    resids(xOptimalMultnoGC, True, False, True)
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)

    xOptimalSeqnoGC = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    residsSeq(xOptimalSeqnoGC, True, False, True)
    modelDFSeqnoGC = residsSeq(xOptimalSeqnoGC, True, False)

    R2_Plot_Cells(modelDFSeqnoGC, ax[2], seq=True, mice=False, training=False)
    R2_Plot_Cells(modelDFmulnoGC, ax[3], seq=False, mice=False, training=False)
    affDemo(ax[4])
    doseResponsePlot(ax[5:8], modelDFmulnoGC, allCells=False)

    R2_Plot_CV(ax[8])
    R2_Plot_RecS(modelDFmulnoGC, ax[9])
    plt.prism()

    return f
