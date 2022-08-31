"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import resids, residsSeq, doseResponsePlot, R2_Plot_Cells, R2_Plot_RecS, R2_Plot_CV, affDemo, AUC_PCA, respCurvesAll

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((14, 10), (3, 5))

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    modelDFmulnoGC = resids(xOptimalMultnoGC, True, False)

    xOptimalSeqnoGC = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    modelDFSeqnoGC = residsSeq(xOptimalSeqnoGC, True, False)

    xOptimalMult = pd.read_csv("src/data/CurrentFit.csv").x.values
    modelDFmul = resids(xOptimalMult, True, True)

    xOptimalSeq = pd.read_csv("src/data/CurrentFitSeq.csv").x.values
    modelDFSeq = residsSeq(xOptimalSeq, True, True)

    AUC_PCA(ax[0:3], True)

    R2_Plot_Cells(modelDFSeqnoGC, ax[3], seq=True, mice=False, training=False, gcFit=False)
    R2_Plot_Cells(modelDFmulnoGC, ax[4], seq=False, mice=False, training=False, gcFit=False)
    R2_Plot_Cells(modelDFSeq, ax[5], seq=True, mice=False, training=False, gcFit=True)
    R2_Plot_Cells(modelDFmul, ax[6], seq=False, mice=False, training=False, gcFit=True)
    doseResponsePlot(ax[7:10], modelDFmulnoGC, allCells=False)

    R2_Plot_CV(ax[10])
    R2_Plot_RecS(modelDFmulnoGC, ax[11])
    plt.prism()
    respCurvesAll(xOptimalMult, xOptimalMultnoGC, xOptimalSeq , xOptimalSeqnoGC, justPrimary=False)

    return f
