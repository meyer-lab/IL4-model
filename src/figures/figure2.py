"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .figureCommon import getSetup
from src.MBmodel import cytBindingModel, resids, residsSeq, R2_Plot_Cells, Exp_Pred, affFit, affFitSeq


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((12, 8), (3, 3))
    xOptimalSeq = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    modelDFseq = residsSeq(xOptimalSeq, True, False)
    R2_Plot_Cells(modelDFseq, ax[0:2], True)
    Exp_Pred(modelDFseq, ax[2], True)

    xOptimalMult = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    modelDFmul = resids(xOptimalMult, True, False)
    R2_Plot_Cells(modelDFmul, ax[3:5], False)
    Exp_Pred(modelDFmul, ax[5], False)

    affFitSeq(ax[6:8])
    affFit(ax[8])

    return f

# calc at different valencie
