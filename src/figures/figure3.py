"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .figureCommon import getSetup
from src.MBmodel import cytBindingModel, resids, residsSeq, R2_Plot_Cells, Exp_Pred, affFit, affFitSeq, ABtest


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((12, 8), (1, 1))
    xOptimalSeq = pd.read_csv("src/data/CurrentFitSeq.csv").x.values
    xOptimalMult = pd.read_csv("src/data/CurrentFit.csv").x.values

    ABtest(ax[0], xOptimalSeq, xOptimalMult)

    return f

# calc at different valencie
