"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import doseResponsePlot, resids, residsSeq, affFit, affFitSeq, ABtestNorm, affDemo

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 18), (6, 4), multz={21: 1})
    ax[22].axis("off")

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    modelDF = resids(xOptimalMultnoGC, retDF=True, gcFit=False, justPrimary=False)
    doseResponsePlot(ax[0:8], modelDF=modelDF, model="Multivalent Model")
    modelDF.to_csv("MultivalentModelOutput.csv")

    xOptimalSeqnoGC = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    modelDFSeq = residsSeq(xOptimalSeqnoGC, retDF=True, gcFit=False, justPrimary=False)
    doseResponsePlot(ax[8:16], modelDF=modelDFSeq, model="Sequential Model")
    modelDFSeq.to_csv("SequentialModelOutput.csv")

    affFit(ax[16], gcFit=False)
    affDemo(ax[17])
    affFitSeq(ax[18:20], gcFit=False)
    affDemo(ax[20], False)
    ABtestNorm(ax[21], xOptimalSeqnoGC, xOptimalMultnoGC)

    return f
