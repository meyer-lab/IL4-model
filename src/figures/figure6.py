"""
This creates Figure 5, main figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .figureCommon import getSetup
from src.MBmodel import doseResponsePlot, resids

rcParams['svg.fonttype'] = 'none'


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 6), (2, 4))

    xOptimalMultnoGC = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    modelDF = resids(xOptimalMultnoGC, retDF=True, gcFit=False, justPrimary=False)

    doseResponsePlot(ax, modelDF=modelDF)

    return f
