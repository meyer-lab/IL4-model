"""
Implementation of a simple multivalent binding model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.MBmodel import resids, fitFunc


xOpt = fitFunc(gcFit=False)

modelDF = resids(xOpt, True, False)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitnoGC.csv")
