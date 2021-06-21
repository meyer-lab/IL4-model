"""
Implementation of a simple multivalent binding model.
"""

import pandas as pd
from sklearn.metrics import r2_score
from src.MBmodel import resids, fitFuncSeq


xOpt = fitFuncSeq()

modelDF = resids(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFit.csv")