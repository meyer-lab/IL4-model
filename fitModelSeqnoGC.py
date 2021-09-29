"""
Implementation of a simple multivalent binding model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.MBmodel import residsSeq, fitFuncSeq


xOpt = fitFuncSeq(gcFit=False)

modelDF = residsSeq(xOpt, True, False)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitSeqnoGC.csv")
