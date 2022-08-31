"""
Fits all model variants simultaneously.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.MBmodel import resids, residsSeq, fitFunc, fitFuncSeq


xOpt = fitFunc()
modelDF = resids(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFit.csv")


xOpt = fitFunc(gcFit=False)
modelDF = resids(xOpt, True, False)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitnoGC.csv")

xOpt = fitFuncSeq()
modelDF = residsSeq(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitSeq.csv")

xOpt = fitFuncSeq(gcFit=False)
modelDF = residsSeq(xOpt, True, False)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitSeqnoGC.csv")
