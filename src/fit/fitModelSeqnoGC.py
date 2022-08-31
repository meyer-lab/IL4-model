"""
Fit sequential model to data without modifying gc.
"""

import pandas as pd
from sklearn.metrics import r2_score
from src.MBmodel import residsSeq, fitFuncSeq


xOpt = fitFuncSeq(gcFit=False)

modelDF = residsSeq(xOpt, True, False)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitSeqnoGC.csv")
