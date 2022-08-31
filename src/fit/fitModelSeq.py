"""
Fit Sequential Model to data.
"""

import pandas as pd
from sklearn.metrics import r2_score
from src.MBmodel import residsSeq, fitFuncSeq


xOpt = fitFuncSeq()

modelDF = residsSeq(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFitSeq.csv")
