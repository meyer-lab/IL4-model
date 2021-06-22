"""
Implementation of a simple multivalent binding model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.MBmodel import residsSeq, fitFuncSeq, seqBindingModel

KdVec = np.power(10, [-9., 0., 1.])
cell = "Fibroblast"
animal = "Human"
relRecp = 1
print(seqBindingModel(KdVec, np.array([-9.]), cell, animal, relRecp))
KdVec = np.power(10, [-9., 1., 1.])
print(seqBindingModel(KdVec, np.array([-9.]), cell, animal, relRecp))
KdVec = np.power(10, [-5., 1., 1.])
print(seqBindingModel(KdVec, np.array([-9.]), cell, animal, relRecp))


xOpt = fitFuncSeq()

modelDF = residsSeq(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFit.csv")
