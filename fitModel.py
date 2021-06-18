"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from src.MBmodel import cytBindingModel, SigData

def fitFunc():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([-11, 0.5, 8.6, 8, 8.01, 7.6, 7, 7.01, 9.08, 8, 8.01, 8.59, 7, 7.01])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ((-14, -10), (0, 10), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11), (4, 11))
    parampredicts = minimize(resids, x0, method="L-BFGS-B", bounds=bnds, options={"disp": 999})
    #parampredicts = minimize(resids, x0, method="trust-constr", bounds=bnds, options={"disp": 999}, constraints=cons)
    assert parampredicts.success
    return parampredicts.x

def resids(x, retDF=False):
    """"Returns residuals against signaling data"""
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    Kx = x[0]
    relRecp = x[1]
    xPow = np.power(10, x)

    CplxDict = {"mIL4": [xPow[2], xPow[3], xPow[4]],
    "mNeo4": [xPow[5], xPow[6], xPow[7]],
    "hIL4": [xPow[8], xPow[9], xPow[10]],
    "hNeo4": [xPow[11], xPow[12], xPow[13]]}
    if retDF:
        SigDataFilt = SigData.loc[(SigData.Cell != "Macrophage") | (SigData.Animal != "Human")]
    if not retDF:
        SigDataFilt = SigData.loc[(SigData.Cell != "Macrophage") | (SigData.Animal != "Human")]
    for cell in SigDataFilt.Cell.unique():
        for animal in SigDataFilt.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigDataFilt.loc[(SigDataFilt.Cell == cell) & (SigDataFilt.Animal == animal)].Ligand.unique():
                isoData = SigDataFilt.loc[(SigDataFilt.Cell == cell) & (SigDataFilt.Animal == animal) & (SigDataFilt.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligCplx = CplxDict[ligand]
                if animal == "Human":
                    results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, relRecp)
                else:
                    results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, relRecp)
                masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs, "Animal": animal, "Experimental": normSigs, "Predicted": results}))
            
            # Normalize
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Experimental"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Experimental.max()
    
    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    if retDF:
        return masterSTAT
    else:
        print(x)
        print(np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values))
        return np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values)

xOpt = fitFunc()

modelDF = resids(xOpt, True)
print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
pd.DataFrame({"x": xOpt}).to_csv("src/data/CurrentFit.csv")