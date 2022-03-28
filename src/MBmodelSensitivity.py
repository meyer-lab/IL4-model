"""
Implementation of a simple multivalent binding model.
"""

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from copy import copy
from scipy.optimize import root, least_squares
from sklearn.metrics import mean_squared_error, r2_score


def loadSigData():
    """Load Signaling Data"""
    sigData = pd.read_csv(join(path_here, "src/data/SignalingData.csv"))
    T3data = sigData.loc[(sigData.Cell == "3T3") & (sigData.Ligand == "mNeo4")]
    SubtractLine = np.polyfit(T3data.Concentration.values, T3data.Signal.values, 1)
    sigData.loc[(sigData.Cell == "3T3") & (sigData.Ligand == "mNeo4"), "Signal"] /= SubtractLine[0]
    sigData.loc[(sigData.Cell == "3T3") & (sigData.Ligand == "mNeo4"), "Signal"] -= SubtractLine[1]
    sigData.loc[(sigData.Cell == "3T3") & (sigData.Ligand == "mIL4") & (sigData.Concentration <= -10), "Signal"] = 0
    return sigData


def Req_func2(Req: np.ndarray, L0, KxStar, Rtot: np.ndarray, Kav: np.ndarray):
    Psi = Req * Kav * KxStar
    Psi = np.pad(Psi, ((0, 0), (0, 1)), constant_values=1)
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * np.sum(Psinorm, axis=0) * np.prod(Psirs)
    return Req + Rbound - Rtot


def IL4mod(L0, KxStar, Rtot, Kav_in):
    """
    IL-4 model
    Rtot = [IL4Ra, IL13Ra, gc] abundance
    Kav_in = [IL4Ra, IL13Ra, gc] affinity to IL4
    Output: signal 1 (IL4Ra-gc) and signal 2 (IL4Ra-IL13Ra)
    """
    # Restructure data
    Kav = np.zeros((2, 3))
    Kav[0, 0] = Kav_in[0]
    Kav[1, 1] = Kav_in[1]
    Kav[1, 2] = Kav_in[2]
    Rtot = np.array(Rtot, dtype=float)
    assert Rtot.size == 3

    # Solve Req
    lsq = root(Req_func2, Rtot, method="lm", args=(L0, KxStar, Rtot, Kav), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req * Kav * KxStar

    IL4_13 = L0 / KxStar * Psi[0, 0] * Psi[1, 1]
    IL4_gc = L0 / KxStar * Psi[0, 0] * Psi[1, 2]

    return IL4_gc, IL4_13


path_here = pathlib.Path().absolute()
recQuantDF = pd.read_csv(join(path_here, "src/data/RecQuantDonor.csv"))


def cytBindingModel(Kx, Cplx, doseVec, cellType, animal, recOpt, recCellAn, macIL4=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    doseVec = np.array(doseVec)
    recQuantDFS = pd.read_csv(join(path_here, "src/data/RecQuantDonor.csv"))
    recQuantDFS.loc[(recQuantDFS.Receptor == recCellAn[0]) & (recQuantDFS["Cell"] == recCellAn[1]) & (recQuantDFS["Animal"] == recCellAn[2]), "Amount"] = np.power(10, recOpt)

    if not macIL4:
        recCount = np.ravel([recQuantDFS.loc[(recQuantDFS.Receptor == "IL4Ra") & (recQuantDFS["Cell"] == cellType) & (recQuantDFS["Animal"] == animal)].Amount.mean(),
                             recQuantDFS.loc[(recQuantDFS.Receptor == "Gamma") & (recQuantDFS["Cell"] == cellType) & (recQuantDFS["Animal"] == animal)].Amount.mean(),
                             recQuantDFS.loc[(recQuantDFS.Receptor == "IL13Ra") & (recQuantDFS["Cell"] == cellType) & (recQuantDFS["Animal"] == animal)].Amount.mean()])
    else:
        recCount = np.ravel([np.power(10, macIL4),
                            recQuantDFS.loc[(recQuantDFS.Receptor == "Gamma") & (recQuantDFS["Cell"] == cellType) & (recQuantDFS["Animal"] == animal)].Amount.mean(),
                            recQuantDFS.loc[(recQuantDFS.Receptor == "IL13Ra") & (recQuantDFS["Cell"] == cellType) & (recQuantDFS["Animal"] == animal)].Amount.mean()])

    output = np.zeros([doseVec.size, 2])

    for i, dose in enumerate(doseVec):
        output[i, :] = IL4mod(np.power(10, dose), np.power(10, Kx), recCount, Cplx)

    return output[:, 0] + output[:, 1]


def fitFunc(gcFit=True):
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    resDF = pd.DataFrame()
    parampredicts = False
    for cell in recQuantDF.Cell.unique():
        for animal in recQuantDF.loc[recQuantDF.Cell == cell].Animal.unique():
            for receptor in recQuantDF.loc[(recQuantDF.Cell == cell) & (recQuantDF.Animal == animal)].Receptor.unique():
                if cell != "Macrophage" or animal != "Human" or receptor != "IL4Ra":
                    if cell != "Ramos" or animal != "Human" or receptor != "IL13Ra":
                        init = recQuantDF.loc[(recQuantDF.Receptor == receptor) & (recQuantDF["Cell"] == cell) & (recQuantDF["Animal"] == animal)].Amount.mean()
                        recCellAn = [receptor, cell, animal]
                        print(recCellAn)
                        if not parampredicts:
                            x0 = np.array([-11, 8.6, 5, 5, 7.6, 5, 9.08, 5, 5, 8.59, 5, 5, 5, 2, np.log10(init)])
                        else:
                            if cell == "Monocyte" and animal == "Human" and receptor == "IL13Ra":
                                x0 = parampredicts.x
                                x0[-1] = np.log10(init) - 2
                            else:
                                x0 = parampredicts.x
                                x0[-1] = np.log10(init)
                        bnds = ([-14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2., np.log10(init) - 2], [-10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 6, 11, 2.7, np.log10(init) + 2])
                        parampredicts = least_squares(resids, x0, bounds=bnds, ftol=1e-5, args=(recCellAn, False))
                        print(parampredicts.x)
                        resultsDF = resids(parampredicts.x, retDF=True, recCellAn=recCellAn)
                        MSE = mean_squared_error(resultsDF.Experimental, resultsDF.Predicted)
                        R2 = r2_score(resultsDF.Experimental, resultsDF.Predicted)
                        print("MSE = ", MSE)
                        print("R2 = ", R2)
                        resDF = resDF.append(pd.DataFrame({"Cell": [cell], "Animal": [animal], "Receptor": [receptor], "MSE": [MSE], r"$R^2$": [R2]}))
    resDF.to_csv("ReceptorFitDF2.csv")


def resids(x, recCellAn=False, retDF=False):
    """"Returns residuals against signaling data"""
    SigData = loadSigData()

    SigData = SigData.loc[SigData["Antibody"] == False]
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    Kx = x[0]
    xPow = np.power(10, x)

    CplxDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
                "mNeo4": [xPow[4], xPow[5], 1e2],
                "hIL4": [xPow[6], xPow[7], xPow[8]],
                "hNeo4": [xPow[9], xPow[10], 1e2],
                "hIL13": [xPow[11], 1e2, xPow[12]]}

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values / 100
                ligCplx = CplxDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, recOpt=x[14], recCellAn=recCellAn, macIL4=x[13])
                    else:
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, recOpt=x[14], recCellAn=recCellAn)
                else:
                    results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, recOpt=x[14], recCellAn=recCellAn)
                masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                             "Animal": animal, "Experimental": normSigs, "Predicted": results}))

            # Normalize
            if animal == "Human":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "hIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            if animal == "Mouse":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "mIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()

    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    if retDF:
        errorCalcDF = copy(masterSTAT)
        """
        for cell in errorCalcDF.Cell.unique():
            for animal in errorCalcDF.loc[errorCalcDF.Cell == cell].Animal.unique():
                cellNum = errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal)].shape[0]
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Predicted"] /= np.sqrt(cellNum)
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Experimental"] /= np.sqrt(cellNum)
        """
        return errorCalcDF
    else:
        errorCalcDF = copy(masterSTAT)
        for cell in errorCalcDF.Cell.unique():
            for animal in errorCalcDF.loc[errorCalcDF.Cell == cell].Animal.unique():
                cellNum = errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal)].shape[0]
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Predicted"] /= np.sqrt(cellNum)
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Experimental"] /= np.sqrt(cellNum)
        return errorCalcDF.Predicted.values - errorCalcDF.Experimental.values
