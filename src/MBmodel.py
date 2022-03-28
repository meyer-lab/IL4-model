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
import scipy.integrate as integrate
from scipy.optimize import root, least_squares
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler


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


def cytBindingModel(Kx, Cplx, doseVec, cellType, animal, ABblock=1.0, macIL4=False, macGC=False, gcFit=True):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    doseVec = np.array(doseVec)

    if not macIL4:
        if cellType == "Monocyte" and animal == "Human" and gcFit:
            recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                 np.power(10, macGC),
                                 recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
        else:
            recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
    else:
        recCount = np.ravel([np.power(10, macIL4),
                            recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                            recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])

    output = np.zeros([doseVec.size, 2])

    for i, dose in enumerate(doseVec):
        output[i, :] = IL4mod(np.power(10, dose), np.power(10, Kx), recCount, Cplx)

    return output[:, 0] + output[:, 1]


def fitFunc(gcFit=True):
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    x0 = np.array([-11, 8.6, 5, 5, 7.6, 5, 9.08, 5, 5, 8.59, 5, 5, 5, 2, 5])
    bnds = ([-14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2., 3], [-10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 6, 11, 2.7, 7])
    parampredicts = least_squares(resids, x0, bounds=bnds, ftol=1e-5, args=(False, gcFit, False))
    #assert parampredicts.success
    return parampredicts.x


def getConfInterval():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    x0 = np.array(pd.read_csv("src/data/CurrentFit.csv").x)
    bnds = (x0 - 0.0000001, x0 + 0.0000001)
    parampredicts = least_squares(resids, x0, bounds=bnds)
    Hess = np.matmul(parampredicts.jac.T, parampredicts.jac)
    conf = np.linalg.inv(Hess)
    confs = np.sqrt(np.diagonal(conf))
    #assert parampredicts.success
    conf95 = 1.96 * confs
    residSolve = resids(x0)
    sigr = np.linalg.norm(residSolve) / (len(residSolve) - len(x0))
    return conf95 * sigr


def resids(x, retDF=False, gcFit=True, justPrimary=False):
    """"Returns residuals against signaling data"""
    SigData = loadSigData()

    if justPrimary:
        SigData = SigData.loc[(SigData.Cell != "Fibroblast") & (SigData.Cell != "Monocyte")]
        SigData = SigData.loc[(SigData.Animal != "Human") | (SigData.Cell != "Macrophage")]

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
                    if cell == "Monocyte":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, macGC=x[14], gcFit=gcFit)
                    elif cell == "Macrophage":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, macIL4=x[13], gcFit=gcFit)
                    else:
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
                else:
                    results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
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
        print(mean_squared_error(masterSTAT.Experimental.values, masterSTAT.Predicted.values))
        return masterSTAT
    else:
        print(x)
        print(np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values))
        errorCalcDF = copy(masterSTAT)
        for cell in errorCalcDF.Cell.unique():
            for animal in errorCalcDF.loc[errorCalcDF.Cell == cell].Animal.unique():
                cellNum = errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal)].shape[0]
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Predicted"] /= np.sqrt(cellNum)
                errorCalcDF.loc[(errorCalcDF.Cell == cell) & (errorCalcDF.Animal == animal), "Experimental"] /= np.sqrt(cellNum)

        return errorCalcDF.Predicted.values - errorCalcDF.Experimental.values


def seqBindingModel(KdVec, doseVec, cellType, animal, lig, ABblock=1.0, macIL4=False, macGC=False, gcFit=True):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    if not macIL4:
        if cellType == "Monocyte" and animal == "Human" and gcFit:
            recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                 np.power(10, macGC),
                                 recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
        else:
            recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
    else:
        recCount = np.ravel([np.power(10, macIL4),
                            recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                            recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
    output = np.zeros([doseVec.size, 1])
    if lig != "hIL13":
        for i, dose in enumerate(doseVec):
            bnds = ((0, recCount[0]))
            solvedIL4Ra = least_squares(IL4Func, x0=recCount[0], bounds=bnds, args=(KdVec, recCount, dose)).x
            output[i, 0] = SignalingFunc(solvedIL4Ra, KdVec, recCount, dose)
    else:
        for i, dose in enumerate(doseVec):
            bnds = ((0, recCount[2] + 0.01))
            solvedIL13Ra = least_squares(IL13Func, x0=recCount[2], bounds=bnds, args=(KdVec, recCount, dose)).x
            output[i, 0] = SignalingFunc13(solvedIL13Ra, KdVec, recCount, dose)

    return output


def fitFuncSeq(gcFit=True):
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    x0 = np.array([-5, 1, 1, -5, 1, -5, 1, 1, -5, 1, 1, -5, 2, 5])
    bnds = ([-11, -4, -4, -11, -4, -11, -4, -4, -11, -4, -4, -11, -1, 4], [-3, 4, 4, -3, 4, -3, 4, 4, -3, 4, 4, -3, 2.7, 7])
    parampredicts = least_squares(residsSeq, x0, bounds=bnds, ftol=1e-5, args=(False, gcFit, False))
    #assert parampredicts.success
    return parampredicts.x


def getConfIntervalSeq():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    x0 = np.array(pd.read_csv("src/data/CurrentFitSeq.csv").x)
    bnds = (x0 - 0.0000001, x0 + 0.0000001)
    parampredicts = least_squares(residsSeq, x0, bounds=bnds)
    Hess = np.matmul(parampredicts.jac.T, parampredicts.jac)
    conf = np.linalg.inv(Hess)
    confs = np.sqrt(np.diagonal(conf))
    #assert parampredicts.success
    conf95 = 1.96 * confs
    residSolve = residsSeq(x0)
    sigr = np.linalg.norm(residSolve) / (len(residSolve) - len(x0))
    return conf95 * sigr


def residsSeq(x, retDF=False, gcFit=True, justPrimary=False):
    """"Returns residuals against signaling data"""
    SigData = loadSigData()

    if justPrimary:
        SigData = SigData.loc[(SigData.Cell != "Fibroblast") & (SigData.Cell != "Monocyte")]
        SigData = SigData.loc[(SigData.Animal != "Human") | (SigData.Cell != "Macrophage")]

    SigData = SigData.loc[SigData["Antibody"] == False]
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted", "Donor"})
    xPow = np.power(10, x)

    KdDict = {"mIL4": [xPow[0], xPow[1], xPow[2]],
              "mNeo4": [xPow[3], xPow[4], 10000],
              "hIL4": [xPow[5], xPow[6], xPow[7]],
              "hNeo4": [xPow[8], xPow[9], 10000],
              "hIL13": [xPow[10], 10000, xPow[11]]}

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values / 100
                ligKDs = KdDict[ligand]
                if animal == "Human":
                    if cell == "Monocyte":
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, macGC=x[13], gcFit=gcFit)
                    elif cell == "Macrophage":
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, macIL4=x[12], gcFit=gcFit)
                    else:
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                else:
                    results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs, "Animal": animal,
                                                             "Experimental": normSigs, "Predicted": np.ravel(results)}))

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
        print(mean_squared_error(masterSTAT.Experimental.values, masterSTAT.Predicted.values))
        return masterSTAT
    else:
        print(x)
        print(np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values))
        return masterSTAT.Predicted.values - masterSTAT.Experimental.values


def IL4Func(x, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[0] - (x + (x * conc) / KDs[0] + recs[1] / (KDs[0] * KDs[1] / (x * conc) + 1) + recs[2] / ((KDs[0] * KDs[2]) / (x * conc) + 1))


def SignalingFunc(IL4Ra, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[1] / (KDs[0] * KDs[1] / (IL4Ra * conc) + 1) + recs[2] / ((KDs[0] * KDs[2]) / (IL4Ra * conc) + 1)


def IL13Func(x, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[2] - (x + (x * conc) / KDs[2] + recs[1] / (KDs[2] * KDs[1] / (x * conc) + 1) + recs[0] / ((KDs[2] * KDs[0]) / (x * conc) + 1))


def SignalingFunc13(IL13Ra, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[1] / (KDs[2] * KDs[1] / (IL13Ra * conc) + 1) + recs[0] / ((KDs[0] * KDs[2]) / (IL13Ra * conc) + 1)


def affFit(ax, gcFit=True, confInt=np.array([False])):
    """Displays fit affinities for the MB model."""
    colors = {"IL4Rα": "mediumorchid", "gc": "gold", "IL13Rα": "lightseagreen"}

    if gcFit:
        fit = pd.read_csv("src/data/CurrentFit.csv").x.values
    else:
        fit = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values
    fitDict = pd.DataFrame(columns=["Ligand", "Receptor", r"$K_D$"])
    receptorList = ["IL4Rα", "gc", "IL13Rα"]
    xPow = fit * -1
    xPow += 9
    if confInt.any() != False:
        confIntHigh = fit - confInt
        confIntLow = fit + confInt
        confIntHigh *= -1
        confIntLow *= -1
        confIntHigh += 9
        confIntLow += 9

    CplxDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
                "mNeo4": [xPow[4], xPow[5], False],
                "hIL4": [xPow[6], xPow[7], xPow[8]],
                "hNeo4": [xPow[9], xPow[10], False],
                "hIL13": [xPow[11], False, xPow[12]]}

    for lig in CplxDict:
        for i, receptor in enumerate(receptorList):
            KD = CplxDict[lig][i]
            if KD:
                fitDict = fitDict.append(pd.DataFrame({"Ligand": [lig], "Receptor": [receptor], r"$K_D$": [KD]}))
    if confInt.any() != False:
        rec = fitDict.Receptor.values
        ligs = fitDict.Ligand.values
        fitDict = fitDict.append(pd.DataFrame({"Ligand": ligs, "Receptor": rec, r"$K_D$": confIntLow[1:-1]}))
        fitDict = fitDict.append(pd.DataFrame({"Ligand": ligs, "Receptor": rec, r"$K_D$": confIntHigh[1:-1]}))
        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDict, ax=ax, palette=colors)
    else:
        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDict, ax=ax, palette=colors)
    ax.set(ylabel=r"$log_{10}(K_D$ (nM))", ylim=(-2, 6), title="Binding Rates Multivalent")
    ax.legend(prop=dict(size=9), loc="upper right")
    if gcFit:
        fitDict.to_csv("LinData/Multivalent Fit GC Fit.csv")
    if gcFit:
        fitDict.to_csv("LinData/Multivalent Fit no GC Fit.csv")


def affFitSeq(ax, gcFit=True, confInt=np.array([False])):
    """Displays fit affinities for the MB model."""
    colorsLig = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    colorsRec = {"IL4Rα": "mediumorchid", "gc": "gold", "IL13Rα": "lightseagreen"}
    if gcFit:
        fit = pd.read_csv("src/data/CurrentFitSeq.csv").x.values
    else:
        fit = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values
    fitDict = pd.DataFrame(columns=["Ligand", "Receptor", r"$K_D$"])
    receptorList = ["IL4Rα", "gc", "IL13Rα"]
    xPow = fit

    KdDict = {"mIL4": [xPow[0], xPow[1], xPow[2]],
              "mNeo4": [xPow[3], xPow[4], False],
              "hIL4": [xPow[5], xPow[6], xPow[7]],
              "hNeo4": [xPow[8], xPow[9], False],
              "hIL13": [xPow[10], False, xPow[11]]}
    for lig in KdDict:
        for i, receptor in enumerate(receptorList):
            KD = KdDict[lig][i]
            if KD:
                fitDict = fitDict.append(pd.DataFrame({"Ligand": [lig], "Receptor": [receptor], r"$K_D$": [KD]}))

    if confInt.any() != False:
        rec = fitDict.Receptor.values
        ligs = fitDict.Ligand.values
        fitDict = fitDict.append(pd.DataFrame({"Ligand": ligs, "Receptor": rec, r"$K_D$": xPow[0:-1] + confInt[0:-1]}))
        fitDict = fitDict.append(pd.DataFrame({"Ligand": ligs, "Receptor": rec, r"$K_D$": xPow[0:-1] - confInt[0:-1]}))

    fitDictKDNorm = fitDict.loc[(fitDict.Receptor == "IL4Rα") & (fitDict.Ligand != "hIL13")]
    fitDictKDNorm = fitDictKDNorm.append(fitDict.loc[(fitDict.Receptor == "IL13Rα") & (fitDict.Ligand == "hIL13")])
    fitDictKDNorm[r"$K_D$"] += 9
    fitDictKDSurf = fitDict.loc[(fitDict.Receptor != "IL4Rα") & (fitDict.Ligand != "hIL13")]
    fitDictKDSurf = fitDictKDSurf.append(fitDict.loc[(fitDict.Receptor == "IL4Rα") & (fitDict.Ligand == "hIL13")])

    if gcFit:
        sns.barplot(x="Ligand", y=r"$K_D$", data=fitDictKDNorm, ax=ax[0], palette=colorsLig)
        ax[0].set(ylabel=r"Initial association $log_{10}(K_D)$ (nM))", ylim=(-1, 7), title="Surface Binding Rates Sequential")
        fitDictKDNorm.to_csv("LinData/Initial Association Params Sequential GC Fit.csv")

        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDictKDSurf, ax=ax[1], palette=colorsRec)
        ax[1].set(ylabel=r"$log_{10}(K_D)$ (#/cell)", ylim=(-5, 5), title="Receptor Multimerization Rates Sequential")
        fitDictKDSurf.to_csv("LinData/Multimerization ParamsSequential GC Fit.csv")
    else:
        sns.barplot(x="Ligand", y=r"$K_D$", data=fitDictKDNorm, ax=ax[0], palette=colorsLig)
        ax[0].set(ylabel=r"Initial association $log_{10}(K_D)$ (nM))", ylim=(-1, 7), title="Surface Binding Rates Sequential")
        fitDictKDNorm.to_csv("LinData/Initial Association Params Sequential no GC Fit.csv")

        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDictKDSurf, ax=ax[1], palette=colorsRec)
        ax[1].set(ylabel=r"$log_{10}(K_D)$ (#/cell)", ylim=(-5, 5), title="Receptor Multimerization Rates Sequential")
        fitDictKDSurf.to_csv("LinData/Multimerization ParamsSequential no GC Fit.csv")


def Exp_Pred(modelDF, ax, seq=False, Mouse=True):
    """Overall plot of experimental vs. predicted for STAT6 signaling"""
    sns.scatterplot(data=modelDF, x="Experimental", y="Predicted", hue="Ligand", style="Cell", ax=ax)
    ax.set(xlim=(-.1, 1), ylim=(-.1, 1))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    if seq:
        ax.set(title="Sequential Binding Model Human")
    else:
        ax.set(title="Mulivalent Binding Model Human")


def affDemo(ax, MB=True):
    """Overall plot of experimental vs. predicted for STAT6 signaling"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    if MB:
        fit = pd.read_csv("src/data/CurrentFitnoGC.csv").x.values * -1 + 9
        fit = np.power(10, fit)
        affDF = pd.read_csv("src/data/ExpAffinities.csv")
        affDF["Inferred IL4Ra Affinity"] = [fit[6], fit[9], fit[1], fit[4]]
        sns.scatterplot(data=affDF, x="Experimental IL4Ra KD", y="Inferred IL4Ra Affinity", hue="Ligand", style="Ligand", palette=colors, ax=ax)
        ax.set(xlim=(1e-2, 1e2), ylim=(1e-2, 1e2), xscale="log", yscale="log")
        ax.set(xscale="log", yscale="log", title="Multivalent Model")
        affDF.to_csv("LinData/Inferred vs. Experimental Affinities Multivalent.csv")
    else:
        fit = pd.read_csv("src/data/CurrentFitSeqnoGC.csv").x.values + 9
        fit = np.power(10, fit)
        affDF = pd.read_csv("src/data/ExpAffinities.csv")
        affDF["Inferred IL4Ra Affinity"] = [fit[5], fit[8], fit[0], fit[3]]
        sns.scatterplot(data=affDF, x="Experimental IL4Ra KD", y="Inferred IL4Ra Affinity", hue="Ligand", style="Ligand", palette=colors, ax=ax)
        ax.set(xlim=(1e-1, 1e7), ylim=(1e-1, 1e7), xscale="log", yscale="log")
        ax.set(xscale="log", yscale="log", title="Sequential Model")
        affDF.to_csv("LinData/Inferred vs. Experimental Affinities Sequential.csv")


def R2_Plot_Cells(df, ax, seq=False, mice=True, training=True, gcFit=True):
    """Plots all accuracies per cell"""
    accDFh = pd.DataFrame(columns={"Cell Type", "Accuracy"})
    accDFm = pd.DataFrame(columns={"Cell Type", "Accuracy"})
    dfh = df.loc[(df.Animal == "Human")]
    dfm = df.loc[(df.Animal == "Mouse")]
    if gcFit:
        add = " with Gc Fit"
    else:
        add = " without Gc Fit"
    if training:
        dfh = dfh.loc[(dfh.Cell.isin(["A549", "Ramos"]))]
        dfm = dfm.loc[(dfm.Cell.isin(["3T3", "A20", "Macrophage"]))]
        ylabel = "Fitting Accuracy (MSE)"
    else:
        #dfh = dfh.loc[(dfh.Cell.isin(["A549", "Ramos"]) == False)]
        ylabel = "Fitting Accuracy (MSE)"

    for cell in dfh.Cell.unique():
        preds = dfh.loc[(dfh.Cell == cell)].Predicted.values
        exps = dfh.loc[(dfh.Cell == cell)].Experimental.values
        r2 = r2_score(exps, preds)
        accDFh = accDFh.append(pd.DataFrame({"Cell Type": [cell], "Accuracy": [r2]}))

    for cell in dfm.Cell.unique():
        preds = dfm.loc[(dfm.Cell == cell)].Predicted.values
        exps = dfm.loc[(dfm.Cell == cell)].Experimental.values
        r2 = r2_score(exps, preds)
        accDFm = accDFm.append(pd.DataFrame({"Cell Type": [cell], "Accuracy": [r2]}))

    sns.barplot(x="Cell Type", y="Accuracy", data=accDFh, ax=ax, color="k")
    ax.set(ylabel=ylabel, ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    if mice:
        sns.barplot(x="Cell Type", y="Accuracy", data=accDFm, ax=ax[1], color="k")
        ax[1].set(ylabel=ylabel, ylim=(0, 0.2))
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

    if seq:
        ax.set(title="Human Cells - Sequential Binding Model")
        accDFm.to_csv("LinData/Accuracy Mouse Cells Sequential" + add + ".csv")
        accDFh.to_csv("LinData/Accuracy Human Cells Sequential" + add + ".csv")
        if mice:
            ax[1].set(title="Mouse Cells")
    else:
        ax.set(title="Human Cells - Multivalent Binding Model")
        accDFm.to_csv("LinData/Accuracy Mouse Cells Multivalent" + add + ".csv")
        accDFh.to_csv("LinData/Accuracy Human Cells Multivalent" + add + ".csv")
        if mice:
            ax[1].set(title="Mouse Cells")


def R2_Plot_Ligs(df, ax=False, training=False):
    """Plots all accuracies per ligand"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    accDF = pd.DataFrame(columns={"Ligand", "Accuracy"})
    if training:
        df = df.loc[(df.Cell.isin(["A549", "Ramos", "3T3", "A20", "Macrophage"]))]
        df = df.loc[(df.Animal == "Mouse") | (df.Cell != "Macrophage")]
        ylabel = "Fitting Accuracy (MSE)"
    else:
        df = df.loc[(df.Cell.isin(["Monocyte", "Macrophage", "Fibroblast"]))]
        df = df.loc[(df.Animal == "Human")]
        ylabel = "Prediction Accuracy Human Macrophage (MSE)"

    for ligand in df.Ligand.unique():
        preds = df.loc[(df.Ligand == ligand)].Predicted.values
        exps = df.loc[(df.Ligand == ligand)].Experimental.values
        r2 = mean_squared_error(exps, preds)
        accDF = accDF.append(pd.DataFrame({"Ligand": [ligand], "Accuracy": [r2]}))
    if not ax:
        sns.barplot(x="Ligand", y="Accuracy", data=accDF, palette=colors)
        plt.ylabel(ylabel)
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
    else:
        sns.barplot(x="Ligand", y="Accuracy", data=accDF, ax=ax, palette=colors)
        ax.set(ylabel=ylabel, ylim=(0, 0.2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def R2_Plot_Mods(dfMult, dfSeq, ax=False, training=False):
    """Plots all accuracies per ligand"""
    accDF = pd.DataFrame(columns={"Model", "Accuracy (MSE)"})
    if training:
        dfMult = dfMult.loc[(dfMult.Cell.isin(["A549", "Ramos", "3T3", "A20", "Macrophage"]))]
        dfMult = dfMult.loc[(dfMult.Animal == "Mouse") | (dfMult.Cell != "Macrophage")]
        dfSeq = dfSeq.loc[(dfSeq.Cell.isin(["A549", "Ramos", "3T3", "A20", "Macrophage"]))]
        dfSeq = dfSeq.loc[(dfSeq.Animal == "Mouse") | (dfSeq.Cell != "Macrophage")]
        ylabel = "Fitting Accuracy (MSE)"
    else:
        dfMult = dfMult.loc[(dfMult.Cell.isin(["Monocyte", "Macrophage", "Fibroblast"]))]
        dfMult = dfMult.loc[(dfMult.Animal == "Human")]
        dfSeq = dfSeq.loc[(dfSeq.Cell.isin(["Monocyte", "Macrophage", "Fibroblast"]))]
        dfSeq = dfSeq.loc[(dfSeq.Animal == "Human")]
        ylabel = "Prediction Accuracy (MSE)"

    models = [dfMult, dfSeq]
    labels = ["Multivalent Binding Model", "Sequential Binding Model"]
    for i, model in enumerate(models):
        preds = model.Predicted.values
        exps = model.Experimental.values
        MSE = mean_squared_error(exps, preds)
        accDF = accDF.append(pd.DataFrame({"Model": [labels[i]], "Accuracy": [MSE]}))
    if not ax:
        sns.barplot(x="Ligand", y="Accuracy", data=accDF, color="k")
        plt.ylabel(ylabel)
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
    else:
        sns.barplot(x="Model", y="Accuracy", data=accDF, ax=ax, color="k")
        ax.set(ylabel=ylabel, ylim=(0, 0.2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    accDF.to_csv("LinData/ModelComparison_1a.csv")


def R2_Plot_RecS(dfMult, ax=False):
    """Plots all accuracies per ligand"""
    colors = {"IL4Ra": "mediumorchid", "Gamma": "gold", "IL13Ra": "lightseagreen"}
    sensDF = pd.read_csv("src/data/ReceptorFitDF.csv")
    sensDF = sensDF.loc[sensDF.Animal == "Human"]
    sensDF["MSE"] = 100 * (mean_squared_error(dfMult.Experimental, dfMult.Predicted) - sensDF["MSE"]) / mean_squared_error(dfMult.Experimental, dfMult.Predicted)
    sensDF["MSE"] = sensDF["MSE"].clip(lower=0)
    sensDF["Cell"] = sensDF.Animal.values + " " + sensDF.Cell.values
    sns.barplot(data=sensDF, x="Cell", y="MSE", hue="Receptor", palette=colors, ax=ax)
    ax.set(ylabel="% Reduction MSE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    sensDF.to_csv("LinData/Receptor Sensitivity Data.csv")


def R2_Plot_CV(ax=False):
    """Plots all accuracies per ligand"""
    cvDF = pd.read_csv("src/data/CellCV.csv")
    cvDF["Cell"] = cvDF.Animal.values + " " + cvDF.Cell.values
    cvDF = cvDF.loc[cvDF.Animal == "Human"]
    sns.barplot(data=cvDF, x="Cell", y=r"$R^2$", color='k', ax=ax)
    ax.set(ylabel=r"Cross Validation $R^2$", ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    cvDF.to_csv("LinData/Cross Validation Accuracy.csv")


def residsAB(x, blockRatio, gcFit=False):
    """"Returns residuals against signaling data"""
    SigData = loadSigData()
    SigData = SigData.loc[SigData["AB Norm"] == True]
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted", "Antibody"})
    Kx = x[0]
    xPow = np.power(10, x)

    CplxDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
                "mNeo4": [xPow[4], xPow[5], 1e2],
                "hIL4": [xPow[6], xPow[7], xPow[8]],
                "hNeo4": [xPow[9], xPow[10], 1e2],
                "hIL13": [xPow[11], 1e2, xPow[12]]}

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for antibody in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Antibody.unique():
                for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Antibody == antibody)].Ligand.unique():
                    isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand) & (SigData.Antibody == antibody)]
                    Concs = isoData.Concentration.values
                    normSigs = isoData.Signal.values
                    ligCplx = CplxDict[ligand]
                    if antibody:
                        if animal == "Human":
                            if cell == "Macrophage":
                                results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, blockRatio, macIL4=x[13], macGC=x[14], gcFit=gcFit)
                            else:
                                results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, blockRatio, gcFit=gcFit)
                        else:
                            results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, blockRatio, gcFit=gcFit)
                    else:
                        if animal == "Human":
                            if cell == "Macrophage":
                                results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, macIL4=x[13], macGC=x[14], gcFit=gcFit)
                            else:
                                results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
                        else:
                            results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
                    masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                   "Animal": animal, "Experimental": normSigs, "Predicted": results, "Antibody": antibody}))

            # Normalize
            if animal == "Human":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "hIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            if animal == "Mouse":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "mIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()

    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    return masterSTAT


def residsSeqAB(x, blockRatio, gcFit=False):
    """"Returns residuals against signaling data"""
    SigData = loadSigData()
    SigData = SigData.loc[SigData["AB Norm"] == True]
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted", "Antibody"})
    xPow = np.power(10, x)

    KdDict = {"mIL4": [xPow[0], xPow[1], xPow[2]],
              "mNeo4": [xPow[3], xPow[4], 10000],
              "hIL4": [xPow[5], xPow[6], xPow[7]],
              "hNeo4": [xPow[8], xPow[9], 10000],
              "hIL13": [xPow[10], 10000, xPow[11]]}

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for antibody in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Antibody.unique():
                for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Antibody == antibody)].Ligand.unique():
                    isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand) & (SigData.Antibody == antibody)]
                    Concs = isoData.Concentration.values
                    normSigs = isoData.Signal.values / 100
                    ligKDs = KdDict[ligand]
                    if antibody:
                        if animal == "Human":
                            if cell == "Macrophage":
                                results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, blockRatio, macIL4=x[12], macGC=x[13], gcFit=gcFit)
                            else:
                                results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, blockRatio, gcFit=gcFit)
                        else:
                            results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, blockRatio, gcFit=gcFit)
                    else:
                        if animal == "Human":
                            if cell == "Macrophage":
                                results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, macIL4=x[12], macGC=x[13], gcFit=gcFit)
                            else:
                                results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                        else:
                            results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                    masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs, "Animal": animal,
                                                   "Experimental": normSigs, "Predicted": np.ravel(results), "Antibody": antibody}))

            # Normalize
            if animal == "Human":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "hIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            if animal == "Mouse":
                masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Ligand == "mIL4")
                                                                                                                         & (masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()

    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    return masterSTAT


def ABtest(ax, xSeq, xMult):
    """Tests Seq vs MB model for a variety of AB blocking ratios"""
    ABblock = np.linspace(start=0, stop=1, num=51)
    models = ["Sequential", "Multivalent"]
    ABtestDF = pd.DataFrame(columns=("Model", "IL13 Ratio", r"Accuracy ($R^2$)", "Ligand"))
    for model in models:
        for ratio in ABblock:
            if model == "Sequential":
                ABdf = residsSeqAB(xSeq, ratio)
            else:
                ABdf = residsAB(xMult, ratio)
            ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "IL13 Ratio": [1 - ratio],
                                       r"Accuracy ($R^2$)": [mean_squared_error(ABdf.Experimental.values, ABdf.Predicted.values)], "Ligand": "All"}))
            for ligand in ABdf.Ligand.unique():
                ligDF = ABdf.loc[ABdf.Ligand == ligand]
                ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "IL13 Ratio": [1 -
                                                                                          ratio], r"Accuracy ($R^2$)": [mean_squared_error(ligDF.Experimental.values, ligDF.Predicted.values)], "Ligand": ligand}))

    sns.lineplot(data=ABtestDF, x="IL13 Ratio", y=r"Accuracy ($R^2$)", hue="Model", ax=ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))


def ABtestNorm(ax, xSeq, xMult):
    """Tests Seq vs MB model for a variety of AB blocking ratios"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    ABblock = np.linspace(start=0, stop=1, num=51)
    models = ["Sequential", "Multivalent"]
    ABtestDF = pd.DataFrame(columns=("Model", "% Available IL13Rα", "Prediction Accuracy (MSE)", "Ligand"))
    for model in models:
        for ratio in ABblock:
            if model == "Sequential":
                ABdf = residsSeqAB(xSeq, ratio)
            else:
                ABdf = residsAB(xMult, ratio)
            ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "% Available IL13Rα": [100 * (1 - ratio)],
                                       "Prediction Accuracy (MSE)": [mean_squared_error(ABdf.Experimental.values / 100, ABdf.Predicted.values)], "Ligand": "All"}))
            for ligand in ABdf.Ligand.unique():
                ligDF = ABdf.loc[ABdf.Ligand == ligand]
                ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "% Available IL13Rα": [100 * (1 - ratio)],
                                           "Prediction Accuracy (MSE)": [mean_squared_error(ligDF.Experimental.values / 100, ligDF.Predicted.values)], "Ligand": ligand}))

    ABtestDF = ABtestDF.loc[(ABtestDF.Model == "Multivalent") & (ABtestDF.Ligand != "All")]
    sns.lineplot(data=ABtestDF.reset_index(), x="% Available IL13Rα", y="Prediction Accuracy (MSE)", hue="Ligand", style="Ligand", palette=colors, ax=ax)
    ax.set(xlim=(0, 100), ylim=(0, 0.1))
    ABtestDF.to_csv("LinData/Antibody Treated Accuracy Data.csv")


def doseResponsePlot(ax, modelDF, allCells=True, model=False):
    """Plot dose response curves for all cells and ligands"""
    meanDF = modelDF.groupby(["Animal", "Cell", "Ligand", "Concentration"])['Experimental'].mean().reset_index()
    stdDF = modelDF.groupby(["Animal", "Cell", "Ligand", "Concentration"])['Experimental'].std().reset_index()
    colorDict = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    ligOrder = ["hIL4", "hNeo4", "mIL4", "mNeo4", "hIL13"]

    if allCells:
        cells = modelDF.Cell.unique()
        cells = ["Ramos", "A549", "Macrophage", "Monocyte", "Fibroblast", "3T3", "A20", "Macrophage"]
        animals = ["Human", "Human", "Human", "Human", "Human", "Mouse", "Mouse", "Mouse"]
    else:
        cells = ["Ramos", "Macrophage", "Monocyte"]
        animals = ["Human", "Human", "Human"]

    for index, cell in enumerate(cells):
        animal = animals[index]
        isoData = modelDF.loc[(modelDF.Cell == cell) & (modelDF.Animal == animal)]
        means = meanDF.loc[(meanDF.Cell == cell) & (meanDF.Animal == animal)]
        stds = stdDF.loc[(stdDF.Cell == cell) & (stdDF.Animal == animal)]
        if animal == "Human":
            ligOrder = ["hIL4", "hNeo4", "hIL13"]
        else:
            ligOrder = ["mIL4", "mNeo4"]
        sns.lineplot(data=isoData.reset_index(), x="Concentration", y="Predicted", hue="Ligand", label="Predicted", ax=ax[index], hue_order=ligOrder, palette=colorDict)
        #sns.scatterplot(data=isoData, x="Concentration", y="Experimental", hue="Ligand", label="Predicted", ax=ax[index], hue_order=ligOrder, palette=colorDict)
        for j, ligand in enumerate(means.Ligand.values):
            ax[index].scatter(x=means.Concentration.values[j], y=means.Experimental.values[j], color=colorDict[ligand])
            ax[index].errorbar(x=means.Concentration.values[j], y=means.Experimental.values[j], yerr=stds.Experimental.values[j], ls='none', color=colorDict[ligand], elinewidth=2, capthick=1)

        if model:
            ax[index].set(title=animal + " " + cell + " - " + model, ylabel="Normalized pSTAT6 (MFI)", ylim=(-.05, 1.25), xlim=(-14, -5))
        else:
            ax[index].set(title=animal + " " + cell, ylabel="Normalized pSTAT6 (MFI)", ylim=(-.05, 1.25), xlim=(-14, -5))
        handles, labels = ax[index].get_legend_handles_labels()
        if len(isoData.Ligand.unique()) == 3:
            ax[index].legend([handles[0]] + handles[3::], [labels[0]] + labels[3::])
        else:
            ax[index].legend([handles[0]] + handles[2::], [labels[0]] + labels[2::])


def EC50PCA(ax, IL13=True):
    """Plot dose response curves for all cells and ligands"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    if IL13:
        ligands = ["hIL13", "hIL4", "hNeo4"]
    else:
        ligands = ["hIL4", "hNeo4"]
    EC50df = pd.read_csv("src/data/EC50df.csv", na_values=["not tested", "ND"])
    EC50df["EC50"] = np.log10(EC50df["EC50"].values)
    EC50df["Cell Donor"] = EC50df["Cell"] + " " + EC50df["Donor"].astype(str)
    EC50df = EC50df.pivot(index=["Cell", "Cell Donor", "Antibody"], columns="Ligand", values="EC50").reset_index()
    if not IL13:
        EC50df = EC50df.drop("hIL13", axis=1)
    EC50df = EC50df.dropna()
    EC50pca = EC50df[ligands].values
    scaler = StandardScaler()
    EC50pca = scaler.fit_transform(EC50pca)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(EC50pca)
    varExp = pca.explained_variance_ratio_ * 100
    loadings = pca.components_
    scoresDF = pd.DataFrame({"Cell": EC50df.Cell.values, "Antibody": EC50df.Antibody.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1]})
    loadingsDF = pd.DataFrame({"Ligand": ligands, "Component 1": loadings[0, :], "Component 2": loadings[1, :]})

    sns.scatterplot(data=scoresDF, x="Component 1", y="Component 2", hue="Cell", style="Antibody", ax=ax[0])
    ax[0].set(xlim=(-3, 3), ylim=(-3, 3), xlabel="PC1 (" + str(varExp[0])[0:4] + "%)", ylabel="PC2 (" + str(varExp[1])[0:4] + "%)")
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Ligand", style="Ligand", ax=ax[1], palette=colors)
    ax[1].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="PC1 (" + str(varExp[0])[0:4] + "%)", ylabel="PC2 (" + str(varExp[1])[0:4] + "%)")


def EmaxPCA(ax, IL13=True):
    """Plot dose response curves for all cells and ligands"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    if IL13:
        ligands = ["hIL13", "hIL4", "hNeo4"]
    else:
        ligands = ["hIL4", "hNeo4"]
    EmaxDF = pd.read_csv("src/data/EmaxDF.csv", na_values=["not tested", "ND"])
    EmaxDF["Cell Donor"] = EmaxDF["Cell"] + " " + EmaxDF["Donor"].astype(str)
    EmaxDF = EmaxDF.pivot(index=["Cell", "Cell Donor", "Antibody"], columns="Ligand", values="Emax").reset_index()
    if not IL13:
        EmaxDF = EmaxDF.drop("hIL13", axis=1)
    EmaxDF = EmaxDF.dropna()
    EmaxPCA = EmaxDF[ligands].values
    scaler = StandardScaler()
    EmaxPCA = scaler.fit_transform(EmaxPCA)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(EmaxPCA)
    loadings = pca.components_
    scoresDF = pd.DataFrame({"Cell": EmaxDF.Cell.values, "Antibody": EmaxDF.Antibody.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1]})
    loadingsDF = pd.DataFrame({"Ligand": ligands, "Component 1": loadings[0, :], "Component 2": loadings[1, :]})

    sns.scatterplot(data=scoresDF, x="Component 1", y="Component 2", hue="Cell", style="Antibody", ax=ax[0])
    ax[0].set(xlim=(-3, 3), ylim=(-3, 3))
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Ligand", style="Ligand", ax=ax[1], palette=colors)
    ax[1].set(xlim=(-3, 3), ylim=(-3, 3))


def sigmoidFunc(x, EC50, upper, lower):
    """Returns the sigmoid function for a given EC50, max, and min"""
    return lower + (upper - lower) / (1 + np.power(10, (EC50 - x)))


def AUC_PCA(ax, IL13=True):
    """Plot dose response curves for all cells and ligands"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    if IL13:
        ligands = ["hIL13", "hIL4", "hNeo4"]
    else:
        ligands = ["hIL4", "hNeo4"]
    EC50df = pd.read_csv("src/data/EC50df.csv", na_values=["not tested", "ND"])
    EC50df = EC50df.loc[EC50df.Antibody == False]
    EC50df["EC50"] = np.log10(EC50df["EC50"].values)
    EC50df["Cell Donor"] = EC50df["Cell"] + " " + EC50df["Donor"].astype(str)
    sigDF = loadSigData()
    minDose, maxDose = np.amin(sigDF.Concentration.values), np.amax(sigDF.Concentration.values)
    AUC = np.zeros(EC50df.shape[0])
    EC50df = EC50df.reset_index()

    for index, row in EC50df.iterrows():
        if np.isnan(row["EC50"]):
            AUC[index] = np.nan
        elif row["Upper"] != 0:
            EC50, upper, lower = row.EC50, row.Upper, row.Lower
            AUC[index] = np.asarray(integrate.quad(sigmoidFunc, minDose, maxDose, args=(EC50, upper, lower)))[0]

    EC50df["AUC"] = AUC
    AUCdf = EC50df.pivot(index=["Cell", "Cell Donor", "Antibody"], columns="Ligand", values="AUC").reset_index()
    if not IL13:
        AUCdf = AUCdf.drop("hIL13", axis=1)
    AUCdf = AUCdf.dropna()
    AUCpca = AUCdf[ligands].values
    scaler = StandardScaler()
    AUCpca = scaler.fit_transform(AUCpca)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(AUCpca)
    varExp = pca.explained_variance_ratio_ * 100
    loadings = pca.components_
    scoresDF = pd.DataFrame({"Cell": AUCdf.Cell.values, "Antibody": AUCdf.Antibody.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1], "Cell Donor": AUCdf["Cell Donor"].values})
    loadingsDF = pd.DataFrame({"Ligand": ligands, "Component 1": loadings[0, :], "Component 2": loadings[1, :]})

    sns.scatterplot(data=scoresDF, x="Component 1", y="Component 2", hue="Cell", style="Antibody", ax=ax[0])
    ax[0].set(xlim=(-3, 3), ylim=(-3, 3), xlabel="PC1 (" + str(varExp[0])[0:4] + "%)", ylabel="PC2 (" + str(varExp[1])[0:4] + "%)")
    ax[0].legend(loc="lower left")
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Ligand", style="Ligand", ax=ax[1], palette=colors)
    ax[1].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="PC1 (" + str(varExp[0])[0:4] + "%)", ylabel="PC2 (" + str(varExp[1])[0:4] + "%)")

    Ratio = np.zeros(scoresDF.shape[0])

    for index, row in scoresDF.reset_index().iterrows():
        Ratio[index] = EC50df.loc[EC50df["Cell Donor"] == row["Cell Donor"]].Ratio.values[0]
    scoresDF["Ratio"] = Ratio
    sns.scatterplot(data=scoresDF, x="Component 1", y="Ratio", hue="Cell", style="Antibody", ax=ax[2])
    ax[2].set(xlim=(-3, 3), yscale="log", xlabel="PC1", ylabel="Type 1 / Type 2 Bias")

    scoresDF.to_csv("LinData/PCA AUC Scores Data.csv")
    loadingsDF.to_csv("LinData/PCA AUC Loadings Data.csv")


def respCurvesAll(xgc, xnogc, xseqgc, xseqnogc, justPrimary=False):
    """Returns Dose Response DF for all"""
    SigData = loadSigData()
    if justPrimary:
        SigData = SigData.loc[(SigData.Cell != "Fibroblast") & (SigData.Cell != "Monocyte")]
        SigData = SigData.loc[(SigData.Animal != "Human") | (SigData.Cell != "Macrophage")]

    SigData = SigData.loc[SigData["Antibody"] == False]
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTATgc = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Predicted"})
    masterSTATnogc = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Predicted"})
    masterSTATseqgc = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Predicted"})
    masterSTATseqnogc = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Predicted"})
    Kxgc = xgc[0]
    Kxnogc = xnogc[0]

    xPowgc = np.power(10, xgc)
    xPownogc = np.power(10, xnogc)
    xPowSeqgc = np.power(10, xseqgc)
    xPowSeqnogc = np.power(10, xseqnogc)

    CplxDictgc = {"mIL4": [xPowgc[1], xPowgc[2], xPowgc[3]],
                "mNeo4": [xPowgc[4], xPowgc[5], 1e2],
                "hIL4": [xPowgc[6], xPowgc[7], xPowgc[8]],
                "hNeo4": [xPowgc[9], xPowgc[10], 1e2],
                "hIL13": [xPowgc[11], 1e2, xPowgc[12]]}
    CplxDictnogc = {"mIL4": [xPownogc[1], xPowgc[2], xPowgc[3]],
                "mNeo4": [xPownogc[4], xPownogc[5], 1e2],
                "hIL4": [xPownogc[6], xPownogc[7], xPownogc[8]],
                "hNeo4": [xPownogc[9], xPownogc[10], 1e2],
                "hIL13": [xPownogc[11], 1e2, xPownogc[12]]}
    KdDictSeqgc = {"mIL4": [xPowSeqgc[0], xPowSeqgc[1], xPowSeqgc[2]],
              "mNeo4": [xPowSeqgc[3], xPowSeqgc[4], 10000],
              "hIL4": [xPowSeqgc[5], xPowSeqgc[6], xPowSeqgc[7]],
              "hNeo4": [xPowSeqgc[8], xPowSeqgc[9], 10000],
              "hIL13": [xPowSeqgc[10], 10000, xPowSeqgc[11]]}
    KdDictSeqnogc = {"mIL4": [xPowSeqnogc[0], xPowSeqnogc[1], xPowSeqnogc[2]],
              "mNeo4": [xPowSeqnogc[3], xPowSeqnogc[4], 10000],
              "hIL4": [xPowSeqnogc[5], xPowSeqnogc[6], xPowSeqnogc[7]],
              "hNeo4": [xPowSeqnogc[8], xPowSeqnogc[9], 10000],
              "hIL13": [xPowSeqnogc[10], 10000, xPowSeqnogc[11]]}



    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = np.linspace(np.amin(isoData.Concentration.values), np.amax(isoData.Concentration.values), 300)

                ligCplxgc = CplxDictgc[ligand]
                ligCplxnogc = CplxDictnogc[ligand]
                ligKDsgc = KdDictSeqgc[ligand]
                ligKDsnogc = KdDictSeqnogc[ligand]
                if animal == "Human":
                    if cell == "Monocyte":
                        resultsgc = cytBindingModel(Kxgc, ligCplxgc, Concs, cell, animal, macGC=xgc[14], gcFit=True)
                        resultsnogc = cytBindingModel(Kxnogc, ligCplxnogc, Concs, cell, animal, macGC=xgc[14], gcFit=False)
                        resultsseqgc = seqBindingModel(ligKDsgc, Concs, cell, animal, ligand, macGC=xseqgc[13], gcFit=True)
                        resultsseqnogc = seqBindingModel(ligKDsnogc, Concs, cell, animal, ligand, macGC=xseqnogc[13], gcFit=False)
                    elif cell == "Macrophage":
                        resultsgc = cytBindingModel(Kxgc, ligCplxgc, Concs, cell, animal, macIL4=xgc[13], gcFit=True)
                        resultsnogc = cytBindingModel(Kxnogc, ligCplxnogc, Concs, cell, animal, macIL4=xnogc[13], gcFit=False)
                        resultsseqgc = seqBindingModel(ligKDsgc, Concs, cell, animal, ligand, macIL4=xseqgc[12], gcFit=True)
                        resultsseqnogc = seqBindingModel(ligKDsnogc, Concs, cell, animal, ligand, macIL4=xseqnogc[12], gcFit=False)
                    else:
                        resultsgc = cytBindingModel(Kxgc, ligCplxgc, Concs, cell, animal, gcFit=True)
                        resultsnogc = cytBindingModel(Kxnogc, ligCplxnogc, Concs, cell, animal, gcFit=False)
                        resultsseqgc = seqBindingModel(ligKDsgc, Concs, cell, animal, ligand, gcFit=True)
                        resultsseqnogc = seqBindingModel(ligKDsnogc, Concs, cell, animal, ligand, gcFit=False)
                else:
                    resultsgc = cytBindingModel(Kxgc, ligCplxgc, Concs, cell, animal, gcFit=True)
                    resultsnogc = cytBindingModel(Kxnogc, ligCplxnogc, Concs, cell, animal, gcFit=False)
                    resultsseqgc = seqBindingModel(ligKDsgc, Concs, cell, animal, ligand, gcFit=True)
                    resultsseqnogc = seqBindingModel(ligKDsnogc, Concs, cell, animal, ligand, gcFit=False)
                
                masterSTATgc = masterSTATgc.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                             "Animal": animal, "Predicted": resultsgc}))
                masterSTATnogc = masterSTATnogc.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                             "Animal": animal, "Predicted": resultsnogc}))
                masterSTATseqgc = masterSTATseqgc.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                             "Animal": animal, "Predicted": np.ravel(resultsseqgc)}))
                masterSTATseqnogc = masterSTATseqnogc.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                             "Animal": animal, "Predicted": np.ravel(resultsseqnogc)}))                                             

            # Normalize
            if animal == "Human":
                masterSTATgc.loc[(masterSTATgc.Cell == cell) & (masterSTATgc.Animal == animal), "Predicted"] /= masterSTATgc.loc[(masterSTATgc.Ligand == "hIL4")
                                                                                                                         & (masterSTATgc.Cell == cell) & (masterSTATgc.Animal == animal)].Predicted.max()
                masterSTATnogc.loc[(masterSTATnogc.Cell == cell) & (masterSTATnogc.Animal == animal), "Predicted"] /= masterSTATnogc.loc[(masterSTATnogc.Ligand == "hIL4")
                                                                                                                         & (masterSTATnogc.Cell == cell) & (masterSTATnogc.Animal == animal)].Predicted.max()
                masterSTATseqgc.loc[(masterSTATseqgc.Cell == cell) & (masterSTATseqgc.Animal == animal), "Predicted"] /= masterSTATseqgc.loc[(masterSTATseqgc.Ligand == "hIL4")
                                                                                                                         & (masterSTATseqgc.Cell == cell) & (masterSTATseqgc.Animal == animal)].Predicted.max()
                masterSTATseqnogc.loc[(masterSTATseqnogc.Cell == cell) & (masterSTATseqnogc.Animal == animal), "Predicted"] /= masterSTATseqnogc.loc[(masterSTATseqnogc.Ligand == "hIL4")
                                                                                                                         & (masterSTATseqnogc.Cell == cell) & (masterSTATseqnogc.Animal == animal)].Predicted.max()
            if animal == "Mouse":
                masterSTATgc.loc[(masterSTATgc.Cell == cell) & (masterSTATgc.Animal == animal), "Predicted"] /= masterSTATgc.loc[(masterSTATgc.Ligand == "mIL4")
                                                                                                                         & (masterSTATgc.Cell == cell) & (masterSTATgc.Animal == animal)].Predicted.max()
                masterSTATnogc.loc[(masterSTATnogc.Cell == cell) & (masterSTATnogc.Animal == animal), "Predicted"] /= masterSTATnogc.loc[(masterSTATnogc.Ligand == "mIL4")
                                                                                                                         & (masterSTATnogc.Cell == cell) & (masterSTATnogc.Animal == animal)].Predicted.max()
                masterSTATseqgc.loc[(masterSTATseqgc.Cell == cell) & (masterSTATseqgc.Animal == animal), "Predicted"] /= masterSTATseqgc.loc[(masterSTATseqgc.Ligand == "mIL4")
                                                                                                                         & (masterSTATseqgc.Cell == cell) & (masterSTATseqgc.Animal == animal)].Predicted.max()
                masterSTATseqnogc.loc[(masterSTATseqnogc.Cell == cell) & (masterSTATseqnogc.Animal == animal), "Predicted"] /= masterSTATseqnogc.loc[(masterSTATseqnogc.Ligand == "mIL4")
                                                                                                                         & (masterSTATseqnogc.Cell == cell) & (masterSTATseqnogc.Animal == animal)].Predicted.max()
    masterSTATgc = masterSTATgc.fillna(0)
    masterSTATgc.replace([np.inf, -np.inf], 0, inplace=True)

    masterSTATnogc = masterSTATnogc.fillna(0)
    masterSTATnogc.replace([np.inf, -np.inf], 0, inplace=True)

    masterSTATseqgc = masterSTATseqgc.fillna(0)
    masterSTATseqgc.replace([np.inf, -np.inf], 0, inplace=True)

    masterSTATseqnogc = masterSTATseqnogc.fillna(0)
    masterSTATseqnogc.replace([np.inf, -np.inf], 0, inplace=True)

    masterSTATgc.to_csv("LinData/Multivalent Dose Curves Monocyte GC Adjusted.csv")
    masterSTATnogc.to_csv("LinData/Multivalent Dose Curves not GC Adjusted.csv")
    masterSTATseqgc.to_csv("LinData/Sequential Dose Curves Monocyte GC Adjusted.csv")
    masterSTATseqnogc.to_csv("LinData/Sequential Dose Curves not GC Adjusted.csv")

    return masterSTATgc
