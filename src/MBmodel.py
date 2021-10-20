"""
Implementation of a simple multivalent binding model.
"""

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from scipy.optimize import root, least_squares
from sklearn.metrics import mean_squared_error


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
        recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                             recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                             recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
    else:
        if gcFit:
            recCount = np.ravel([np.power(10, macIL4),
                                np.power(10, macGC),
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
    bnds = ([-14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2., 4], [-10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 6, 11, 2.7, 7])
    parampredicts = least_squares(resids, x0, bounds=bnds, ftol=1e-5, args=(False, gcFit, True))
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

    SigData = SigData.loc[SigData["AB Norm"] == False]
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
            for donor in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Donor.unique():
                for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Donor == donor)].Ligand.unique():
                    isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                    Concs = isoData.Concentration.values
                    normSigs = isoData.Signal.values / 100
                    ligCplx = CplxDict[ligand]
                    if animal == "Human":
                        if cell == "Macrophage":
                            results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, macIL4=x[13], macGC=x[14], gcFit=gcFit)
                        else:
                            results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
                    else:
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, gcFit=gcFit)
                    masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs,
                                                   "Animal": animal, "Experimental": normSigs, "Predicted": results, "Donor": donor}))

            # Normalize
            """
            CellLigData = masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)]
            scaleFact = np.linalg.lstsq(np.reshape(CellLigData.Predicted.values, (-1, 1)), np.reshape(CellLigData.Experimental.values, (-1, 1)), rcond=None)[0][0]
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] *= scaleFact
            """
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()

    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    if retDF:
        print(mean_squared_error(masterSTAT.Experimental.values, masterSTAT.Predicted.values))
        return masterSTAT
    else:
        print(x)
        print(np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values))
        return masterSTAT.Predicted.values - masterSTAT.Experimental.values


def seqBindingModel(KdVec, doseVec, cellType, animal, lig, ABblock=1.0, macIL4=False, macGC=False, gcFit=True):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    if not macIL4:
        recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                             recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean(),
                             recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.mean() * ABblock])
    else:
        if gcFit:
            recCount = np.ravel([np.power(10, macIL4),
                                np.power(10, macGC),
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
    # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    x0 = np.array([-5, 1, 1, -5, 1, -5, 1, 1, -5, 1, 1, -5, 2, 5])
    bnds = ([-11, -4, -4, -11, -4, -11, -4, -4, -11, -4, -4, -11, -1, 4], [-3, 4, 4, -3, 4, -3, 4, 4, -3, 4, 4, -3, 2.7, 7])
    parampredicts = least_squares(residsSeq, x0, bounds=bnds, ftol=1e-5, args=(False, gcFit, True))
    #assert parampredicts.success
    return parampredicts.x


def getConfIntervalSeq():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
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
        SigData = SigData.loc[(SigData.Animal != "Human") & (SigData.Cell != "Macrophage")]

    SigData = SigData.loc[SigData["AB Norm"] == False]
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
            for donor in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Donor.unique():
                for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Donor == donor)].Ligand.unique():
                    isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                    Concs = isoData.Concentration.values
                    normSigs = isoData.Signal.values
                    ligKDs = KdDict[ligand]
                    if animal == "Human":
                        if cell == "Macrophage":
                            results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, macIL4=x[12], macGC=x[13], gcFit=gcFit)
                        else:
                            results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                    else:
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, gcFit=gcFit)
                    masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs, "Animal": animal,
                                                   "Experimental": normSigs, "Predicted": np.ravel(results), "Donor": donor}))

            # Normalize
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Experimental"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Experimental.max()

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
    ax.set(ylabel=r"$log_{10}(K_D$ (nM))", ylim=(-2, 6), title="Fit Affinities")
    ax.legend(prop=dict(size=9), loc="upper right")


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

    sns.barplot(x="Ligand", y=r"$K_D$", data=fitDictKDNorm, ax=ax[0], palette=colorsLig)
    ax[0].set(ylabel=r"IL4Rα $log_{10}(K_D)$ (nM))", ylim=(-1, 7), title="Surface Binding Rates Sequential")

    sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDictKDSurf, ax=ax[1], palette=colorsRec)
    ax[1].set(ylabel=r"$log_{10}(K_D)$ (#/cell)", ylim=(-5, 5), title="Receptor Multimerization Rates Sequential")


def Exp_Pred(modelDF, ax, seq=False, Mouse=True):
    """Overall plot of experimental vs. predicted for STAT6 signaling"""
    sns.scatterplot(data=modelDF, x="Experimental", y="Predicted", hue="Ligand", style="Cell", ax=ax)
    ax.set(xlim=(-.1, 1), ylim=(-.1, 1))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    if seq:
        ax.set(title="Sequential Binding Model Human")
    else:
        ax.set(title="Mulivalent Binding Model Human")


def R2_Plot_Cells(df, ax, seq=False, mice=True, training=True):
    """Plots all accuracies per cell"""
    accDFh = pd.DataFrame(columns={"Cell Type", "Accuracy"})
    accDFm = pd.DataFrame(columns={"Cell Type", "Accuracy"})
    dfh = df.loc[(df.Animal == "Human")]
    dfm = df.loc[(df.Animal == "Mouse")]
    if training:
        dfh = dfh.loc[(dfh.Cell.isin(["A549", "Ramos"]))]
        dfm = dfm.loc[(dfm.Cell.isin(["3T3", "A20", "Macrophage"]))]
        ylabel = "Fitting Accuracy (RMSE)"
    else:
        dfh = dfh.loc[(dfh.Cell.isin(["A549", "Ramos"]) == False)]
        ylabel = "Prediction Accuracy (RMSE)"

    for cell in dfh.Cell.unique():
        preds = dfh.loc[(dfh.Cell == cell)].Predicted.values
        exps = dfh.loc[(dfh.Cell == cell)].Experimental.values
        r2 = mean_squared_error(exps, preds)
        accDFh = accDFh.append(pd.DataFrame({"Cell Type": [cell], "Accuracy": [r2]}))

    for cell in dfm.Cell.unique():
        preds = dfm.loc[(dfm.Cell == cell)].Predicted.values
        exps = dfm.loc[(dfm.Cell == cell)].Experimental.values
        r2 = mean_squared_error(exps, preds)
        accDFm = accDFm.append(pd.DataFrame({"Cell Type": [cell], "Accuracy": [r2]}))

    sns.barplot(x="Cell Type", y="Accuracy", data=accDFh, ax=ax[0], color="k")
    ax[0].set(ylabel=ylabel, ylim=(0, 0.2))
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    if mice:
        sns.barplot(x="Cell Type", y="Accuracy", data=accDFm, ax=ax[1], color="k")
        ax[1].set(ylabel=ylabel, ylim=(0, 0.2))
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

    if seq:
        ax[0].set(title="Human Cells")
        if mice:
            ax[1].set(title="Mouse Cells")
    else:
        ax[0].set(title="Human Cells")
        if mice:
            ax[1].set(title="Mouse Cells")


def R2_Plot_Ligs(df, ax=False, training=False):
    """Plots all accuracies per ligand"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    accDF = pd.DataFrame(columns={"Ligand", "Accuracy"})
    if training:
        df = df.loc[(df.Cell.isin(["A549", "Ramos", "3T3", "A20", "Macrophage"]))]
        df = df.loc[(df.Animal == "Mouse") | (df.Cell != "Macrophage")]
        ylabel = "Fitting Accuracy (RMSE)"
    else:
        df = df.loc[(df.Cell.isin(["Monocyte", "Macrophage", "Fibroblast"]))]
        df = df.loc[(df.Animal == "Human")]
        ylabel = "Prediction Accuracy Human Macrophage (RMSE)"

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

            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Experimental"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Experimental.max()

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
                    normSigs = isoData.Signal.values
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
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Predicted"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Predicted.max()
            masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal), "Experimental"] /= masterSTAT.loc[(masterSTAT.Cell == cell) & (masterSTAT.Animal == animal)].Experimental.max()

    masterSTAT = masterSTAT.fillna(0)
    masterSTAT.replace([np.inf, -np.inf], 0, inplace=True)

    return masterSTAT


def ABtest(ax, xSeq, xMult):
    """Tests Seq vs MB model for a variety of AB blocking ratios"""
    ABblock = np.linspace(start=0, stop=1, num=51)
    models = ["Sequential", "Multivalent"]
    ABtestDF = pd.DataFrame(columns=("Model", "IL13 Ratio", r"Accuracy ($R^2$)"))
    for model in models:
        for ratio in ABblock:
            if model == "Sequential":
                ABdf = residsSeqAB(xSeq, ratio)
            else:
                ABdf = residsAB(xMult, ratio)
            ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "IL13 Ratio": [1 - ratio], r"Accuracy ($R^2$)": [mean_squared_error(ABdf.Experimental.values, ABdf.Predicted.values)]}))

    sns.lineplot(data=ABtestDF, x="IL13 Ratio", y=r"Accuracy ($R^2$)", hue="Model", ax=ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))


def ABtestNorm(ax, xSeq, xMult):
    """Tests Seq vs MB model for a variety of AB blocking ratios"""
    colors = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen"}
    ABblock = np.linspace(start=0, stop=1, num=51)
    models = ["Sequential", "Multivalent"]
    ABtestDF = pd.DataFrame(columns=("Model", "% Available IL13Rα", "Ligand", "Prediction Accuracy (RMSE)"))
    for model in models:
        for ratio in ABblock:
            if model == "Sequential":
                ABdf = residsSeqAB(xSeq, ratio)
            else:
                ABdf = residsAB(xMult, ratio)
            for ligand in ABdf.Ligand.unique():
                ligDF = ABdf.loc[(ABdf.Ligand == ligand)]
                ABtestDF = ABtestDF.append(pd.DataFrame({"Model": [model], "% Available IL13Rα": [100 * (1 - ratio)], "Ligand": [ligand],
                                           "Prediction Accuracy (RMSE)": [mean_squared_error(ligDF.Experimental.values, ligDF.Predicted.values)]}))
    sns.lineplot(data=ABtestDF, x="% Available IL13Rα", y="Prediction Accuracy (RMSE)", hue="Ligand", style="Model", ax=ax, palette=colors)
    ax.set(xlim=(0, 100), ylim=(-.1, 1))


def doseResponsePlot(ax, modelDF, allCells=True):
    """Plot dose response curves for all cells and ligands"""
    meanDF = modelDF.groupby(["Animal", "Cell", "Ligand", "Concentration"])['Experimental'].mean().reset_index()
    stdDF = modelDF.groupby(["Animal", "Cell", "Ligand", "Concentration"])['Experimental'].std().reset_index()
    colorDict = {"hIL4": "k", "hNeo4": "lime", "hIL13": "lightseagreen", "mIL4": "k", "mNeo4": "lime"}
    ligOrder = ["hIL4", "hNeo4", "mIL4", "mNeo4", "hIL13"]
    index = 0

    if allCells:
        cells = modelDF.Cell.unique()
    else:
        cells = ["Macrophage", "Fibroblast", "Monocyte"]

    for cell in cells:
        if allCells:
            animals = modelDF.loc[modelDF.Cell == cell].Animal.unique()
        else:
            animals = ["Human"]
        for animal in animals:
            isoData = modelDF.loc[(modelDF.Cell == cell) & (modelDF.Animal == animal)]
            means = meanDF.loc[(meanDF.Cell == cell) & (meanDF.Animal == animal)]
            stds = stdDF.loc[(stdDF.Cell == cell) & (stdDF.Animal == animal)]
            if animal == "Human":
                ligOrder = ["hIL4", "hNeo4", "hIL13"]
            else:
                ligOrder = ["mIL4", "mNeo4"]
            sns.lineplot(data=isoData, x="Concentration", y="Predicted", hue="Ligand", label="Predicted", ax=ax[index], hue_order=ligOrder, palette=colorDict)
            #sns.scatterplot(data=isoData, x="Concentration", y="Experimental", hue="Ligand", label="Predicted", ax=ax[index], hue_order=ligOrder, palette=colorDict)
            for j, ligand in enumerate(means.Ligand.values):
                ax[index].scatter(x=means.Concentration.values[j], y=means.Experimental.values[j], color=colorDict[ligand])
                ax[index].errorbar(x=means.Concentration.values[j], y=means.Experimental.values[j], yerr=stds.Experimental.values[j], ls='none', color=colorDict[ligand], elinewidth=2, capthick=1)

            ax[index].set(title="Dose Response Curve for " + animal + " " + cell, ylabel="Normalized pSTAT6 (MFI)", ylim=(-.05, 1.25), xlim=(-14, -5))
            handles, labels = ax[index].get_legend_handles_labels()
            if len(isoData.Ligand.unique()) == 3:
                ax[index].legend([handles[0]] + handles[3::], [labels[0]] + labels[3::])
            else:
                ax[index].legend([handles[0]] + handles[2::], [labels[0]] + labels[2::])

            index += 1
