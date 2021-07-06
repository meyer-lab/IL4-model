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
recQuantDF = pd.read_csv(join(path_here, "src/data/RecQuant.csv"))


def cytBindingModel(Kx, Cplx, doseVec, cellType, animal, relRecp, macIL4=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    doseVec = np.array(doseVec)

    if not macIL4:
        recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values])
    else:
        recCount = np.ravel([np.power(10, macIL4),
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values])

    output = np.zeros([doseVec.size, 2])

    for i, dose in enumerate(doseVec):
        output[i, :] = IL4mod(np.power(10, dose), np.power(10, Kx), recCount, Cplx)

    return output[:, 0] +  output[:, 1] * relRecp


def fitFunc():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([-11, 1, 8.6, 5, 5, 7.6, 5, 9.08, 5, 5, 8.59, 5, 8, 5, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([-14, 0.5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1], [-10, 10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 11, 6, 2.7])
    parampredicts = least_squares(resids, x0, bounds=bnds)
    #assert parampredicts.success
    return parampredicts.x


def getConfInterval():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array(pd.read_csv("src/data/CurrentFit.csv").x) # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
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


def resids(x, retDF=False):
    """"Returns residuals against signaling data"""
    SigData = pd.read_csv(join(path_here, "src/data/SignalingData.csv"))
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    Kx = x[0]
    relRecp = x[1]
    xPow = np.power(10, x)

    CplxDict = {"mIL4": [xPow[2], xPow[3], xPow[4]],
    "mNeo4": [xPow[5], xPow[6], 1e2],
    "hIL4": [xPow[7], xPow[8], xPow[9]],
    "hNeo4": [xPow[10], xPow[11], 1e2],
    "hIL13": [xPow[12], 1e2, xPow[13]]}

    #if not retDF:
    #    SigData = SigData.loc[(SigData.Cell != "Macrophage") & (SigData.Cell != "Monocyte")]

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligCplx = CplxDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, relRecp, macIL4=x[14])
                    else:
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
        return masterSTAT.Predicted.values - masterSTAT.Experimental.values


def fitFuncSeq():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([1, -5, 1, 1, -5, 1, -5, 1, 1, -5, 1, -5, 1, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([0.2, -11, -4, -4, -11, -4, -11, -4, -4, -11, -4, -11, -4, -1], [3, -3, 4, 4, -3, 4, -3, 4, 4, -3, 4, -3, 4, 2.7])
    parampredicts = least_squares(residsSeq, x0, bounds=bnds)
    #assert parampredicts.success
    return parampredicts.x


def getConfIntervalSeq():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array(pd.read_csv("src/data/CurrentFitSeq.csv").x) # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
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


def residsSeq(x, retDF=False):
    """"Returns residuals against signaling data"""
    SigData = pd.read_csv(join(path_here, "src/data/SignalingData.csv"))
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    relRecp = x[0]
    xPow = np.power(10, x)

    KdDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
    "mNeo4": [xPow[4], xPow[5], 10000],
    "hIL4": [xPow[6], xPow[7], xPow[8]],
    "hNeo4": [xPow[9], xPow[10], 10000],
    "hIL13": [xPow[11], 10000, xPow[12]]}

    #if not retDF:
    #    SigData = SigData.loc[(SigData.Cell != "Macrophage") & (SigData.Cell != "Monocyte")]


    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligKDs = KdDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = seqBindingModel(ligKDs, Concs, cell, animal, relRecp, macIL4=x[13])
                    else:
                        results = seqBindingModel(ligKDs, Concs, cell, animal, relRecp)
                else:
                    results = seqBindingModel(ligKDs, Concs, cell, animal, relRecp)
                masterSTAT = masterSTAT.append(pd.DataFrame({"Cell": cell, "Ligand": ligand, "Concentration": Concs, "Animal": animal, "Experimental": normSigs, "Predicted": np.ravel(results)}))
            
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
        return masterSTAT.Predicted.values - masterSTAT.Experimental.values


def IL4Func(x, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[0] - (x + (x*conc)/KDs[0] + recs[1]/(KDs[0]*KDs[1]/(x*conc)+1) + recs[2]/((KDs[0]*KDs[2])/(x*conc)+1))


def SignalingFunc(IL4Ra, KDs, recs, conc, T2W):
    conc = np.power(10, conc)
    return recs[1]/(KDs[0]*KDs[1]/(IL4Ra*conc)+1) + recs[2]/((KDs[0]*KDs[2])/(IL4Ra*conc)+1) * T2W


def seqBindingModel(KdVec, doseVec, cellType, animal, relRecp, macIL4=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    if not macIL4:
        recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values])
    else:
        recCount = np.ravel([np.power(10, macIL4),
                                recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                                recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values])
    output = np.zeros([doseVec.size, 1])
    bnds = ((0, recCount[0]))
    for i, dose in enumerate(doseVec):
        solvedIL4Ra = least_squares(IL4Func, x0=recCount[0], bounds=bnds, args=(KdVec, recCount, dose)).x
        output[i, 0] = SignalingFunc(solvedIL4Ra, KdVec, recCount, dose, relRecp)
    return output


def affFit():
    """Displays fit affinities for the MB model."""
    fit = pd.read_csv("src/data/CurrentFit.csv").x.values
    fitDict = pd.DataFrame(columns=["Ligand", "Receptor", r"K_D"])
    receptorList = ["IL4Rα", "gc", "IL13Rα"]
    xPow = fit * -1
    xPow += 9

    CplxDict = {"mIL4": [xPow[2], xPow[3], xPow[4]],
    "mNeo4": [xPow[5], xPow[6], 1e2],
    "hIL4": [xPow[7], xPow[8], xPow[9]],
    "hNeo4": [xPow[10], xPow[11], 1e2],
    "hIL13": [xPow[12], 1e2, xPow[13]]}
    for lig in CplxDict:
        for i, receptor in enumerate(receptorList):
            KD = CplxDict[lig][i]
            fitDict = fitDict.append(pd.DataFrame({"Ligand": [lig], "Receptor": [receptor], r"K_D": [KD]}))

    sns.barplot(x="Ligand", y=r"K_D", hue="Receptor", data=fitDict)
    plt.ylabel(r"log_{10}(KD (nM))")
    plt.ylim(((-2, 5)))


def affFitSeq():
    """Displays fit affinities for the MB model."""
    fit = pd.read_csv("src/data/CurrentFitSeq.csv").x.values
    fitDict = pd.DataFrame(columns=["Ligand", "Receptor", r"K_D"])
    receptorList = ["IL4Rα", "gc", "IL13Rα"]
    xPow = fit

    KdDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
    "mNeo4": [xPow[4], xPow[5], 10000],
    "hIL4": [xPow[6], xPow[7], xPow[8]],
    "hNeo4": [xPow[9], xPow[10], 10000],
    "hIL13": [xPow[11], 10000, xPow[12]]}
    for lig in KdDict:
        for i, receptor in enumerate(receptorList):
            KD = KdDict[lig][i]
            fitDict = fitDict.append(pd.DataFrame({"Ligand": [lig], "Receptor": [receptor], r"K_D": [KD]}))

    sns.barplot(x="Ligand", y=r"K_D", hue="Receptor", data=fitDict)
    plt.ylabel(r"log_{10}(KD (nM))")
    plt.ylim(((-11, 5)))