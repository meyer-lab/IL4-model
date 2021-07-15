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
from sklearn.metrics import r2_score


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


def cytBindingModel(Kx, Cplx, doseVec, cellType, animal, macIL4=False):
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

    return output[:, 0] +  output[:, 1]


def fitFunc():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([-11, 8.6, 5, 5, 7.6, 5, 9.08, 5, 5, 8.59, 5, 8, 5, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([-14, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1], [-10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 11, 6, 2.7])
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
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    Kx = x[0]
    xPow = np.power(10, x)

    CplxDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
    "mNeo4": [xPow[4], xPow[5], 1e2],
    "hIL4": [xPow[6], xPow[7], xPow[8]],
    "hNeo4": [xPow[9], xPow[10], 1e2],
    "hIL13": [xPow[11], 1e2, xPow[12]]}

    #SigData = SigData.loc[SigData.Cell != "Monocyte"]
    #SigData = SigData.loc[SigData.Ligand == "hNeo4"]

    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligCplx = CplxDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, macIL4=x[13])
                    else:
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal)
                else:
                    results = cytBindingModel(Kx, ligCplx, Concs, cell, animal)
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
    x0 = np.array([-5, 1, 1, -5, 1, -5, 1, 1, -5, 1, 1, -5, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([-11, -4, -4, -11, -4, -11, -4, -4, -11, -4, -4, -11, -1], [-3, 4, 4, -3, 4, -3, 4, 4, -3, 4, 4, -3, 2.7])
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
    SigData['Signal'] = SigData['Signal'].clip(lower=0)
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    xPow = np.power(10, x)

    KdDict = {"mIL4": [xPow[0], xPow[1], xPow[2]],
    "mNeo4": [xPow[3], xPow[4], 10000],
    "hIL4": [xPow[5], xPow[6], xPow[7]],
    "hNeo4": [xPow[8], xPow[9], 10000],
    "hIL13": [xPow[10], 10000, xPow[11]]}

    #if not retDF:
    #SigData = SigData.loc[SigData.Cell != "Monocyte"]
    #SigData = SigData.loc[SigData.Ligand == "hNeo4"]


    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligKDs = KdDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand, macIL4=x[12])
                    else:
                        results = seqBindingModel(ligKDs, Concs, cell, animal, ligand)
                else:
                    results = seqBindingModel(ligKDs, Concs, cell, animal, ligand)
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


def SignalingFunc(IL4Ra, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[1]/(KDs[0]*KDs[1]/(IL4Ra*conc)+1) + recs[2]/((KDs[0]*KDs[2])/(IL4Ra*conc)+1)


def IL13Func(x, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[2] - (x + (x*conc)/KDs[2] + recs[1]/(KDs[2]*KDs[1]/(x*conc)+1) + recs[0]/((KDs[2]*KDs[0])/(x*conc)+1))


def SignalingFunc13(IL13Ra, KDs, recs, conc):
    conc = np.power(10, conc)
    return recs[1]/(KDs[2]*KDs[1]/(IL13Ra*conc)+1) + recs[0]/((KDs[0]*KDs[2])/(IL13Ra*conc)+1)


def seqBindingModel(KdVec, doseVec, cellType, animal, lig, macIL4=False):
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
    if lig != "hIL13":
        for i, dose in enumerate(doseVec):
            bnds = ((0, recCount[0]))
            solvedIL4Ra = least_squares(IL4Func, x0=recCount[0], bounds=bnds, args=(KdVec, recCount, dose)).x
            output[i, 0] = SignalingFunc(solvedIL4Ra, KdVec, recCount, dose)
    else:
        for i, dose in enumerate(doseVec):
            bnds = ((0, recCount[2]+0.01))
            solvedIL13Ra = least_squares(IL13Func, x0=recCount[2], bounds=bnds, args=(KdVec, recCount, dose)).x
            output[i, 0] = SignalingFunc13(solvedIL13Ra, KdVec, recCount, dose)
        
    return output


def affFit(confInt=False):
    """Displays fit affinities for the MB model."""
    fit = pd.read_csv("src/data/CurrentFit.csv").x.values
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
        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDict)
    else:
        sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDict)
    plt.ylabel(r"$log_{10}(K_D$ (nM))")
    plt.ylim(((-2, 6)))


def affFitSeq(confInt=False):
    """Displays fit affinities for the MB model."""
    fit = pd.read_csv("src/data/CurrentFitSeq.csv").x.values
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

    sns.barplot(x="Ligand", y=r"$K_D$", data=fitDictKDNorm)
    plt.ylabel(r"IL4Rα $log_{10}(K_D)$ (nM))")
    plt.ylim(((-1, 7)))
    plt.savefig("KDNorm.svg")

    plt.figure()
    print(fitDictKDSurf)
    sns.barplot(x="Ligand", y=r"$K_D$", hue="Receptor", data=fitDictKDSurf)
    plt.ylabel(r"$log_{10}(K_D)$ (#/cell)")
    plt.ylim(((-5, 5)))
    plt.savefig("KDSurf.svg")


def R2_Plot_Cells(df, ax=False):
    """Plots all accuracies per cell"""
    accDF = pd.DataFrame(columns={"Cell Type", "Accuracy"})
    for cell in df.Cell.unique():
        preds = df.loc[(df.Cell == cell)].Predicted.values
        exps = df.loc[(df.Cell == cell)].Experimental.values
        r2 = r2_score(exps, preds)
        accDF = accDF.append(pd.DataFrame({"Cell Type": [cell], "Accuracy": [r2]}))
    if not ax:
        sns.barplot(x="Cell Type", y="Accuracy", data=accDF)
        plt.ylabel(r"Accuracy ($R^2$)")
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
    else:
        sns.barplot(x="Cell Type", y="Accuracy", data=accDF, ax=ax)
        ax.set(ylabel=r"Accuracy ($R^2$)", ylim=(0, 1))
        ax.set_xticklabels(rotation=45)   


def R2_Plot_Ligs(df, ax=False):
    """Plots all accuracies per ligand"""
    accDF = pd.DataFrame(columns={"Ligand", "Accuracy"})
    for ligand in df.Ligand.unique():
            preds = df.loc[(df.Ligand == ligand)].Predicted.values
            exps = df.loc[(df.Ligand == ligand)].Experimental.values
            r2 = r2_score(exps, preds)
            accDF = accDF.append(pd.DataFrame({"Ligand": [ligand], "Accuracy": [r2]}))
    if not ax:
        sns.barplot(x="Ligand", y="Accuracy", data=accDF)
        plt.ylabel(r"Accuracy ($R^2$)")
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
    else:
        sns.barplot(x="Ligand", y="Accuracy", data=accDF, ax=ax)
        ax.set(ylabel=r"Accuracy ($R^2$)", ylim=(0, 1))
        ax.set_xticklabels(rotation=45)    
