"""
Implementation of a simple multivalent binding model.
"""

import pathlib
import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import root
from scipy.optimize import least_squares, fsolve


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
    x0 = np.array([-11, 1, 8.6, 5, 5, 7.6, 5, 9.08, 5, 5, 8.59, 5, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([-14, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1], [-10, 10, 11, 6, 6, 11, 6, 11, 6, 6, 11, 6, 2.7])
    parampredicts = least_squares(resids, x0, bounds=bnds)
    #parampredicts = minimize(resids, x0, method="trust-constr", bounds=bnds, options={"disp": 999}, constraints=cons)
    assert parampredicts.success
    return parampredicts.x


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
    "hNeo4": [xPow[10], xPow[11], 1e2]}
    if not retDF:
        SigData = SigData.loc[(SigData.Cell != "Macrophage") & (SigData.Cell != "Monocyte")]


    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligCplx = CplxDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = cytBindingModel(Kx, ligCplx, Concs, cell, animal, relRecp, macIL4=x[12])
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
        return np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values)


def fitFuncSeq():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([1, -9, 0, 0, -9, 0, -9, 0, 0, -9, 0, 2])  # KXSTAR, slopeT2, mIL4-IL4Ra, mIL4-Gamma, mIL4-IL13Ra, mNeo4-IL4Ra, mNeo4-Gamma, mNeo4-IL13Ra, hIL4-IL4Ra, hIL4-Gamma, hIL4-IL13Ra, hNeo4-IL4Ra, hNeo4-Gamma, hNeo4-IL13Ra (Log 10)
    bnds = ([0.2, -11, -4, -4, -11, -4, -11, -4, -4, -11, -4, -1], [3, -7, 4, 4, -7, 4, -7, 4, 4, -7, 4, 2.7])
    parampredicts = least_squares(residsSeq, x0, bounds=bnds)
    assert parampredicts.success
    return parampredicts.x


def residsSeq(x, retDF=False):
    """"Returns residuals against signaling data"""
    SigData = pd.read_csv(join(path_here, "src/data/SignalingData.csv"))
    masterSTAT = pd.DataFrame(columns={"Cell", "Ligand", "Concentration", "Animal", "Experimental", "Predicted"})
    relRecp = x[0]
    xPow = np.power(10, x)

    KdDict = {"mIL4": [xPow[1], xPow[2], xPow[3]],
    "mNeo4": [xPow[4], xPow[5], 10],
    "hIL4": [xPow[6], xPow[7], xPow[8]],
    "hNeo4": [xPow[9], xPow[10], 10]}

    if not retDF:
        SigData = SigData.loc[(SigData.Cell != "Macrophage") & (SigData.Cell != "Monocyte")]


    for cell in SigData.Cell.unique():
        for animal in SigData.loc[SigData.Cell == cell].Animal.unique():
            for ligand in SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal)].Ligand.unique():
                isoData = SigData.loc[(SigData.Cell == cell) & (SigData.Animal == animal) & (SigData.Ligand == ligand)]
                Concs = isoData.Concentration.values
                normSigs = isoData.Signal.values
                ligKDs = KdDict[ligand]
                if animal == "Human":
                    if cell == "Macrophage":
                        results = seqBindingModel(ligKDs, Concs, cell, animal, relRecp, macIL4=x[11])
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
        return np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values)


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
    bnds = ([0, 0, 0, 0, 0, 0], [recCount[0], recCount[0], max(recCount[0], recCount[1]), max(recCount[0], recCount[2]), recCount[1], recCount[2] + .01])
    for i, dose in enumerate(doseVec):
        solvedAbunds = least_squares(modFunc, bounds=bnds, x0=(recCount[0], 1, 1, 1, recCount[1], recCount[2]), args=(recCount, KdVec, dose))
        solvedAbunds = solvedAbunds.x
        output[i, 0] = solvedAbunds[2] + solvedAbunds[3] * relRecp
    return output


def modFunc(x, recs, KDs, conc):
    conc = np.power(10, conc)
    IL4Ra = x[0]
    IL4RaL = x[1]
    IL4RaGcL = x[2]
    IL4Ra13L = x[3]
    gc = x[4]
    IL13Ra = x[5]

    return [KDs[0] - (IL4Ra*conc)/IL4RaL +
            KDs[1] - (IL4RaL*gc)/IL4RaGcL +
            KDs[2] - (IL4RaL*IL13Ra)/IL4Ra13L +
            recs[0] - IL4Ra - IL4RaL - IL4RaGcL - IL4Ra13L +
            recs[1] - IL4RaGcL - gc +
            recs[2] - IL13Ra - IL4Ra13L]

"""

    return [KDs[0] - (IL4Ra*conc)/IL4RaL,
            KDs[1] - (IL4RaL*gc)/IL4RaGcL,
            KDs[2] - (IL4RaL*IL13Ra)/IL4Ra13L,
            recs[0] - IL4Ra - IL4RaL - IL4RaGcL - IL4Ra13L,
            recs[1] - IL4RaGcL - gc,
            recs[2] - IL13Ra - IL4Ra13L]
"""
