"""
Implementation of a simple multivalent binding model.
"""

import pathlib
import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import root


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
SigData = pd.read_csv(join(path_here, "src/data/SignalingData.csv"))


def cytBindingModel(Kx, Cplx, doseVec, cellType, animal, relRecp):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    doseVec = np.array(doseVec)

    recCount = np.ravel([recQuantDF.loc[(recQuantDF.Receptor == "IL4Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                             recQuantDF.loc[(recQuantDF.Receptor == "Gamma") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values,
                             recQuantDF.loc[(recQuantDF.Receptor == "IL13Ra") & (recQuantDF["Cell"] == cellType) & (recQuantDF["Animal"] == animal)].Amount.values])

    output = np.zeros([doseVec.size, 2])

    for i, dose in enumerate(doseVec):
        output[i, :] = IL4mod(np.power(10, dose), np.power(10, Kx), recCount, Cplx)

    return output[:, 0] +  output[:, 1] * relRecp
