"""
Implementation of a simple multivalent binding model.
"""

import pathlib
import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import root


def Req_func2(Req: np.ndarray, L0, KxStar, Rtot: np.ndarray, Cplx: np.ndarray, Ctheta: np.ndarray, Kav: np.ndarray):
    Psi = Req * Kav * KxStar
    Psi = np.pad(Psi, ((0, 0), (0, 1)), constant_values=1)
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * np.sum(Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log1p(Psirs - 1))), axis=0)
    return Req + Rbound - Rtot


def polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Lbound: a list of Lbound of each complex
        Rbound: a list of Rbound of each kind of receptor
    """
    # Consistency check
    Kav = np.array(Kav)
    assert Kav.ndim == 2
    Rtot = np.array(Rtot, dtype=float)
    assert Rtot.ndim == 1
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    Ctheta = np.array(Ctheta)
    assert Ctheta.ndim == 1

    assert Kav.shape[0] == Cplx.shape[1]
    assert Kav.shape[1] == Rtot.size
    assert Cplx.shape[0] == Ctheta.size
    Ctheta = Ctheta / np.sum(Ctheta)

    # Solve Req
    lsq = root(Req_func2, Rtot, method="lm", args=(L0, KxStar, Rtot, Cplx, Ctheta, Kav), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log1p(Psirs - 1)))
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Rbound


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
        output[i, :] = polyc(np.power(10, dose), np.power(10, Kx), recCount, [[1, 1]], [1.0], Cplx)[0][1:2]

    return output[:, 0] +  output[:, 1] * relRecp