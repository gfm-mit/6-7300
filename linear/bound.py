import numpy as np


def getIterationBoundCond(J, tol = 1e-6):
    # Chebyshev bound   
    cond = np.linalg.cond(J)
    temp = ((cond ** 0.5)-1)/(cond ** 0.5 + 1)
    return (np.log(tol) - np.log(2))/(np.log(temp))

def getIterationBoundGirshgorin(J, tol = 1e-6):
    # Girshgorin bound
    diag = np.diag(J)
    temp = np.abs(np.sum(J - np.diag(diag), axis = 1))
    lmax = np.max(diag + temp)
    lmin = max(0.000001, np.min(diag - temp))
    cond = lmax/lmin
    temp = ((cond ** 0.5)-1)/(cond ** 0.5 + 1)
    # print(J)
    # print(diag, J - np.diag(diag), np.sum(J - np.diag(diag), axis = 0), temp, lmax, lmin, cond)
    return (np.log(tol) - np.log(2))/(np.log(temp))

def getIterationBoundEigen(J, tol = 1e-6):
    # Different Eigen Values bound
    ls, _ = np.linalg.eig(J)
    return np.unique(ls).shape[0]