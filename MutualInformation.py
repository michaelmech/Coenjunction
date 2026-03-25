import numpy as np
from CopulaEntropy import CopulaEntropyEstimator  # adjust import if needed

def estimate_mi_from_ce(
    X,
    Y,
    *,
    degree=2,
    max_num_mc=10_000,
    max_num_mc_adapt=5_000,
    alpha_reg=0,
    random_state=None,
):
    """
    Simplest MI estimator using copula entropy:

        MI(X;Y) = Hc(X) + Hc(Y) - Hc([X,Y])

    Assumes CopulaEntropyEstimator.compute() already performs
    the ECDF -> uniform transform internally.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of samples")

    def make_est():
        return CopulaEntropyEstimator(
            degree=degree,
            max_num_mc=max_num_mc,
            max_num_mc_adapt=max_num_mc_adapt,
            alpha_reg=alpha_reg,
            random_state=random_state,
        )

    # Marginals
    h_x = 0.0 if X.shape[1] == 1 else float(make_est().compute(X)[0])
    h_y = 0.0 if Y.shape[1] == 1 else float(make_est().compute(Y)[0])

    # Joint
    XY = np.hstack([X, Y])
    h_xy = float(make_est().compute(XY)[0])

    return float(h_x + h_y - h_xy)
