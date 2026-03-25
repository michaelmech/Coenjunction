import numpy as np

from .MutualInformation import estimate_mi_from_ce


def calculate_transfer_entropy(X, Y, max_lag=10, return_lag=False, mi_kwargs=None):
    """
    Calculate transfer entropy from Y to X, TE(Y -> X).

    The lag is selected from 1..max_lag using the first local minimum of
    auto-mutual information for X; if none exists, uses the global minimum.
    """
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")
    if X.shape[0] <= max_lag:
        raise ValueError("max_lag must be smaller than the number of samples.")

    mi_kwargs = {} if mi_kwargs is None else dict(mi_kwargs)

    ami_values = []
    for lag in range(1, max_lag + 1):
        x_curr = X[lag:].reshape(-1, 1)
        x_past = X[:-lag].reshape(-1, 1)
        ami_values.append(estimate_mi_from_ce(x_curr, x_past, **mi_kwargs))

    optimal_lag = 1
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i - 1] and ami_values[i] < ami_values[i + 1]:
            optimal_lag = i + 1
            break
    else:
        optimal_lag = int(np.argmin(ami_values) + 1)

    x_curr = X[optimal_lag:].reshape(-1, 1)
    x_past = X[:-optimal_lag].reshape(-1, 1)
    y_past = Y[:-optimal_lag].reshape(-1, 1)

    mi_full = estimate_mi_from_ce(x_curr, np.hstack([x_past, y_past]), **mi_kwargs)
    mi_self = estimate_mi_from_ce(x_curr, x_past, **mi_kwargs)
    te = max(float(mi_full - mi_self), 0.0)

    if return_lag:
        return te, optimal_lag
    return te


def calculate_transfer_entropy_with_edge_lag(X, Y, max_lag=10, return_lag=False, mi_kwargs=None):
    """
    Calculate TE(Y -> X), selecting the lag in 1..max_lag that maximizes TE.
    """
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")
    if X.shape[0] <= max_lag:
        raise ValueError("max_lag must be smaller than the number of samples.")

    mi_kwargs = {} if mi_kwargs is None else dict(mi_kwargs)

    best_te = -np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        x_curr = X[lag:].reshape(-1, 1)
        x_past = X[:-lag].reshape(-1, 1)
        y_past = Y[:-lag].reshape(-1, 1)

        mi_full = estimate_mi_from_ce(x_curr, np.hstack([x_past, y_past]), **mi_kwargs)
        mi_self = estimate_mi_from_ce(x_curr, x_past, **mi_kwargs)
        te = float(mi_full - mi_self)

        if te > best_te:
            best_te = te
            best_lag = lag

    best_te = max(float(best_te), 0.0)
    if return_lag:
        return best_te, best_lag
    return best_te
