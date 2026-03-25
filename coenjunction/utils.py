import numpy as np
from scipy.special import logsumexp, ndtr


import numpy as np

def expand_interpretable_transforms(
    X: np.ndarray,
    *,
    center: str = "median",          # "median" or "mean" or "none"
    add_identity: bool = True,
    add_abs_centered: bool = True,   # |x - μ|
    add_eps_insensitive: bool = False,  # max(|x-μ|-eps, 0)
    eps: float = 0.0,
    add_clip: bool = False,          # clip(x, lo, hi)
    clip_lo_q: float = 0.01,
    clip_hi_q: float = 0.99,
    add_range_clip: bool = False,    # clip(|x-μ|, 0, tau) with tau from quantile
    range_tau_q: float = 0.90,
    add_periodic_fold: bool = False, # x mod period (paper's folding idea)
    period: float = np.pi,
    drop_near_constant: bool = True,
    var_tol: float = 1e-14,
):
    """
    Expand columns with a small set of interpretable non-1-to-1 transforms
    to reveal relationships while keeping copula basis simple (e.g. degree=2).

    Returns:
      X_aug: (n, k_aug)
      groups: list[list[int]] mapping original feature j -> indices in X_aug
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D")

    n, k = X.shape
    cols = []
    groups = [[] for _ in range(k)]

    # center per column
    if center == "median":
        mu = np.median(X, axis=0)
    elif center == "mean":
        mu = np.mean(X, axis=0)
    elif center == "none":
        mu = np.zeros(k, dtype=X.dtype)
    else:
        raise ValueError("center must be 'median', 'mean', or 'none'")

    # helper for constant-ish drops
    def _maybe_add(col, j):
        col = np.asarray(col, dtype=float).reshape(n, 1)
        if drop_near_constant and float(np.var(col)) <= var_tol:
            return
        idx = len(cols)
        cols.append(col)
        groups[j].append(idx)

    # precompute quantiles if needed
    if add_clip:
        lo = np.quantile(X, clip_lo_q, axis=0)
        hi = np.quantile(X, clip_hi_q, axis=0)
    if add_range_clip:
        abs_dev = np.abs(X - mu)
        tau = np.quantile(abs_dev, range_tau_q, axis=0)

    for j in range(k):
        x = X[:, j]
        if add_identity:
            _maybe_add(x, j)

        if add_abs_centered:
            _maybe_add(np.abs(x - mu[j]), j)

        if add_eps_insensitive:
            if eps < 0:
                raise ValueError("eps must be >= 0")
            _maybe_add(np.maximum(np.abs(x - mu[j]) - eps, 0.0), j)

        if add_clip:
            _maybe_add(np.clip(x, lo[j], hi[j]), j)

        if add_range_clip:
            _maybe_add(np.clip(np.abs(x - mu[j]), 0.0, float(tau[j])), j)

        if add_periodic_fold:
            if period <= 0:
                raise ValueError("period must be > 0")
            # fold into [0, period)
            _maybe_add(x - np.floor(x / period) * period, j)

    if not cols:
        # fallback: at least identity
        cols = [X.astype(float)]
        groups = [[j] for j in range(k)]
        return np.hstack(cols), groups

    # cols list contains (n,1) blocks; stack -> (n, k_aug)
    X_aug = np.hstack(cols)
    return X_aug, groups


# -----------------------------------------------------------------------------
# Boundary features (more conservative clipping)
# -----------------------------------------------------------------------------
def _modified_ecdf_to_uniform(X: np.ndarray) -> np.ndarray:
    """Column-wise Modified ECDF: rank/(n+1) => strictly inside (0,1)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    n, k = X.shape
    U = np.empty((n, k), dtype=float)

    for j in range(k):
        order = np.argsort(X[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        U[:, j] = ranks / (n + 1.0)

    return U

def _augment_boundary_features(
    phi: np.ndarray,
    U: np.ndarray,
    powers: np.ndarray,
    *,
    boundary_clip: float,
    pairwise: bool = True,
    cross: bool = False,
):
    """Add boundary-aware features.

    Always adds univariate tail features per dimension:
      - log(u_i) and log(1-u_i)

    Optionally adds pairwise tail interaction features (helpful for tail dependence
    like t-copulas):
      - log(u_i) * log(u_j) for i<j   (lower-lower tail)
      - log(1-u_i) * log(1-u_j) for i<j (upper-upper tail)

    If cross=True, also add cross-tail interactions for i<j:
      - log(u_i) * log(1-u_j)
      - log(1-u_i) * log(u_j)

    We append 2*k columns; their powers_ rows are set to -1 to mark
    'do not enforce marginal constraints'.

    IMPORTANT: clip away from 0/1 much more conservatively than machine eps.
    """
    n,k=U.shape

    if boundary_clip is not None:
      boundary_clip=(0.5 / (n + 1)) ** (1.0 / k)

    eps = np.clip(float(boundary_clip),1e-4, 5e-2)
    Uc = np.clip(U, eps, 1.0 - eps)

    logs = np.log(Uc)
    log1m = np.log1p(-Uc)

    parts = [logs, log1m]

    k = U.shape[1]
    if pairwise and k >= 2:
        i, j = np.triu_indices(k, 1)
        parts.append(logs[:, i] * logs[:, j])
        parts.append(log1m[:, i] * log1m[:, j])
        if cross:
            parts.append(logs[:, i] * log1m[:, j])
            parts.append(log1m[:, i] * logs[:, j])

    bnd = np.hstack(parts)
    bnd_powers = -np.ones((bnd.shape[1], powers.shape[1]), dtype=int)

    phi2 = np.hstack([phi, bnd])
    powers2 = np.vstack([powers, bnd_powers])
    return phi2, powers2


# -----------------------------------------------------------------------------
# Importance sampling helper for diagonal-jitter proposal
# -----------------------------------------------------------------------------

def _gauss_legendre_01(n_nodes: int):
    """Nodes/weights for integrating over [0,1] using Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    # map [-1,1] -> [0,1]
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    return t.astype(float), wt.astype(float)


def _log_q_diag_jitter_truncnorm(
    U: np.ndarray,
    *,
    sigma: float,
    quad_nodes: int = 32,
    z_clip: float = 1e-30,
) -> np.ndarray:
    """Log density for the diagonal-jitter proposal.

    Sample scheme:
      t ~ Uniform(0,1)
      u_i | t ~ TruncNormal(mean=t, sd=sigma, support=[0,1]) i.i.d.

    q(u) = ∫_0^1 ∏_i [phi((u_i-t)/sigma)/sigma] / Z(t) dt
    Z(t) = Phi((1-t)/sigma) - Phi((-t)/sigma).

    Integral computed with 1D Gauss-Legendre quadrature.
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must be 2D")
    n, k = U.shape

    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    t_nodes, t_w = _gauss_legendre_01(quad_nodes)  # (R,), (R,)
    log_w = np.log(t_w)
    R = t_nodes.shape[0]

    # log normal constant per-dimension
    log_norm = -0.5 * np.log(2.0 * np.pi) - np.log(sigma)

    # Z(t) for truncation (same for all dims)
    a = (0.0 - t_nodes) / sigma
    b = (1.0 - t_nodes) / sigma
    Z = ndtr(b) - ndtr(a)
    Z = np.clip(Z, z_clip, None)
    logZ = np.log(Z)  # (R,)

    # --- Key change: avoid (R,n,k) by expanding the square and summing over dims ---
    # sum_j (U_{n,j} - t_r)^2 = sum_u2[n] - 2 t_r sum_u[n] + k t_r^2
    sum_u = np.sum(U, axis=1)          # (n,)
    sum_u2 = np.sum(U * U, axis=1)     # (n,)

    inv2sig2 = 0.5 / (sigma * sigma)

    t = t_nodes                          # (R,)
    t2 = t * t                           # (R,)

    # (R,n): k*log_norm - 0.5/sigma^2 * (sum_u2 - 2 t sum_u + k t^2)
    s = (k * log_norm) - inv2sig2 * (
        sum_u2[None, :] - 2.0 * t[:, None] * sum_u[None, :] + (k * t2)[:, None]
    )

    # subtract k*logZ(t_r) for each node r
    s = s - (k * logZ)[:, None]          # (R,n)

    # quadrature integral: log ∫ exp(s) dt ~ logsumexp(log_w + s)
    log_q = logsumexp(log_w[:, None] + s, axis=0)  # (n,)
    return log_q
