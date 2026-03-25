
"""pure_spline_features.py

Pure-spline feature map for your CopulaEntropyEstimator.

What this gives you
-------------------
- Exponential-family max-entropy copula stays the same:
    p(u) ∝ exp(theta · phi(u))
- You replace the Legendre-polynomial basis with B-splines.

Two interaction options
-----------------------
1) restricted (default): per pair (i,j) add same-index products S_i[r]*S_j[r]
   -> O(B) features per pair, good tails, manageable.

2) full_tensor: per pair (i,j) add all products S_i[r]*S_j[s]
   -> O(B^2) features per pair, best expressivity, can explode.

Usage
-----
1) Put this file next to OrthogonalPolynomials.py (or merge into it).
2) In OrthogonalPolynomials.py, import and wire _make_feature_map to return
   _SplineOnlyFeatures when feature_mode is 'spline' (or similar).

Example:
    feat = _SplineOnlyFeatures(k=20, n_basis=8, degree=3, pairwise='restricted')
    phi = feat.transform(U)

Notes
-----
- This feature map expects U in (0,1) with shape (n,k).
- Columns are centered so E_unif[phi] ~ 0 for stability.
- Consider standardizing phi columns on the MC grid (optional); see comment.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _clamped_knots_cosine(n_basis: int, degree: int) -> np.ndarray:
    """Clamped knot vector on [0,1] with cosine-spaced interior knots.

    Cosine spacing concentrates knot density near 0 and 1 (tail regions).

    Knot vector length must be n_basis + degree + 1 for BSpline.
    """
    n_basis = int(n_basis)
    degree = int(degree)
    if n_basis <= degree:
        raise ValueError("n_basis must be > degree")

    n_int = n_basis - degree - 1
    if n_int <= 0:
        interior = np.empty((0,), dtype=float)
    else:
        # evenly spaced in angle -> more knots near 0 and 1
        x = np.linspace(0.0, 1.0, n_int + 2, dtype=float)[1:-1]
        interior = 0.5 * (1.0 - np.cos(np.pi * x))

    t0 = np.zeros(degree + 1, dtype=float)
    t1 = np.ones(degree + 1, dtype=float)
    return np.concatenate([t0, interior, t1])


def _bspline_basis_matrix(u: np.ndarray, *, n_basis: int, degree: int) -> np.ndarray:
    """Evaluate all clamped B-spline basis functions at u ∈ [0,1].

    Returns shape (n, n_basis).
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("u must be 1D")

    t = _clamped_knots_cosine(n_basis=n_basis, degree=degree)
    out = np.empty((u.shape[0], n_basis), dtype=float)

    # Construct each basis spline by using a one-hot coefficient vector.
    for r in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[r] = 1.0
        bs = BSpline(t, c, degree, extrapolate=False)
        out[:, r] = bs(u)

    # BSpline returns nan outside support if extrapolate=False; clamp to 0.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class _SplineOnlyFeatures:
    """Pure spline features: bias + spline univariates + spline interactions.

    Parameters
    ----------
    k : int
        Dimension.
    n_basis : int
        Number of B-spline basis functions per dimension.
    degree : int
        B-spline degree (3 = cubic).
    pairwise : {'none','restricted','full_tensor'}
        Interaction feature set.
    center : bool
        Center each spline basis column to have ~zero mean under U(0,1).
    """

    def __init__(
        self,
        *,
        k: int,
        n_basis: int = 8,
        degree: int = 3,
        pairwise: str = "restricted",
        center: bool = True,
    ):
        self.k = int(k)
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        self.pairwise = str(pairwise).lower()
        self.center = bool(center)

        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.n_basis <= self.degree:
            raise ValueError("n_basis must be > degree")
        if self.pairwise not in {"none", "restricted", "full_tensor"}:
            raise ValueError("pairwise must be one of: none, restricted, full_tensor")

        # powers_ is only used by your marginal-constraint heuristic.
        # Mark spline univariates as 'univariate' (sum(exp)>0 and only one dim active).
        # Mark interactions as 'pairwise'.
        powers = []
        powers.append([0] * self.k)  # bias

        # univariate spline basis terms
        for i in range(self.k):
            for _ in range(self.n_basis):
                p = [0] * self.k
                p[i] = 1
                powers.append(p)

        # interactions
        if self.pairwise != "none" and self.k >= 2:
            if self.pairwise == "restricted":
                # B terms per pair
                for i in range(self.k):
                    for j in range(i + 1, self.k):
                        for _ in range(self.n_basis):
                            p = [0] * self.k
                            p[i] = 1
                            p[j] = 1
                            powers.append(p)
            else:
                # full tensor: B^2 terms per pair
                for i in range(self.k):
                    for j in range(i + 1, self.k):
                        for _ in range(self.n_basis * self.n_basis):
                            p = [0] * self.k
                            p[i] = 1
                            p[j] = 1
                            powers.append(p)

        self.powers_ = np.asarray(powers, dtype=int)

        # Precompute centering constants under U(0,1) via a dense grid.
        if self.center:
            g = (np.arange(8192, dtype=float) + 0.5) / 8192.0
            Bg = _bspline_basis_matrix(g, n_basis=self.n_basis, degree=self.degree)
            self._means = Bg.mean(axis=0)
        else:
            self._means = np.zeros(self.n_basis, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform U ∈ (0,1)^(n,k) -> feature matrix phi ∈ R^(n,q)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, k = X.shape
        if k != self.k:
            raise ValueError("X has wrong number of columns")

        # feature count
        q = 1 + k * self.n_basis
        if self.pairwise != "none" and k >= 2:
            n_pairs = (k * (k - 1)) // 2
            if self.pairwise == "restricted":
                q += n_pairs * self.n_basis
            else:
                q += n_pairs * (self.n_basis * self.n_basis)

        out = np.empty((n, q), dtype=np.float64)
        col = 0

        # bias
        out[:, col] = 1.0
        col += 1

        # univariate spline blocks
        S = []
        for i in range(k):
            Bi = _bspline_basis_matrix(X[:, i], n_basis=self.n_basis, degree=self.degree)
            Bi = Bi - self._means[None, :]
            S.append(Bi)
            out[:, col : col + self.n_basis] = Bi
            col += self.n_basis

        # pairwise interactions
        if self.pairwise != "none" and k >= 2:
            if self.pairwise == "restricted":
                for i in range(k):
                    Si = S[i]
                    for j in range(i + 1, k):
                        out[:, col : col + self.n_basis] = Si * S[j]
                        col += self.n_basis
            else:
                # full tensor products
                for i in range(k):
                    Si = S[i]
                    for j in range(i + 1, k):
                        Sj = S[j]
                        # produce (n, B^2) block in row-major (r,s)
                        block = (Si[:, :, None] * Sj[:, None, :]).reshape(n, -1)
                        out[:, col : col + block.shape[1]] = block
                        col += block.shape[1]

        return out


# -----------------------------------------------------------------------------
# Drop-in wiring snippet for OrthogonalPolynomials.py
# -----------------------------------------------------------------------------

def make_feature_map_spline_only(feature_mode: str, k: int, *, n_basis: int = 8, degree: int = 3):
    """Helper: returns a spline feature map based on a feature_mode string."""
    mode = str(feature_mode).lower()
    if mode in {"spline", "spline_only", "spline-restricted", "spline_restricted"}:
        return _SplineOnlyFeatures(k=k, n_basis=n_basis, degree=degree, pairwise="restricted")
    if mode in {"spline_full", "spline-full", "spline_tensor", "spline-tensor", "spline_full_tensor"}:
        return _SplineOnlyFeatures(k=k, n_basis=n_basis, degree=degree, pairwise="full_tensor")
    if mode in {"spline_uni", "spline-univariate", "spline_univariate", "spline_none"}:
        return _SplineOnlyFeatures(k=k, n_basis=n_basis, degree=degree, pairwise="none")
    raise ValueError("unknown spline feature_mode")
