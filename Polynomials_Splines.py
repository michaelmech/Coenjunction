from pure_spline_features import make_feature_map_spline_only

import numpy as np
from scipy.interpolate import BSpline

# -----------------------------------------------------------------------------
# Orthonormal shifted Legendre basis
# -----------------------------------------------------------------------------

def _shifted_legendre(u: np.ndarray, m: int) -> np.ndarray:
    n = u.shape[0]
    x = 2.0 * u - 1.0

    if m == 0:
        out = np.empty(n, np.float64)
        out.fill(1.0)
        return out

    if m == 1:
        return np.sqrt(3.0) * x

    if m == 2:
        return (0.5 * np.sqrt(5.0)) * (3.0 * x * x - 1.0)

    if m == 3:
        return (0.5 * np.sqrt(7.0)) * (5.0 * x * x * x - 3.0 * x)

    # recurrence for m >= 4
    p0 = np.ones(n, dtype=np.float64)
    p1 = x.copy()
    for k in range(2, m + 1):
        p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        p0, p1 = p1, p2
    return np.sqrt(2.0 * m + 1.0) * p1


# -----------------------------------------------------------------------------
# Feature maps
# -----------------------------------------------------------------------------

class _FullOrthoFeatures:
    """Tensor-product shifted-Legendre basis up to total degree."""

    def __init__(self, degree: int, k: int):
        self.degree = int(degree)
        self.k = int(k)

        powers = []

        def rec(pos, remaining, cur):
            if pos == self.k:
                powers.append(cur.copy())
                return
            for d in range(remaining + 1):
                cur[pos] = d
                rec(pos + 1, remaining - d, cur)

        rec(0, self.degree, [0] * self.k)
        self.powers_ = np.asarray(powers, dtype=int)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n, k = X.shape
        if k != self.k:
            raise ValueError("X has wrong number of columns")

        q_ = self.powers_.shape[0]

        # Precompute all univariate bases
        B = np.empty((k, n, self.degree + 1), dtype=float)
        B[:, :, 0] = 1.0
        for j in range(k):
            for m in range(1, self.degree + 1):
                B[j, :, m] = _shifted_legendre(X[:, j], m)

        out = np.ones((n, q_), dtype=float)
        for j in range(k):
            out *= np.take(B[j], self.powers_[:, j], axis=1)

        return out


class _PairwiseOrthoFeatures:
    """Cheap map: bias + univariate bases + pairwise L1*L1."""

    def __init__(self, degree: int, k: int, pairwise_order: int = 1):
        self.degree = int(degree)
        self.k = int(k)
        self.pairwise_order = int(pairwise_order)

        powers = [[0] * k]
        for i in range(k):
            for m in range(1, self.degree + 1):
                p = [0] * k
                p[i] = m
                powers.append(p)

        if self.pairwise_order >= 1 and k >= 2:
            for i in range(k):
                for j in range(i + 1, k):
                    p = [0] * k
                    p[i] = 1
                    p[j] = 1
                    powers.append(p)

        self.powers_ = np.asarray(powers, dtype=int)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n, k = X.shape

        q = 1 + k * self.degree
        if self.pairwise_order >= 1 and k >= 2:
            q += (k * (k - 1)) // 2

        out = np.empty((n, q), dtype=np.float64)
        out[:, 0] = 1.0

        col_idx = 1
        for i in range(k):
            for m in range(1, self.degree + 1):
                out[:, col_idx] = _shifted_legendre(X[:, i], m)
                col_idx += 1

        if self.pairwise_order >= 1 and k >= 2:
            L1 = [_shifted_legendre(X[:, i], 1) for i in range(k)]
            for i in range(k):
                for j in range(i + 1, k):
                    out[:, col_idx] = L1[i] * L1[j]
                    col_idx += 1

        return out


# -----------------------------------------------------------------------------
# Hybrid-B: Legendre + centered B-splines + restricted spline pairwise
# -----------------------------------------------------------------------------

def _clamped_knots_cosine(n_basis: int, degree: int) -> np.ndarray:
    """
    Clamped knot vector on [0,1] with cosine-spaced interior knots:
    concentrates resolution near 0 and 1 (tail regions).
    Length = n_basis + degree + 1, suitable for BSpline.
    """
    n_basis = int(n_basis)
    degree = int(degree)
    if n_basis <= degree:
        raise ValueError("n_basis must be > degree")

    n_int = n_basis - degree - 1
    if n_int == 0:
        interior = np.empty((0,), dtype=float)
    else:
        x = np.linspace(0.0, 1.0, n_int + 2, dtype=float)[1:-1]
        interior = 0.5 * (1.0 - np.cos(np.pi * x))

    t0 = np.zeros(degree + 1, dtype=float)
    t1 = np.ones(degree + 1, dtype=float)
    return np.concatenate([t0, interior, t1])


def _bspline_basis_matrix(u: np.ndarray, *, n_basis: int, degree: int) -> np.ndarray:
    """Evaluate all clamped B-spline basis functions at u in [0,1]. Shape (n, n_basis)."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("u must be 1D")

    t = _clamped_knots_cosine(n_basis=n_basis, degree=degree)
    out = np.empty((u.shape[0], n_basis), dtype=float)

    for r in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[r] = 1.0
        bs = BSpline(t, c, degree, extrapolate=False)
        out[:, r] = bs(u)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)



import numpy as np

# assumes you already have:
#   _shifted_legendre(u, m)
#   _bspline_basis_matrix(u, n_basis, degree)

class _HybridBFeatures:
    """
    Hybrid-Full feature map:
      - Bias
      - Full tensor-product shifted-Legendre terms up to total degree degree_poly
        (i.e., all multi-indices p with sum(p) <= degree_poly)
      - Centered B-spline univariate terms (n_spline_basis per dim)
      - Restricted spline pairwise: S_i[r] * S_j[r] (same-index) for each pair
    """

    def __init__(self, degree_poly: int, k: int, *, n_spline_basis: int = 6, spline_degree: int = 3):
        self.degree_poly = int(degree_poly)
        self.k = int(k)
        self.n_spline_basis = int(n_spline_basis)
        self.spline_degree = int(spline_degree)

        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.degree_poly < 0:
            raise ValueError("degree_poly must be >= 0")
        if self.n_spline_basis <= self.spline_degree:
            raise ValueError("n_spline_basis must be > spline_degree")

        # --- polynomial powers: all multi-indices with total degree <= degree_poly ---
        poly_powers = []

        def rec(pos: int, remaining: int, cur: list[int]):
            if pos == self.k:
                poly_powers.append(cur.copy())
                return
            for d in range(remaining + 1):
                cur[pos] = d
                rec(pos + 1, remaining - d, cur)

        rec(0, self.degree_poly, [0] * self.k)
        self._poly_powers = np.asarray(poly_powers, dtype=int)  # includes bias row [0..0]

        # --- powers_ bookkeeping for your alpha enforcement heuristic ---
        # Order matches transform(): [poly (incl bias)] + [spline univariates] + [restricted spline pairwise]
        powers = []
        powers.extend(self._poly_powers.tolist())

        # spline univariates (encode as exp=1 along that dim)
        for i in range(self.k):
            for _ in range(self.n_spline_basis):
                p = [0] * self.k
                p[i] = 1
                powers.append(p)

        # restricted spline pairwise (encode as exp=1,1)
        if self.k >= 2:
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    for _ in range(self.n_spline_basis):
                        p = [0] * self.k
                        p[i] = 1
                        p[j] = 1
                        powers.append(p)

        self.powers_ = np.asarray(powers, dtype=int)

        # Center spline columns so E_U[b_r(U)] ≈ 0 under U~Unif(0,1)
        g = (np.arange(4096, dtype=float) + 0.5) / 4096.0
        Bg = _bspline_basis_matrix(g, n_basis=self.n_spline_basis, degree=self.spline_degree)
        self._spline_means = Bg.mean(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, k = X.shape
        if k != self.k:
            raise ValueError("X has wrong number of columns")

        q_poly = self._poly_powers.shape[0]
        q = q_poly + k * self.n_spline_basis
        if k >= 2:
            q += ((k * (k - 1)) // 2) * self.n_spline_basis

        out = np.empty((n, q), dtype=np.float64)
        col = 0

        # ---- full polynomial block (includes bias) ----
        # Precompute univariate Legendre bases up to degree_poly for each dim
        # B[j, :, m] = L_m(X[:, j])
        B = np.empty((k, n, self.degree_poly + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        for j in range(k):
            for m in range(1, self.degree_poly + 1):
                B[j, :, m] = _shifted_legendre(X[:, j], m)

        # Build tensor products using powers matrix (vectorized across all columns)
        P = self._poly_powers  # (q_poly, k)
        poly = np.ones((n, q_poly), dtype=np.float64)
        for j in range(k):
            poly *= np.take(B[j], P[:, j], axis=1)  # (n, q_poly)

        out[:, col : col + q_poly] = poly
        col += q_poly

        # ---- spline univariates (centered) ----
        S = []
        for i in range(k):
            Bi = _bspline_basis_matrix(X[:, i], n_basis=self.n_spline_basis, degree=self.spline_degree)
            Bi = Bi - self._spline_means[None, :]
            S.append(Bi)
            out[:, col : col + self.n_spline_basis] = Bi
            col += self.n_spline_basis

        # ---- restricted spline pairwise: same-index products ----
        if k >= 2:
            for i in range(k):
                Si = S[i]
                for j in range(i + 1, k):
                    out[:, col : col + self.n_spline_basis] = Si * S[j]
                    col += self.n_spline_basis

        return out

def _make_feature_map(feature_mode: str, degree: int, k: int,n_spline_basis=6,spline_degree=3):
    mode = str(feature_mode).lower()

    if mode == "full":
        base = _FullOrthoFeatures(degree=degree, k=k)
    elif mode in {"pairwise", "pw", "pairwise2"}:
        base = _PairwiseOrthoFeatures(degree=degree, k=k, pairwise_order=1)
    elif mode in {"hybrid_b", "hybrid-b", "hybridb", "hb"}:
        base = _HybridBFeatures(degree_poly=degree, k=k, n_spline_basis=n_spline_basis, spline_degree=spline_degree)
    elif mode.startswith("spline"):
        base = make_feature_map_spline_only(feature_mode, k)
    else:
        raise ValueError("feature_mode must be 'full', 'pairwise', 'hybrid_b' or 'spline'")
    return base
