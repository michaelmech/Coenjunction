import numpy as np
try:
    from numba import njit, prange
except Exception:  # numba optional
    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        return _wrap
    prange = range


# -----------------------------------------------------------------------------
# Orthonormal shifted Legendre basis
# -----------------------------------------------------------------------------



@njit(fastmath=True, cache=True)
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
        # sqrt(5) * 0.5 * (3x^2 - 1)
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
    """
    Cheaper map than full tensor-product:

      - bias
      - univariate shifted-Legendre bases L_m(u_i), m=1..degree
      - pairwise interactions L_a(u_i) * L_b(u_j) for all 1<=a<=degree, 1<=b<=degree

    This is substantially richer than only L1*L1 and is much better at capturing
    tail / corner structure (e.g., from t-copulas) without paying the full cost
    of a k-way tensor basis.
    """

    def __init__(self, degree: int, k: int, pairwise_order: int = 1, pairwise_max_degree: int | None = None):
        self.degree = int(degree)
        self.k = int(k)
        self.pairwise_order = int(pairwise_order)
        self.pairwise_max_degree = int(pairwise_max_degree) if pairwise_max_degree is not None else self.degree

        d = self.pairwise_max_degree
        if d < 1:
            raise ValueError("pairwise_max_degree must be >= 1")

        powers = [[0] * k]

        # univariate
        for i in range(k):
            for m in range(1, self.degree + 1):
                p = [0] * k
                p[i] = m
                powers.append(p)

        # pairwise interactions (degrees up to d per coordinate)
        if self.pairwise_order >= 1 and k >= 2:
            for i in range(k):
                for j in range(i + 1, k):
                    for a in range(1, d + 1):
                        for b in range(1, d + 1):
                            p = [0] * k
                            p[i] = a
                            p[j] = b
                            powers.append(p)

        self.powers_ = np.asarray(powers, dtype=int)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n, k = X.shape
        if k != self.k:
            raise ValueError("X has wrong number of columns")

        d = self.pairwise_max_degree

        # feature count: 1 + k*degree + C(k,2)*d*d
        q = 1 + k * self.degree
        if self.pairwise_order >= 1 and k >= 2:
            q += (k * (k - 1) // 2) * (d * d)

        out = np.empty((n, q), dtype=np.float64)
        out[:, 0] = 1.0

        # precompute univariate bases up to max(degree, d)
        max_m = max(self.degree, d)
        B = np.empty((k, n, max_m + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        for i in range(k):
            for m in range(1, max_m + 1):
                B[i, :, m] = _shifted_legendre(X[:, i], m)

        col = 1
        for i in range(k):
            for m in range(1, self.degree + 1):
                out[:, col] = B[i, :, m]
                col += 1

        if self.pairwise_order >= 1 and k >= 2:
            for i in range(k):
                for j in range(i + 1, k):
                    for a in range(1, d + 1):
                        Bi = B[i, :, a]
                        for b in range(1, d + 1):
                            out[:, col] = Bi * B[j, :, b]
                            col += 1

        return out

def _make_feature_map(feature_mode: str, degree: int, k: int):
    mode = str(feature_mode).lower()
    if mode == "full":
        return _FullOrthoFeatures(degree=degree, k=k)
    if mode in {"pairwise", "pw", "pairwise2"}:
        return _PairwiseOrthoFeatures(degree=degree, k=k, pairwise_order=1)
    raise ValueError("feature_mode must be 'full' or 'pairwise'")
