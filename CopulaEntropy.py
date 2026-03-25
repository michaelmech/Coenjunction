import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc
from numba import njit

from Polynomials_Splines import _shifted_legendre, _FullOrthoFeatures, _PairwiseOrthoFeatures, _make_feature_map
from utils import _augment_boundary_features, _modified_ecdf_to_uniform, _log_q_diag_jitter_truncnorm


class CopulaEntropyEstimator:
    """Barebones maximum-entropy copula entropy estimator (L-BFGS only).

    Fixed in this version:
      - pairwise orthogonal features (for speed / scaling)
      - Sobol base MC + adaptive diagonal-jitter importance samples
      - L2 regularization only (smooth + fast)
      - no early stopping / patience logic
      - adaptive MC budget based on k and degree

    External helpers expected:
      _make_feature_map, _augment_boundary_features,
      _log_q_diag_jitter_truncnorm, _modified_ecdf_to_uniform
    """

    def __init__(
        self,
        degree=2,
        max_num_mc=10_000,
        max_num_mc_adapt=5_000,
        alpha_reg=0.08,
        random_state=None,
        boundary_clip=1e-6,
        adapt_diag_sigma=0.03,
        adapt_quad_nodes=32,
        ess_min_frac=0.10,
        lbfgs_maxiter=200,
        lbfgs_gtol=1e-5,
        lbfgs_ftol=1e-9,
        # --- base MC proposal ---
        mc_base="sobol",  # only "sobol" supported now
        feature_mode="pairwise",
        n_spline_basis=6,
        n_spline_degree=3,
        boundary_features=False,
        cross=False,
        pairwise=False,
    ):
        # core params
        self.mc_base = str(mc_base)
        self.degree = int(degree)
        self.max_num_mc = int(max_num_mc)
        self.max_num_mc_adapt = int(max_num_mc_adapt)
        self.alpha_reg = float(alpha_reg)
        self.random_state = random_state
        self.cross=cross
        self.pairwise=pairwise

        if self.mc_base != "sobol":
            raise ValueError('mc_base must be "sobol" (IsolationForest-based options removed).')

        # boundary / adaptation params
        self.boundary_clip = float(boundary_clip)
        self.adapt_diag_sigma = float(adapt_diag_sigma)
        self.adapt_quad_nodes = int(adapt_quad_nodes)
        self.ess_min_frac = float(ess_min_frac)

        # optimizer params
        self.lbfgs_maxiter = int(lbfgs_maxiter)
        self.lbfgs_gtol = float(lbfgs_gtol)
        self.lbfgs_ftol = float(lbfgs_ftol)

        # rng
        self.rng = np.random.default_rng(random_state)

        # fixed design choices
        self.feature_mode = str(feature_mode).lower()
        self.boundary_features = boundary_features

        # caches
        self.feat_by_k = {}
        self.theta0_by_k = {}
        self.powers_by_k = {}

        self.mc_base_by_key = {}      # (k, n_base) -> points
        self.phi_base_by_key = {}     # (k, n_base) -> (phi, powers)

        self.mc_adapt_by_key = {}     # (k, n_adapt) -> points
        self.phi_adapt_by_key = {}    # (k, n_adapt) -> phi
        self.logq_adapt_by_key = {}   # (k, n_adapt) -> logq

        self.last_diag = {}
        self.last_reliable = True
        self.n_spline_degree=n_spline_degree
        self.n_spline_basis=n_spline_basis

    # -------- budgeting --------

    def _plan_mc_budget(self, k: int, adapt_scale: int = 1) -> tuple[int, int]:
        target_mc = max(3000, 80 * self.degree * k)
        n_base = min(self.max_num_mc, target_mc)
        n_adapt = min(self.max_num_mc_adapt * int(adapt_scale), max(1000, target_mc // 3))
        return int(n_base), int(n_adapt)

    # -------- MC points --------

    def _get_base_mc_points(
        self, U: np.ndarray | None, k: int, n_base: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pts, logq) for the base proposal on (0,1)^k (Sobol-uniform)."""
        key = (k, int(n_base))
        if key in self.mc_base_by_key:
            pts = self.mc_base_by_key[key]
        else:
            eng = qmc.Sobol(d=k, scramble=True, seed=self.random_state)
            pts = eng.random(n=int(n_base))
            self.mc_base_by_key[key] = pts
        logq = np.zeros(pts.shape[0], dtype=float)  # uniform on unit cube
        return pts, logq

    def _diag_jitter_points(self, k: int, n: int) -> np.ndarray:
        """Truncated diagonal-jitter proposal samples on (0,1)^k."""
        n = int(n)
        if n <= 0:
            return np.empty((0, k), dtype=float)

        sigma = self.adapt_diag_sigma
        out = np.empty((n, k), dtype=float)
        filled = 0
        while filled < n:
            m = max(256, n - filled)
            t = self.rng.uniform(0.0, 1.0, size=(m, 1))
            noise = self.rng.normal(0.0, sigma, size=(m, k))
            pts = t + noise
            ok = np.all((pts > 0.0) & (pts < 1.0), axis=1)
            pts = pts[ok]
            take = min(pts.shape[0], n - filled)
            if take:
                out[filled : filled + take] = pts[:take]
                filled += take
        return out

    # -------- features --------

    def _get_feat(self, k: int):
        if k not in self.feat_by_k:
            self.feat_by_k[k] = _make_feature_map(self.feature_mode, self.degree, k,self.n_spline_basis, self.n_spline_degree)
        return self.feat_by_k[k]

    def _get_phi_base_and_powers(
        self, U: np.ndarray | None, k: int, n_base: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (phi_base, powers, logq_base) for the base proposal."""
        key = (k, int(n_base))

        # cache across calls (Sobol base is data-independent)
        if key in self.phi_base_by_key:
            phi, powers = self.phi_base_by_key[key]
            if k not in self.powers_by_k:
                self.powers_by_k[k] = powers
            logq = np.zeros(phi.shape[0], dtype=float)
            return phi, powers, logq

        feat = self._get_feat(k)
        pts, logq = self._get_base_mc_points(U, k, n_base)

        phi = feat.transform(pts)
        powers = getattr(feat, "powers_", None)
        if powers is None:
            raise RuntimeError("feature map must expose powers_")

        if self.boundary_features:
            phi, powers = _augment_boundary_features(
                phi, pts, powers, boundary_clip=self.boundary_clip,cross=self.cross,pairwise=self.pairwise
            )

        phi = np.asarray(phi, dtype=np.float64, order="C")

        self.phi_base_by_key[key] = (phi, powers)
        self.powers_by_k[k] = powers
        if k not in self.theta0_by_k:
            self.theta0_by_k[k] = np.zeros(phi.shape[1], dtype=float)

        return phi, powers, np.asarray(logq, dtype=float)

    def _get_phi_adapt_and_logq(self, k: int, n_adapt: int) -> tuple[np.ndarray, np.ndarray]:
        key = (k, int(n_adapt))
        if key in self.phi_adapt_by_key and key in self.logq_adapt_by_key:
            return self.phi_adapt_by_key[key], self.logq_adapt_by_key[key]

        if n_adapt <= 0:
            self.phi_adapt_by_key[key] = np.empty((0, 1), dtype=np.float32)
            self.logq_adapt_by_key[key] = np.empty((0,), dtype=np.float32)
            return self.phi_adapt_by_key[key], self.logq_adapt_by_key[key]

        feat = self._get_feat(k)
        pts = self._diag_jitter_points(k, n_adapt)
        phi = feat.transform(pts)
        powers = getattr(feat, "powers_", None)
        if self.boundary_features:
            phi, _ = _augment_boundary_features(
                phi, pts, powers, boundary_clip=self.boundary_clip,cross=self.cross,pairwise=self.pairwise
            )

        logq = _log_q_diag_jitter_truncnorm(
            pts, sigma=self.adapt_diag_sigma, quad_nodes=self.adapt_quad_nodes
        )

        phi = np.asarray(phi, dtype=np.float64, order="C")
        logq = np.asarray(logq, dtype=np.float64, order="C")

        self.mc_adapt_by_key[key] = pts
        self.phi_adapt_by_key[key] = phi
        self.logq_adapt_by_key[key] = logq
        return phi, logq

    def _assemble_mc_set(self, U: np.ndarray | None, k: int, adapt_scale: int = 1):
        n_base, n_adapt = self._plan_mc_budget(k, adapt_scale=adapt_scale)
        phi_base, powers, logq_base = self._get_phi_base_and_powers(U, k, n_base)

        log_impw_base = -np.asarray(logq_base, dtype=np.float64, order="C")
        phi_mc = np.asarray(phi_base, dtype=np.float64, order="C")
        log_impw = log_impw_base

        if n_adapt > 0:
            phi_adapt, logq_adapt = self._get_phi_adapt_and_logq(k, n_adapt)
            if phi_adapt.size:
                log_impw_adapt = -np.asarray(logq_adapt, dtype=np.float64, order="C")
                phi_mc = np.vstack([phi_mc, np.asarray(phi_adapt, dtype=np.float64, order="C")])
                log_impw = np.concatenate([log_impw, log_impw_adapt])

        return phi_mc, log_impw, powers, n_base, n_adapt

    # -------- constraints --------

    @staticmethod
    def _enforce_marginal_constraints(powers: np.ndarray, emp_alpha: np.ndarray) -> np.ndarray:
        """Enforce copula marginal constraints on empirical feature means."""
        alpha = emp_alpha.copy()
        q_ = powers.shape[0]
        for col in range(q_):
            exp = powers[col]
            if np.any(exp < 0):
                continue
            s = int(np.sum(exp))
            if s == 0:
                alpha[col] = 1.0
            elif np.sum(exp > 0) == 1:
                alpha[col] = 0.0
        return alpha

    # -------- regularization --------

    def _reg_obj_grad(self, theta: np.ndarray):
        if self.alpha_reg <= 0:
            return 0.0, np.zeros_like(theta)
        obj = 0.5 * self.alpha_reg * float(np.dot(theta, theta))
        grad = self.alpha_reg * theta
        return obj, grad

    # -------- objective / grad --------

    def _objective_and_grad_only(
        self,
        theta: np.ndarray,
        alpha: np.ndarray,
        phi_mc: np.ndarray,
        log_impw: np.ndarray,
    ):
        obj, grad, ess_frac, w_max, a_range = self._compute_obj_grad_numba(
            theta,
            alpha,
            phi_mc,
            log_impw,
            self.alpha_reg
        )
        return float(obj), grad, float(ess_frac), float(w_max), float(a_range)

    @staticmethod
    @njit(cache=True)
    def _compute_obj_grad_numba(
        theta: np.ndarray,
        alpha: np.ndarray,
        phi_mc: np.ndarray,
        log_impw: np.ndarray,
        alpha_reg: float
    ):
        z = phi_mc.dot(theta)
        a = z + log_impw

        a_max = np.max(a)
        a_min = np.min(a)

        exp_a = np.exp(a - a_max)
        sum_exp = np.sum(exp_a)
        log_s = a_max + np.log(sum_exp)

        n_samples = a.shape[0]

        w = exp_a / sum_exp
        log_Z = log_s - np.log(n_samples)

        mu = w.dot(phi_mc)

        if alpha_reg > 0.0:
            reg_obj = 0.5 * alpha_reg * np.dot(theta, theta)
            reg_grad = alpha_reg * theta
        else:
            reg_obj = 0.0
            reg_grad = np.zeros_like(theta)

        obj = log_Z - np.dot(theta, alpha) + reg_obj
        grad = (mu - alpha) + reg_grad

        sum_w2 = np.sum(w * w)
        ess = 1.0 / sum_w2
        ess_frac = ess / n_samples
        w_max = np.max(w)
        a_range = a_max - a_min

        return obj, grad, ess_frac, w_max, a_range

    # -------- solver (L-BFGS only) --------

    def _lbfgs_solve(
        self,
        theta0: np.ndarray,
        alpha: np.ndarray,
        phi_mc: np.ndarray,
        log_impw: np.ndarray,
    ) -> np.ndarray:
        def fun_and_grad(t):
            obj, grad, ess_frac, w_max, a_range = self._objective_and_grad_only(
                t, alpha, phi_mc, log_impw
            )
            self.last_diag = dict(
                obj=float(obj),
                grad_inf=float(np.max(np.abs(grad))),
                ess_frac=float(ess_frac),
                w_max=float(w_max),
                z_range=float(a_range),
                theta_norm=float(np.linalg.norm(t)),
            )
            return obj, grad

        res = minimize(
            lambda t: fun_and_grad(t)[0],
            theta0,
            jac=lambda t: fun_and_grad(t)[1],
            method="L-BFGS-B",
            options={
                "disp": False,
                "maxiter": self.lbfgs_maxiter,
                "gtol": self.lbfgs_gtol,
                "ftol": self.lbfgs_ftol,
            },
        )

        self.last_diag = dict(self.last_diag or {})
        self.last_diag["stop_reason"] = f"lbfgs(status={res.status}, nit={res.nit})"
        return res.x

    # -------- public API --------

    def compute(self, samples: np.ndarray, theta0=None):
        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError("samples must be 2D")
        n, k = samples.shape
        if k < 1:
            raise ValueError("samples must have at least 1 column")

        # map raw samples -> uniform marginals
        U = _modified_ecdf_to_uniform(samples)

        feat = self._get_feat(k)

        # empirical constraints
        emp_phi = feat.transform(U)
        emp_powers = getattr(feat, "powers_", None)
        if emp_powers is None:
            raise RuntimeError("feature map must expose powers_")

        if self.boundary_features:
            emp_phi, emp_powers = _augment_boundary_features(
                emp_phi, U, emp_powers, boundary_clip=self.boundary_clip,cross=self.cross,pairwise=self.pairwise
            )

        emp_phi = np.asarray(emp_phi, dtype=np.float32, order="C")
        emp_alpha = np.mean(emp_phi, axis=0, dtype=np.float64)
        alpha = self._enforce_marginal_constraints(emp_powers, emp_alpha)

        # base fit with adaptive MC budget
        phi_mc, log_impw, powers, n_base, n_adapt = self._assemble_mc_set(U, k, adapt_scale=1)

        if theta0 is None:
            theta0 = self.theta0_by_k.get(k)
            if theta0 is None or theta0.shape[0] != phi_mc.shape[1]:
                theta0 = np.zeros(phi_mc.shape[1], dtype=float)
        else:
            theta0 = np.asarray(theta0, dtype=float)

        theta_star = self._lbfgs_solve(theta0, alpha, phi_mc, log_impw)

        # ESS guard: if IS is weak, increase adaptive budget and refit once (rare)
        _, _, ess_frac_first, _, _ = self._objective_and_grad_only(theta_star, alpha, phi_mc, log_impw)
        if ess_frac_first < (1.5 * self.ess_min_frac) and self.max_num_mc_adapt > 0:
            phi_mc2, log_impw2, _, n_base2, n_adapt2 = self._assemble_mc_set(U, k, adapt_scale=2)
            theta_star = self._lbfgs_solve(theta_star, alpha, phi_mc2, log_impw2)
            phi_mc, log_impw, n_base, n_adapt = phi_mc2, log_impw2, n_base2, n_adapt2

        # cache warm start for same k
        self.theta0_by_k[k] = theta_star

        # final entropy estimate (accumulate in float64)
        z = phi_mc @ theta_star
        a = z + log_impw.astype(np.float64)
        log_Z = logsumexp(a) - np.log(len(a))
        h = -float(np.dot(theta_star, alpha)) + float(log_Z)

        # reliability diagnostics
        w = np.exp(a - logsumexp(a))
        ess = 1.0 / float(np.sum(w * w))
        ess_frac = ess / len(w)
        self.last_reliable = bool(ess_frac >= self.ess_min_frac)

        self.last_diag = dict(self.last_diag or {})
        self.last_diag.update(
            dict(
                n=int(n),
                k=int(k),
                q=int(phi_mc.shape[1]),
                mc_base=int(n_base),
                mc_adapt=int(n_adapt),
                ess_frac=float(ess_frac),
                reliable=bool(self.last_reliable),
            )
        )

        return float(h), theta_star
