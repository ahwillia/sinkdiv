#  import jax.numpy as jx
# from jax.scipy.special import logsumexp
import numpy as jx
from scipy.special import logsumexp
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from numpy.testing import assert_allclose

EPS = 1e-8


# === UTIL FUNCTIONS === #

def inner_prod(A, B):
    if A.shape != B.shape:
        raise ValueError("Dimension mismatch.")
    return jx.dot(A.ravel(), B.ravel())

def softmin(x, u, eps, axis):
    return eps * logsumexp(-x / eps, b=u, axis=axis)


# === MAIN FUNCTIONS ==== #

class Balanced:

    def __call__(self, p, q):
        return 0.0 if np.allclose(p, q) else np.inf

    def ent(self, t):
        return 0.0 if np.allclose(t, 1) else np.inf

    def conj_ent(self, z):
        return z

    def aprox(self, p, eps):
        return p


class ForwardKL:
    """
    Scaled forward KL divergence.
    """

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, p, q):
        return self.lam * (
            jx.sum(p * (jx.log(p + EPS) - jx.log(q + EPS))) - p.sum() + q.sum()
        )

    def ent(self, t):
        return self.lam * (t * jx.log(t) - t + 1)

    def conj_ent(self, z):
        return self.lam * (jx.exp(z / self.lam) - 1)

    def aprox(self, p, eps):
        return p * (self.lam / (self.lam + eps))
        # return p / (1 + eps / self.lam)


class OTCost:

    def __init__(self, margdiv, eps, tol):
        self.margdiv = margdiv
        self.eps = eps
        self.tol = tol

    def fit(self, a, x, b=None, y=None):

        # Fit symmetric optimal transport cost.
        if (b is None) and (y is None):
            self._symm_cost(a, x)
        
        # Fit asymmetric optimal transport cost.
        else:
            self._asymm_cost(a, x, b, y)

        return self

    def _symm_cost(self, a, x):

        # Compute pairwise cost matrix.
        self.C_ = jx.asarray(
            squareform(pdist(x, metric="sqeuclidean")))

        # Run fixed point iteration for symmetric case.
        w0 = jx.zeros(a.size)
        thres = self.tol * np.sqrt(a.size)
        self.w_ = symmetric_sinkhorn(
            w0, a, self.C_, self.eps, self.margdiv, thres
        )

        # In the symmetric case, the dual potentials match.
        self.h_ = self.w_

        # Evaluate the primal and dual objectives.
        self._calc_objectives(a, b)

    def _asymm_cost(self, a, x, b, y):

        # Compute pairwise cost matrix
        self.C_ = jx.asarray(cdist(x, y, metric="sqeuclidean"))

        # Run classical Sinkhorn iterations, generalized for unbalanced marginal penalties.
        w0, h0 = jx.zeros(a.size), jx.zeros(b.size)
        thres = self.tol * np.sqrt(a.size + b.size)
        self.w_, self.h_ = asymmetric_sinkhorn(
            w0, h0, a, b, self.C_, self.eps, self.margdiv, thres
        )

        # Evaluate the primal and dual objectives.
        self._calc_objectives(a, b)

    def _calc_objectives(self, a, b):

        # Compute transport plan.
        M = (self.w_[:, None] + self.h_[None, :] - self.C_) / self.eps
        log_P = jx.log(a)[:, None] + jx.log(b)[None, :] + M
        self.P_ = jx.exp(log_P)

        # !! Sanity check   !!
        # assert_allclose(self.P_, (a[:, None] * b[None, :]) * np.exp(M))

        # Compute dual objective.
        self.dual_obj_ = (
            - inner_prod(a, self.margdiv.conj_ent(-self.w_))
            - inner_prod(b, self.margdiv.conj_ent(-self.h_))
            - self.eps * jx.sum(self.P_)
            + self.eps * jx.sum(a[:, None] * b[None, :])
        )

        kl = ForwardKL(1.0)

        # Compute primal objective.
        self.primal_obj_ = (
            inner_prod(self.C_, self.P_)
            + self.margdiv(self.P_.sum(axis=1), a)
            + self.margdiv(self.P_.sum(axis=0), b)
            # + self.eps * inner_prod(self.P_, M)
            # this ^^ is same as, self.eps * inner_prod(P, log(P) - log(a)[:, None] - jx.log(b)[None, :])
            # + self.eps * inner_prod(self.P_, jx.log(self.P_) - jx.log(a)[:, None] - jx.log(b)[None , :])
            + self.eps * kl(self.P_, a[:, None] * b[None, :])
        )


class SinkDiv:
    """
    Sinkhorn Divergence.
    """

    def __init__(self, margdiv, eps=1.0, tol=1e-6, compute_primal=False):
        self.margdiv = margdiv
        self.eps = eps
        self.tol = tol
        self.compute_primal = compute_primal
        self.fitted_ = False

    def __call__(self, a, x, b, y):

        # Check inputs.
        a, x = jx.asarray(a), jx.asarray(x)
        b, y = jx.asarray(b), jx.asarray(y)
        if (a.size != x.shape[0]) or (b.size != y.shape[0]):
            raise ValueError("Dimension mismatch.")

        # Initialize OTCost objects.
        self.ot_ab = OTCost(self.margdiv, self.eps, self.tol)
        self.ot_aa = OTCost(self.margdiv, self.eps, self.tol)
        self.ot_bb = OTCost(self.margdiv, self.eps, self.tol)

        # Compute OT costs via Sinkhorn iterations.
        self.ot_ab.fit(a, x, b, y)
        self.ot_aa.fit(a, x)
        self.ot_bb.fit(b, y)

        # Compute the Sinkhorn divergence.
        return (
            self.ot_ab.primal_obj_
            - .5 * self.ot_aa.primal_obj_
            - .5 * self.ot_bb.primal_obj_
            + .5 * self.eps * ((a.sum() - b.sum()) ** 2)
        )


def asymmetric_sinkhorn(w, h, a, b, C, eps, margdiv, thres):
    
    converged = False

    # while not converged:
    for itr in range(10000):
        
        # Compute new params
        w1 = -margdiv.aprox(
            softmin(C - h[None, :], b[None, :], eps, axis=1), eps
        )
        h1 = -margdiv.aprox(
            softmin(C - w1[:, None], a[:, None], eps, axis=0), eps
        )

        # Check convergence
        wr, hr = (w - w1), (h - h1) 
        param_change = jx.sqrt(jx.dot(hr, hr) + jx.dot(wr, wr))
        converged = param_change < thres

        # Param update
        w = w1
        h = h1

    return w, h


def symmetric_sinkhorn(w, a, C, eps, margdiv, thres):

    converged = False

    # while not converged:
    for itr in range(10000):
        
        # Compute new param
        wn = margdiv.aprox(softmin(C - w[None, :], a, eps, axis=1), eps)
        w1 = 0.5 * (w - wn)

        # Check convergence
        converged = jx.linalg.norm(w - w1) < thres

        # Param update
        w = w1

    return w

