#  import jax.numpy as jx
# from jax.scipy.special import logsumexp
import numpy as jx
from scipy.special import logsumexp
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

EPS = 1e-8


# === UTIL FUNCTIONS === #

def inner_prod(A, B):
    if A.shape != B.shape:
        raise ValueError("Dimension mismatch.")
    return jx.dot(A.ravel(), B.ravel())

def softmin(x, u, eps, axis):
    return eps * logsumexp(-x / eps, b=u, axis=axis)


# === MAIN FUNCTIONS ==== #

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
        return (self.lam * p) / (self.lam + eps)


class OTCost:

    def __init__(self, margdiv, eps, tol):
        self.margdiv = margdiv
        self.eps = eps
        self.tol = tol

    def fit(self, a, x, b=None, y=None):

        if (b is None) and (y is None):
            self._symm_cost(a, x)
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

        # Compute dual objective.
        self.dual_obj_ = (
            - inner_prod(a, self.margdiv.conj_ent(-self.w_))
            - inner_prod(b, self.margdiv.conj_ent(-self.h_))
            + self.eps * jx.sum(self.P_)
            - self.eps * jx.sum(a[:, None] * b[None, :])
        )

        # Compute primal objective.
        self.primal_obj_ = (
            inner_prod(self.C_, self.P_)
            + self.margdiv(self.P_.sum(axis=1), a)
            + self.margdiv(self.P_.sum(axis=0), b)
            + self.eps * inner_prod(self.P_, M)
            # this ^^ is same as, self.eps * inner_prod(P, log(P) - log(a)[:, None] - jx.log(b)[None, :])
            # + self.eps * inner_prod(self.P_, jx.log(self.P_) - jx.log(a)[:, None] - jx.log(b)[None, :])
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
        self.ot_ab(a, x, b, y)
        self.ot_aa(a, x)
        self.ot_bb(b, y)

        # Compute the Sinkhorn divergence.
        return (
            self.ot_ab.primal_obj_
            - .5 * self.ot_aa.primal_obj_
            - .5 * self.ot_bb.primal_obj_
            + .5 * self.eps * ((a.sum() - b.sum()) ** 2)
        )


def asymmetric_sinkhorn(w, h, a, b, C, eps, margdiv, thres):
    
    converged = False

    while not converged:
        
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
    for itr in range(100):
        
        # Compute new param
        wn = margdiv.aprox(softmin(C - w[None, :], a, eps, axis=1), eps)
        w1 = 0.5 * (w - wn)

        # Check convergence
        converged = jx.linalg.norm(w - w1) < thres

        # Param update
        w = w1

    return w


def test_1():
    """
    Check that increasing epsilon increases blur in the
    transport plan.
    """

    import matplotlib.pyplot as plt

    margdiv = ForwardKL(1.0)
    x = np.linspace(-4, 4, 51)[:, None]
    y = np.linspace(-4, 4, 50)[:, None]
    a = np.squeeze(np.exp(-x ** 2))
    b = np.squeeze(np.exp(-y ** 2))

    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)

    for eps, ax in zip((0.01, 0.1, 1.0), axes):
        ax.imshow(
            OTCost(margdiv, eps, 1e-6).fit(a, x, b, y).P_,
            aspect="auto"
        )
        ax.set_title("eps = {}".format(eps))

    fig.set_size_inches((4, 2))
    fig.tight_layout()

    plt.show()


def test_2():
    """
    Check that primal and dual match.
    """

    margdiv = ForwardKL(1.0)
    x = np.linspace(-4, 4, 51)[:, None]
    y = np.linspace(-4, 4, 50)[:, None]
    a = np.squeeze(np.exp(-x ** 2))
    b = np.squeeze(np.exp(-y ** 2))

    print("DUALITY GAP SHOULD BE ZERO...")
    for eps in (0.01, 0.1, 1.0):
        ot = OTCost(margdiv, eps, 1e-6).fit(a, x, b, y)
        print("EPS = {}; Duality Gap = {}".format(
            eps, ot.primal_obj_ - ot.dual_obj_))

    print("\nDUAL OBJECTIVE SHOULD BE NONNEGATIVE...")
    for eps in (0.01, 0.1, 1.0):
        ot = OTCost(margdiv, eps, 1e-6).fit(a, x, b, y)
        print("EPS = {}; Dual Objective = {}".format(
            eps, ot.dual_obj_))
