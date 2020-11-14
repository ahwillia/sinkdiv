import pytest
import numpy as np
from numpy.testing import assert_allclose
from sinkdiv import OTCost, ForwardKL, Balanced
from scipy.optimize import approx_fprime


def test_entropy_increases(make_fig=False):
    """
    Check that increasing epsilon increases blur in the
    transport plan.
    """
    epsilons = (0.01, 0.1, 1.0)

    margdiv = ForwardKL(1.0)
    x = np.linspace(-4, 4, 51)[:, None]
    y = np.linspace(-4, 4, 50)[:, None]
    a = np.squeeze(np.exp(-x ** 2))
    b = np.squeeze(np.exp(-y ** 2))

    a /= np.sum(a)
    b /= np.sum(b)

    # Fit transport plans.
    plans = []
    for eps in epsilons:
        plans.append(
            OTCost(margdiv, eps, 1e-6).fit(a, x, b, y).P_
        )

    # Test that the entropy of the optimal plan increases.
    entropies = [np.sum(-P * np.log(P + 1e-10) - P + 1) for P in plans]
    assert np.all(np.diff(entropies) > 0)

    if make_fig:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)
        for P, eps, ax in zip(plans, epsilons, axes):
            ax.imshow(P, aspect="auto")
            ax.set_title("eps = {}".format(eps))

        fig.set_size_inches((4, 2))
        fig.tight_layout()
        plt.show()


# @pytest.mark.parametrize('eps', [0.01, 0.1, 1.0])
# @pytest.mark.parametrize('tol', [1e-6])
# def test_balanced_duality_gap(eps, tol):
#     """
#     Check agreement between primal and dual objectives,
#     balanced transport case.
#     """
#     np.random.seed(1234)

#     margdiv = Balanced()
#     x = np.linspace(-4, 4, 51)[:, None]
#     y = np.linspace(-4, 4, 50)[:, None]
#     a = np.squeeze(np.exp(-x ** 2))
#     b = np.squeeze(np.exp(-y ** 2))

#     a /= a.sum()
#     b /= b.sum()

#     ot = OTCost(margdiv, eps, tol).fit(a, x, b, y)
#     assert_allclose(ot.primal_obj_, ot.dual_obj_, atol=1e-3)


@pytest.mark.parametrize('seed', [123])
@pytest.mark.parametrize('eps', [1.0])
@pytest.mark.parametrize('lam', [1000])  # <-- !! currently works for large lam, but not small !!
@pytest.mark.parametrize('b_mass', [1.0])
@pytest.mark.parametrize('tol', [1e-6])
def test_reference_implementation(seed, eps, lam, b_mass, tol):
    """
    Compare transport plan to Python Optimal Transpot (POT)
    library.
    """
    from ot.unbalanced import sinkhorn_stabilized_unbalanced
    rs = np.random.RandomState(seed)

    # Random locations for atoms.
    x = rs.randn(25, 1)
    y = rs.randn(24, 1)

    # Random mass vectors.
    a = np.random.rand(x.size)
    b = np.random.rand(y.size)

    # Normalize masses.
    a *= (1.0 / a.sum())
    b *= (b_mass / b.sum())

    # Fit OTCost, get transport plan
    margdiv = ForwardKL(lam)
    otcost = OTCost(margdiv, eps, tol).fit(a, x, b, y)

    # Fit with reference library.
    transport_plan = sinkhorn_stabilized_unbalanced(
        a, b, otcost.C_, eps, lam, numItermax=10000
    )

    # Assert optimal transport plans match.
    assert_allclose(otcost.P_, transport_plan, atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize('seed', [123])
@pytest.mark.parametrize('tol', [1e-6])
@pytest.mark.parametrize('eps', [1e-6])
def test_zero_cost(seed, eps, tol):
    """
    Assert cost is zero if epsilon and lambda penalties are both very small.
    In this case, an optimal transport plan could just be the zeros matrix.
    """

    rs = np.random.RandomState(seed)

    # Random locations for atoms.
    x = rs.randn(25, 1)
    y = rs.randn(24, 1)

    # Random mass vectors.
    a = np.random.rand(x.size)
    b = np.random.rand(y.size)

    # Normalize masses.
    a *= (1.0 / a.sum())
    b *= (1.0 / b.sum())

    # Fit model with very small marginal penalty
    margdiv = ForwardKL(1e-6)
    otcost = OTCost(margdiv, eps, tol).fit(a, x, b, y)

    # Assert cost is essentially zero.
    assert_allclose(otcost.primal_obj_, 0.0, atol=1e-5)
    assert_allclose(otcost.dual_obj_, 0.0, atol=1e-5)


@pytest.mark.parametrize('seed', [123])
@pytest.mark.parametrize('eps', [0.1, 1.0, 10])
@pytest.mark.parametrize('lam', [0.1, 1.0, 10])
@pytest.mark.parametrize('b_mass', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('tol', [1e-6])
def test_unbalanced_kl_duality_gap(seed, eps, lam, b_mass, tol):
    """
    Compare transport plan to Python Optimal Transpot (POT)
    library.
    """
    rs = np.random.RandomState(seed)

    # Random locations for atoms.
    x = rs.randn(25, 1)
    y = rs.randn(24, 1)

    # Random mass vectors.
    a = np.random.rand(x.size)
    b = np.random.rand(y.size)

    # Normalize masses.
    a *= (1.0 / a.sum())
    b *= (b_mass / b.sum())

    # Calculate OT cost.
    margdiv = ForwardKL(lam)
    otcost = OTCost(margdiv, eps, tol).fit(a, x, b, y)

    # Duality gap should be small.
    assert_allclose(otcost.primal_obj_, otcost.dual_obj_, atol=1e-4)


@pytest.mark.parametrize('seed', [123, 1234])
@pytest.mark.parametrize('eps', [0.1, 1.0, 10])
@pytest.mark.parametrize('lam', [0.1, 1.0, 10])
@pytest.mark.parametrize('b_mass', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('tol', [1e-6])
def test_ot_kl_gradients(seed, eps, lam, b_mass, tol):
    """
    Compare transport plan to Python Optimal Transpot (POT)
    library.
    """
    rs = np.random.RandomState(seed)

    # Random locations for atoms.
    x = rs.randn(25, 1)
    y = rs.randn(24, 1)

    # Random mass vectors.
    a = np.random.rand(x.size)
    b = np.random.rand(y.size)

    # Normalize masses.
    a *= (1.0 / a.sum())
    b *= (b_mass / b.sum())

    # Calculate OT cost.
    margdiv = ForwardKL(lam)
    otcost = OTCost(margdiv, eps, tol)

    # Fit OT cost, compute gradients for a and b.
    otcost.fit(a, x, b, y)
    grad_a = otcost.grad_a_.copy()
    grad_b = otcost.grad_b_.copy()

    # Compute gradient of a by finite differencing.
    def f(a_):
        otcost.fit(a_, x, b, y)
        return otcost.primal_obj_
    approx_grad_a = approx_fprime(a, f, np.sqrt(np.finfo(float).eps))

    # Check gradients approximately match finite differencing.
    assert_allclose(grad_a, approx_grad_a, atol=1e-4, rtol=1e-3)

    # Function to compute otcost given mass vector b.
    def g(b_):
        otcost.fit(a, x, b_, y)
        return otcost.primal_obj_
    approx_grad_b = approx_fprime(b, g, np.sqrt(np.finfo(float).eps))

    # Check gradients approximately match finite differencing.
    assert_allclose(grad_b, approx_grad_b, atol=1e-4, rtol=1e-3)
