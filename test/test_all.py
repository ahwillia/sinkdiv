import pytest
import numpy as np
from numpy.testing import assert_allclose
from sinkdiv import OTCost, ForwardKL, Balanced


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


@pytest.mark.parametrize('eps', [0.01, 0.1, 1.0])
@pytest.mark.parametrize('tol', [1e-6])
def test_balanced_duality_gap(eps, tol):
    """
    Check agreement between primal and dual objectives,
    balanced transport case.
    """
    np.random.seed(1234)

    margdiv = Balanced()
    x = np.linspace(-4, 4, 51)[:, None]
    y = np.linspace(-4, 4, 50)[:, None]
    a = np.squeeze(np.exp(-x ** 2))
    b = np.squeeze(np.exp(-y ** 2))

    a /= a.sum()
    b /= b.sum()

    ot = OTCost(margdiv, eps, tol).fit(a, x, b, y)
    assert_allclose(ot.primal_obj_ - ot.dual_obj_, 0.0, atol=1e-7)


@pytest.mark.parametrize('seed', [123])
@pytest.mark.parametrize('eps', [1.0])
@pytest.mark.parametrize('lam', [1, 1000])  # <-- !! currently works for large lam, but not small !!
@pytest.mark.parametrize('b_mass', [1.0])
@pytest.mark.parametrize('tol', [1e-6])
def test_reference_implementation(seed, eps, lam, b_mass, tol):
    """
    Compare transport plan to Python Optimal Transpot (POT)
    library.
    """
    from ot.unbalanced import sinkhorn_knopp_unbalanced
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
    transport_plan = sinkhorn_knopp_unbalanced(
        a, b, otcost.C_, eps, lam
    )

    # Assert optimal transport plans match.
    assert_allclose(otcost.P_, transport_plan, atol=1e-5, rtol=1e-2)


# def test_unbalanced_kl_duality_gap():
#     """
#     Check that primal and dual match.
#     """
#     np.random.seed(1234)

#     margdiv = ForwardKL(1.0)
#     x = np.linspace(-4, 4, 51)[:, None]
#     y = np.linspace(-4, 4, 50)[:, None]
#     a = np.squeeze(np.exp(-x ** 2))
#     b = np.squeeze(np.exp(-y ** 2))

#     a /= a.sum()
#     b /= b.sum()

#     print("DUALITY GAP SHOULD BE ZERO...")
#     for eps in (0.01, 0.1, 1.0):
#         ot = OTCost(margdiv, eps, 1e-6).fit(a, x, b, y)
#         print("EPS = {}; Duality Gap = {}".format(
#             eps, ot.primal_obj_ - ot.dual_obj_))

#     print("\nDUAL OBJECTIVE SHOULD BE NONNEGATIVE...")
#     for eps in (0.01, 0.1, 1.0):
#         ot = OTCost(margdiv, eps, 1e-6).fit(a, x, b, y)
#         print("EPS = {}; Dual Objective = {}".format(
#             eps, ot.dual_obj_))


