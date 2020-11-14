import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


def make_pinwheel_data(
        num_classes, num_per_class,
        warp_rate=0.8, warp_scale=np.pi/8
    ):

    # Sample multivariate Gaussians.
    Xs = []
    for ang in np.linspace(0, 2*np.pi, num_classes, endpoint=False):
        Q = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)]
        ])
        X = npr.randn(num_per_class, 2) 
        X *= np.array([0.15, 1.0])[None, :]
        X += np.array([0.0, 3.0])[None, :]
        Xs.append(X @ Q)

    # Warp samples by additional rotation, radially weighted.
    Xs = np.row_stack(Xs)
    for i, x in enumerate(Xs):
        ang = warp_scale * np.tanh((np.linalg.norm(x) - 3.0) * warp_rate)
        Q = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)]
        ])
        Xs[i] = Q @ x

    return Xs


if __name__ == "__main__":
    Xs = make_pinwheel_data(5, 200)
    plt.scatter(Xs[:, 0], Xs[:, 1], lw=0, s=10, color="k")
    plt.show()
