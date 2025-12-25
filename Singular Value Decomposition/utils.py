import numpy as np


# LaTeX helper for matrices
def latex_matrix(M):
    if np.shape(M)[0] < 20 and np.shape(M)[1] < 20:
        return (
            r"\begin{bmatrix}"
            + r" \\ ".join([" & ".join(f"{x:.3f}" for x in row) for row in M])
            + r"\end{bmatrix}"
        )
    else:
        return r"\begin{bmatrix} \text{matrix too large to display} \end{bmatrix}"


def ugly_2d(
    nx=10,
    ny=10,
    xi=-2.0,
    xf=2.0,
    yi=-2.0,
    yf=2.0,
    sigmax=1.0,
    sigmay=1.0,
    x0=None,
    y0=None,
    skew=0.0,
    rotate=0.0,
):
    x = np.linspace(xi, xf, nx)
    y = np.linspace(yi, yf, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if x0 is None:
        x0 = 0.5 * (xi + xf)
    if y0 is None:
        y0 = 0.5 * (yi + yf)

    # shift
    dx = X - x0
    dy = Y - y0

    # rotation (break x/y symmetry)
    Xr = np.cos(rotate) * dx + np.sin(rotate) * dy
    Yr = -np.sin(rotate) * dx + np.cos(rotate) * dy

    # ugly envelope (non-elliptical, non-separable)
    envelope = np.exp(
        -(Xr**2 / (2 * sigmax**2) + Yr**2 / (2 * sigmay**2) + skew * Xr * Yr)
    )

    return envelope
