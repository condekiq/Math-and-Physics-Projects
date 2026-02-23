"""Utilities for the SVD notebook: LaTeX matrix formatting and 2D envelope generation."""

import numpy as np


def latex_matrix(M):
    """
    Format a 2D array as a LaTeX bmatrix string for display in Jupyter (e.g. with display(Math(...))).
    Matrices larger than 20x20 are shown as a placeholder.
    """
    M = np.asarray(M)
    rows, cols = M.shape
    if rows < 20 and cols < 20:
        return (
            r"\begin{bmatrix}"
            + r" \\ ".join([" & ".join(f"{x:.3f}" for x in row) for row in M])
            + r"\end{bmatrix}"
        )
    return r"\begin{bmatrix} \text{matrix too large to display} \end{bmatrix}"


def gaussian_2d_envelope(
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
    """
    Build a 2D Gaussian-like envelope on a grid (optionally rotated and skewed).

    Parameters
    ----------
    nx, ny : int
        Grid size.
    xi, xf, yi, yf : float
        Domain [xi, xf] x [yi, yf].
    sigmax, sigmay : float
        Widths along the (rotated) axes.
    x0, y0 : float or None
        Center; if None, use domain center.
    skew : float
        Cross term coefficient (breaks separability).
    rotate : float
        Rotation angle in radians.

    Returns
    -------
    envelope : ndarray, shape (nx, ny)
        Non-negative envelope values.
    """
    x = np.linspace(xi, xf, nx)
    y = np.linspace(yi, yf, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if x0 is None:
        x0 = 0.5 * (xi + xf)
    if y0 is None:
        y0 = 0.5 * (yi + yf)

    dx = X - x0
    dy = Y - y0

    # Rotate (breaks x/y symmetry for visualization)
    x_rot = np.cos(rotate) * dx + np.sin(rotate) * dy
    y_rot = -np.sin(rotate) * dx + np.cos(rotate) * dy

    # Gaussian envelope with optional skew term
    envelope = np.exp(
        -(x_rot**2 / (2 * sigmax**2) + y_rot**2 / (2 * sigmay**2) + skew * x_rot * y_rot)
    )
    return envelope


# Backward compatibility for the notebook
ugly_2d = gaussian_2d_envelope
