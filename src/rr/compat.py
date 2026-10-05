"""Runtime compatibility shims for the vendored Chameleon code (applied before loading a model).

SciPy >= 1.14 removed `scipy.interpolate.interp2d`, which `model/transformers/helpers.py`
(`resize_rel_pos_bias`) calls to resize BEiT's relative-position tables whenever img_size != 224.
On the regular grids used there, SciPy's documented drop-in replacement is RectBivariateSpline
(s = 0, same cubic B-spline): interp2d(x, y, z)(xn, yn) == RectBivariateSpline(y, x, z)(yn, xn).
"""

import numpy as np


class _Interp2d:
    def __init__(self, x, y, z, kind='cubic'):
        from scipy.interpolate import RectBivariateSpline
        k = {'linear': 1, 'cubic': 3, 'quintic': 5}[kind]
        self._spline = RectBivariateSpline(np.asarray(y, float), np.asarray(x, float), np.asarray(z, float), kx=k, ky=k)

    def __call__(self, xn, yn):
        return self._spline(np.asarray(yn, float), np.asarray(xn, float))


def patch_scipy_interp2d():
    import scipy.interpolate as si
    try:
        si.interp2d([0, 1, 2, 3], [0, 1, 2, 3], np.zeros((4, 4)), kind='cubic')
    except (NotImplementedError, AttributeError):
        si.interp2d = _Interp2d
        return True
    return False
