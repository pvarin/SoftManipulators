import numpy as np
from SmoothManipulatorBase import SmoothManipulatorBase


def _cinc(theta):
    """A well-behaved version of (1-cos(x))/x.

    The function (1-cos(x))/x is undefined at the origin, but the limit
    exists. When evaluating this function near zero we use the
    appropriate Taylor series approximation. 

    The third order taylor expansion is:
        (1-cos(x))/x = x/2 + x^3/24 + O(x^5)
        
    When the third order term is less than machine precision we can 
    safely use the second order taylor expansion. This happens when
    eps > x^3/24.  Or equivalently when x < cbrt(24*x). For double 
    precision numbers, this is approximately 1.7x10^-5.
    """
    eps = np.finfo(float).eps
    threshold = np.cbrt(24.0 * eps)
    if np.isscalar(theta):
        if np.abs(theta) < threshold:
            return theta / 2  # Second order taylor expansion
        else:
            return (1 - np.cos(theta)) / theta
    else:
        bad_indices = np.nonzero(np.abs(theta) < threshold)
        good_indices = np.logical_not(bad_indices)
        res = np.empty_like(theta)
        res[good_indices] = (1 -
                             np.cos(theta[good_indices])) / theta[good_indices]
        res[bad_indices] = theta[bad_indices]  # Second order taylor expansion.
        return res


class PiecewiseConstantCurvaturePath(SmoothManipulatorBase):

    def pos_relative_to_parent(self, curvature, length_along_segment):
        angle = curvature * length_along_segment
        x = length_along_segment * np.sinc(angle / np.pi)
        y = length_along_segment * _cinc(angle)
        return np.stack([x, y])

    def angle_relative_to_parent(self, curvature, length_along_segment):
        return curvature * length_along_segment
