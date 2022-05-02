import numpy as np
import quaternion


class DimensionError(Exception):

    def __init__(self, ndim):
        super().__init__(
            f"You seem to be working in {ndim} dimensions. We can't help you.")


class Transform2D:

    def __init__(self, pos, orientation):
        self.pos = pos
        self.orientation = orientation

    def __mul__(self, other):
        if isinstance(other, Transform2D):
            return Transform2D(
                self.pos + rotationMatrix(self.orientation).dot(other.pos),
                self.orientation + other.orientation)
        return self.pos + rotationMatrix(self.orientation).dot(other)


class Transform3D:

    def __init__(self, pos, orientation):
        self.pos = pos
        self.orientation = orientation

    def __mul__(self, other):
        if isinstance(other, Transform3D):
            return Transform3D(self.pos + quaternion.rotate_vectors(other.pos),
                               self.orientation * other.orientation)
        return self.pos + quaternion.rotate_vectors(other)


def makeTransform(ndim, pos, orientation):
    if ndim == 2:
        return Transform2D(pos, orientation)
    if ndim == 3:
        return Transform3D(pos, orientation)
    raise DimensionError(ndim)


def identityTransform(ndim):
    if ndim == 2:
        return Transform2D(np.zeros(ndim), 0)
    if ndim == 3:
        return Transform2D(np.zeros(ndim),
                           quaternion.from_rotation_vector(np.zeros(3)))
    raise DimensionError(ndim)


def rotationMatrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


def transform(transform, frame):
    t_pos, t_angle = transform
    f_pos, f_angle = frame
    return t_pos + rotationMatrix(t_angle).dot(f_pos), t_angle + f_angle


def transformPoint(transform, point):
    t_pos, t_angle = transform
    return t_pos + rotationMatrix(t_angle).dot(point)