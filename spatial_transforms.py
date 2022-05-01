import numpy as np


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