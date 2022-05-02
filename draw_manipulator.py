import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def default_colormap():
    return colors.LinearSegmentedColormap.from_list(
        "custom",
        [[0.25098039215686274, 0.8235294117647058, 1.0],
         [0.10980392156862745, 0.3137254901960784, 0.8313725490196079]])


def draw_manipulator(model, q, **kwargs):
    spine = []
    for s in np.linspace(0, model.total_length):
        transform = model.forwardKin(q, s)
        spine.append(transform.pos)
    spine = np.stack(spine)
    plt.plot(spine[..., 0], spine[..., 1], **kwargs)

    contact_points = []
    for contact_point in model.contact_points:
        point = model.forwardKinPoint(q, contact_point)
        contact_points.append(point)
    contact_points = np.stack(contact_points)
    plt.plot(contact_points[..., 0], contact_points[..., 1], '.', **kwargs)
    plt.axis('square')
