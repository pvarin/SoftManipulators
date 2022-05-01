import numpy as np
from PiecewiseConstantCurvatureModel import PiecewiseConstantCurvaturePath
import matplotlib.pyplot as plt
from draw_manipulator import draw_manipulator, default_colormap
from contact_point_utils import make_evenly_spaced_contact_points, get_taper_fn

# Some model parameters.
n_contacts_per_side = 10
lengths = np.array([np.pi, np.pi])

# Generate the contact points on the top and bottom.
total_length = np.sum(lengths)
top_taper = get_taper_fn(0.5, 0.2, total_length)
bottom_taper = get_taper_fn(-0.5, -0.3, total_length)
top_contacts = make_evenly_spaced_contact_points(0,
                                                 total_length,
                                                 top_taper,
                                                 N=n_contacts_per_side)
bottom_contacts = make_evenly_spaced_contact_points(0,
                                                    2 * np.pi,
                                                    bottom_taper,
                                                    N=n_contacts_per_side)
contact_points = np.concatenate([top_contacts, bottom_contacts])

# Build the model
model = PiecewiseConstantCurvaturePath(lengths, contact_points=contact_points)

# Plot the model in a sequence of poses
n_states = 3
cmap = default_colormap()
for scale in np.linspace(0, 1, n_states):
    q = scale * np.array([1.0, -1.0, 0.5])
    draw_manipulator(model, q, color=cmap(scale + (1 - scale) * 0.1))
plt.show()