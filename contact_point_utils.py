import numpy as np


def make_evenly_spaced_contact_points(start, end, offset_fn, N=10):
    s = np.linspace(start, end, N)
    offset = offset_fn(s)
    return np.stack([s, offset], axis=-1)


def get_taper_fn(start_width, end_width, total_length):

    def taper_fn(s):
        return start_width - (start_width - end_width) * s / total_length

    return taper_fn