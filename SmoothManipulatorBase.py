from abc import abstractmethod
import numpy as np
from spatial_transforms import transform, transformPoint

class SmoothManipulatorBase:

    def __init__(self, lengths, contact_points = None):
        self.lengths = lengths
        if contact_points is None:
            contact_points = []
        self.contact_points = contact_points

    def get_segment_index_and_residual_length(self, s):
        cumulative_length = 0
        for i, l in self.lengths:
            if s <= cumulative_length + l:
                return i, cumulative_length - s
            cumulative_length += l
        raise ValueError(
            'length is greater than the total segment length {cumulative_length}'
        )

    @abstractmethod
    def pos_relative_to_parent(self, curvature, length_along_segment):
        pass

    @abstractmethod
    def angle_relative_to_parent(self, curvature, length_along_segment):
        pass

    @property
    def total_length(self):
        if not hasattr(self, '_total_length'):
            self._total_length = np.sum(self.lengths)
        return self._total_length
    
    def transform_relative_to_parent(self, curvature, length_along_segment):
        return self.pos_relative_to_parent(curvature, length_along_segment), \
               self.angle_relative_to_parent(curvature, length_along_segment)

    def forwardKin(self, q, s):
        s_remaining = s.copy()
        ee_pos = np.zeros(2)
        ee_angle = 0
        T_root_parent = (ee_pos, ee_angle)
        for i, l, in enumerate(self.lengths):
            if s_remaining <= l:
                T_parent_child = self.transform_relative_to_parent(
                    q[i], s_remaining)
                T_root_parent = transform(T_root_parent, T_parent_child)
                break
            else:
                T_parent_child = self.transform_relative_to_parent(q[i], l)
                T_root_parent = transform(T_root_parent, T_parent_child)
                s_remaining -= l
        return T_root_parent

    def forwardKinPoint(self, q, pt):
        s = pt[...,0]
        offset = pt[...,1]

        T_root_spine = self.forwardKin(q, s)
        return transformPoint(T_root_spine, np.stack([np.zeros_like(offset), offset]))
        
