# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pyquaternion import Quaternion


def to_quat(arr):
    if isinstance(arr, Quaternion):
        return arr.unit
    if len(arr.shape) == 2:
        return Quaternion(matrix=arr).unit
    elif len(arr.shape) == 1 and arr.shape[0] == 9:
        return Quaternion(matrix=arr.reshape((3,3))).unit
    return Quaternion(array=arr).unit


def rotation_distance(q1, q2):
    delta_quat = to_quat(q2) * to_quat(q1).inverse
    return np.abs(delta_quat.angle)


def root_to_point(root_pos, root_rotation, point):
    if isinstance(root_rotation, Quaternion) \
                  or root_rotation.shape != (3,3):
        root_rotation = to_quat(root_rotation).rotation_matrix
    root_rotation_inv = root_rotation.T
    delta = (point - root_pos).reshape((3,1))
    return root_rotation_inv.dot(delta).reshape(-1)


def to_transform_mat(R, t):
    pad = np.array([0, 0, 0, 1]).astype(np.float32).reshape((1, 4))
    Rt = np.concatenate((R, t.reshape((3, 1))), 1)
    return np.concatenate((Rt, pad), 0)


def axis_angle_to_rot(axis_angle):
    angle = max(1e-8, np.linalg.norm(axis_angle))
    axis = axis_angle / angle
    quat = Quaternion(axis=axis, angle=angle)
    return quat.rotation_matrix


class Pose(object):
    def __init__(self, pos, rotation):
        assert len(pos.shape) == 2, "pos should be batched"
        assert len(rotation.shape) >= 2, "rotation should be batched"
        assert pos.shape[0] == rotation.shape[0], "Batch sizes should match"
        self.pos = pos
        self.rot = rotation
    
    def __len__(self):
        return self.pos.shape[0]


class PoseAndVelocity(Pose):
    def __init__(self, pos, rotation, linear_vel, angular_vel):
        super().__init__(pos, rotation)
        assert len(linear_vel.shape) == 2, "linear_vel should be batched"
        assert len(angular_vel.shape) == 2, "angular_vel should be batched"
        assert pos.shape[0] == angular_vel.shape[0], "Batch sizes should match"
        assert linear_vel.shape[0] == angular_vel.shape[0], "Batch sizes should match"
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
