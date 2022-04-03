# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import numpy as np
from tcdm.motion_util import Pose, PoseAndVelocity


class HandReferenceMotion(object):
    def __init__(self, motion_file, start_step=0):
        self._load_motion(motion_file)
        self._substeps = int(self._reference_motion['SIM_SUBSTEPS'])                
        self._data_substeps = self._reference_motion.get('DATA_SUBSTEPS', self._substeps)              
        self._step, self._start_step = 0, int(start_step)
    
    def _load_motion(self, motion_file):
        motion_file = np.load(motion_file, allow_pickle=True)
        self._reference_motion =  {k:v for k, v in motion_file.items()}
        self._reference_motion['s_0'] = self._reference_motion['s_0'][()]

    def reset(self):
        self._step = 0
        return copy.deepcopy(self._reference_motion['s_0'])

    def step(self):
        self._check_valid_step()
        self._step += self._data_substeps
    
    def revert(self):
        self._step -= self._data_substeps
        self._check_valid_step()

    def __len__(self):
        return self.length

    @property
    def t(self):
        return self._step
    
    @property
    def data_substep(self):
        return self._data_substeps

    @property
    def time(self):
        return float(self._step) / self.length

    @property
    def qpos(self):
        self._check_valid_step()
        return self._reference_motion['s'][self._step].copy()

    @property
    def qvel(self):
        self._check_valid_step()
        return self._reference_motion['sdot'][self._step].copy()

    @property
    def eef_pos(self):
        self._check_valid_step()
        return self._reference_motion['eef_pos'][self._step].copy()
    
    @property
    def human_joint_coords(self):
        self._check_valid_step()
        return self._reference_motion['human_joint_coords'][self._step].copy()

    @property
    def eef_quat(self):
        self._check_valid_step()
        return self._reference_motion['eef_quat'][self._step].copy()

    @property
    def eef_linear_velocity(self):
        self._check_valid_step()
        return self._reference_motion['eef_velp'][self._step].copy()

    @property
    def eef_angular_velocity(self):
        self._check_valid_step()
        return self._reference_motion['eef_velr'][self._step].copy()

    @property
    def body_poses(self):
        pos = self.eef_pos
        rot = self.eef_quat
        lv = self.eef_linear_velocity
        av = self.eef_angular_velocity
        return PoseAndVelocity(pos, rot, lv, av)

    @property
    def substeps(self):
        return self._substeps

    @property
    def done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length
    
    @property
    def next_done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length - self._data_substeps

    @property
    def n_left(self):
        assert self._step is not None, "Motion must be reset before lengths calculated"
        n_left = (self.length - self._step) / float(self._data_substeps) - 1
        return int(max(n_left, 0))
    
    @property
    def n_steps(self):
        n_steps = self.length / float(self._data_substeps) - 1
        return int(max(n_steps, 0))

    @property
    def length(self):
        if 'length' in self._reference_motion:
            return self._reference_motion['length']
        return self._reference_motion['s'].shape[0]
    
    @property
    def start_step(self):
        return self._start_step

    def _check_valid_step(self):
        assert not self.done, "Attempting access data and/or step 'done' motion"
        assert self._step >= self._start_step, "step must be at least start_step"
    
    def __getitem__(self, key):
        value =  copy.deepcopy(self._reference_motion[key])
        if not isinstance(value, np.ndarray):
            return value
        if len(value.shape) >= 2:
            return value[self._start_step::self._data_substeps]
        return value


class HandObjectReferenceMotion(HandReferenceMotion):
    def __init__(self, object_name, motion_file):
        super().__init__(motion_file)
        self._object_name = object_name

    @property
    def object_name(self):
        return self._object_name

    @property
    def object_pos(self):
        self._check_valid_step()
        return self._reference_motion['object_translation'][self._step].copy()
    
    @property
    def floor_z(self):
        return float(self._reference_motion['object_translation'][0,2])
    
    @property
    def object_rot(self):
        self._check_valid_step()
        return self._reference_motion['object_orientation'][self._step].copy()
    
    @property
    def object_pose(self):
        pos = self.object_pos[None]
        rot = self.object_rot[None]
        return Pose(pos, rot)

    @property
    def goals(self):
        g = []
        for i in [1, 5, 10]:
            i = min(self._step + i, self.length-1)
            for k in ('object_orientation', 'object_translation'):
                g.append(self._reference_motion[k][i].flatten())
        return np.concatenate(g)
