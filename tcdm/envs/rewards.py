# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import abc
import copy
import numpy as np
from tcdm.motion_util import rotation_distance


def get_reward(name):
    if name == 'objectmimic':
        return ObjectMimic
    if name == 'dummy':
        return Dummy
    raise ValueError("Reward {} not supported!".format(name)) 


def norm2(x):
    return np.sum(np.square(x))


class _ErrorTracker:
    def __init__(self, targets, thresh=0.01, start=0):
        self._targets = targets.copy()
        self._values = np.zeros_like(self._targets)
        self._thresh, self._i = thresh, 0
        self._start = start
    
    def append(self, v):
        if self._i >= len(self._values):
            return
        self._values[self._i:] = v[None]
        self._i += 1
    
    @property
    def N(self):
        return len(self._targets) - self._start

    @property
    def error(self):
        v, t = self._values[self._start:], self._targets[self._start:]
        return np.linalg.norm(v - t) / self.N

    @property
    def success(self):
        v, t = self._values[self._start:], self._targets[self._start:]
        deltas = np.sqrt(np.sum(np.square(v - t), axis=-1))
        if len(deltas.shape) > 1:
            deltas = np.mean(deltas.reshape((self.N, -1)), axis=1)
        return np.mean(deltas <= self._thresh)


class RewardFunction(abc.ABC):
    def __init__(self, **override_hparams):
        """
        Overrides default hparams with values passed into initializer
        """
        params = copy.deepcopy(self.DEFAULT_HPARAMS)
        for k, v in override_hparams.items():
            assert k in params, "Param {} does not exist in struct".format(k)
            params[k] = v
        
        for k, v in params.items():
            setattr(self, k, v)
    
    @abc.abstractproperty
    def DEFAULT_HPARAMS(self):
        """
        Returns default hyperparamters for reward function
        """
    
    def initialize_rewards(self, parent_task, physics):
        """
        Gets parent task and sets constants as required from it
        """
        self._parent_task = parent_task

    @abc.abstractmethod
    def get_reward(self, physics):
        """
        Calculates reward and success stats from phyiscs data
        Returns reward, info_dict
        """

    @abc.abstractmethod
    def check_termination(self, physics):
        """
        Checks if trajectory should terminate
        Returns terminate_flag
        """
    
    def __call__(self, physics):
        return self.get_reward(physics)


class Dummy(RewardFunction):
    @property
    def DEFAULT_HPARAMS(self):
        return dict()
    
    def get_reward(self, _):
        return 0.0, dict()
    
    def check_termination(self, _):
        return False


class ObjectMimic(RewardFunction):
    @property
    def DEFAULT_HPARAMS(self):
        return  {
                    'obj_err_scale': 50,
                    'object_reward_scale': 10,
                    'lift_bonus_thresh': 0.02,
                    'lift_bonus_mag': 2.5,
                    'obj_com_term': 0.25,
                    'n_envs': 1,
                    'obj_reward_ramp': 0,
                    'obj_reward_start': 0
                }
    
    def initialize_rewards(self, parent_task, physics):
        self._step_count = parent_task.step_count
        self._reference_motion = parent_task.reference_motion
        self._object_name = self._reference_motion.object_name
        floor_z = physics.named.data.xipos[self._object_name][2]
        self._lift_z = floor_z + self.lift_bonus_thresh

        # register metric tracking data
        self._obj = _ErrorTracker(self._reference_motion['object_translation'][1:])

    def get_reward(self, physics):
        # get targets from reference object
        tgt_obj_com = self._reference_motion.object_pos
        tgt_obj_rot = self._reference_motion.object_rot

        # get real values from physics object
        obj_com = physics.named.data.xipos[self._object_name].copy()
        obj_rot = physics.named.data.xquat[self._object_name].copy()
        self._obj.append(obj_com)

        # calculate both object "matching" reward and lift bonus
        obj_com_err = np.sqrt(norm2(tgt_obj_com - obj_com))
        obj_rot_err = rotation_distance(obj_rot, tgt_obj_rot) / np.pi
        obj_reward = np.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))
        lift_bonus = (tgt_obj_com[2] >= self._lift_z) and (obj_com[2] >= self._lift_z)

        obj_scale = self._object_reward_scale()
        reward = obj_scale * obj_reward + self.lift_bonus_mag * float(lift_bonus)

        # populate info dict
        info = {
            'time_frac': self._reference_motion.time,
            'obj_err': self._obj.error,
            'obj_success': self._obj.success,
            'step_obj_err': obj_com_err
        }
        info['obj_err_scale'] = obj_scale / self.object_reward_scale \
                                if self.object_reward_scale else 0
        return reward, info

    def check_termination(self, physics):
        # terminate if object delta greater than threshold
        tgt_obj_com = self._reference_motion.object_pos
        obj_com = physics.named.data.xipos[self._object_name].copy()
        return norm2(obj_com - tgt_obj_com) >= self.obj_com_term ** 2

    def _object_reward_scale(self):
        if self.obj_reward_ramp > 0:
            delta = self._step_count * self.n_envs - self.obj_reward_start
            delta /= float(self.obj_reward_ramp)
        else:
            delta = 1.0 if self._step_count >= self.obj_reward_start \
                    else 0.0
        return self.object_reward_scale * min(max(delta, 0), 1)
