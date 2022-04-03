# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import copy, collections
from dm_env import specs
from dm_control.rl import control


def _denormalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip(action, -1, 1) * ac_range + ac_mid


def _normalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip((action - ac_mid) / ac_range, -1, 1)


class Environment(control.Environment):
    def __init__(self, physics, task, default_camera_id=0, **kwargs):
        self._default_camera_id = default_camera_id
        super().__init__(physics, task, **kwargs)

    def get_state(self):
        return dict(physics=self.physics.get_state(), 
                    task=self.task.get_state())
    
    def set_state(self, state):
        self.physics.set_state(state['physics'])
        self.task.set_state(state['task'])

    @property
    def flat_obs(self):
        return self._flat_observation
    
    @property
    def default_camera_id(self):
        return self._default_camera_id


class Task(control.Task):
    def __init__(self, reward_fns, reward_weights=None, random=None):
        # initialize 
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self._info = {}

        # store reward functions and weighting terms
        self._step_count = 0
        self._reward_fns = copy.deepcopy(reward_fns)
        reward_wgts = [1.0 for _ in self._reward_fns] if reward_weights is None \
                     else reward_weights
        self._reward_wgts = copy.deepcopy(reward_wgts)

    @property
    def random(self):
        """Task-specific `numpy.random.RandomState` instance."""
        return self._random

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray((physics.adim,), np.float32, -1, 1)

    def initialize_episode(self, physics):
        """ Sets the state of the environment at the start of each episode.
            Called by `control.Environment` at the start of each episode *within*
            `physics.reset_context()` (see the documentation for `base.Physics`)

        Args:
            physics: An instance of `mujoco.Physics`.
        """
        # initialize info dict and rewards
        self._info = {}
        self.initialize_rewards(physics)
    
    def initialize_rewards(self, physics):
        """ Initializes reward function objects with necessarily data/objects in task
        
        Args:
            physics: An instance of `mujoco.Physics`
        """
        for reward_fn in self._reward_fns:
            reward_fn.initialize_rewards(self, physics)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        self._step_count += 1
        action = _denormalize_action(physics, action)
        physics.set_control(action)

    def after_step(self, physics):
        """Called immediately after environment step: no-op by default"""

    def get_observation(self, physics):
        """Returns a default observation of current physics state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos.astype(np.float32).copy()
        obs['velocity'] = physics.data.qvel.astype(np.float32).copy()
        motor_joints = physics.data.qpos[:physics.adim]
        obs['zero_ac'] = _normalize_action(physics, motor_joints)
        return obs
    
    @property
    def step_info(self):
        """Compatability function to pipe extra step information for gym compat """
        return self._info

    @property
    def step_count(self):
        return self._step_count

    def get_reward(self, physics):
        reward = 0
        for reward_fn, lambda_r in zip(self._reward_fns, self._reward_wgts):
            r_i, info_i = reward_fn(physics)
            reward += lambda_r * r_i
            self._info.update(info_i)
        return reward

    def get_termination(self, physics):
        for reward_fn in self._reward_fns:
            if reward_fn.check_termination(physics):
                return 0.0
        return None


class SingleObjectTask(Task):
    def __init__(self, object_name, reward_fns, reward_weights=None, random=None):
        self._object_name = object_name
        super().__init__(reward_fns, reward_weights=reward_weights, random=random)
    
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        base_pos = obs['position']
        base_vel = obs['velocity']

        hand_poses = physics.body_poses
        hand_com = hand_poses.pos.reshape((-1, 3))
        hand_rot = hand_poses.rot.reshape((-1, 4))
        hand_lv = hand_poses.linear_vel.reshape((-1, 3))
        hand_av = hand_poses.angular_vel.reshape((-1, 3))
        hand_vel = np.concatenate((hand_lv, hand_av), 1)

        object_name = self.object_name
        obj_com = physics.named.data.xipos[object_name].copy()
        obj_rot = physics.named.data.xquat[object_name].copy()
        obj_vel = physics.data.object_velocity(object_name, 'body')
        obj_vel = obj_vel.reshape((1, 6))
        
        full_com = np.concatenate((hand_com, obj_com.reshape((1,3))), 0)
        full_rot = np.concatenate((hand_rot, obj_rot.reshape((1,4))), 0)
        full_vel = np.concatenate((hand_vel, obj_vel), 0)

        obs['position'] = np.concatenate((base_pos, full_com.reshape(-1), 
                                          full_rot.reshape(-1))).astype(np.float32)
        obs['velocity'] = np.concatenate((base_vel, 
                                          full_vel.reshape(-1))).astype(np.float32)
        obs['state'] = np.concatenate((obs['position'], obs['velocity']))
        return obs
    
    @property
    def object_name(self):
        return self._object_name

class ReferenceMotionTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns, init_key,
                       reward_weights=None, random=None):
        self.reference_motion =reference_motion
        self._init_key = init_key
        object_name = reference_motion.object_name
        super().__init__(object_name, reward_fns, reward_weights, random)

    def initialize_episode(self, physics):
        start_state = self.reference_motion.reset()[self._init_key]
        with physics.reset_context():
            physics.data.qpos[:] = start_state['position']
            physics.data.qvel[:] = start_state['velocity']
        return super().initialize_episode(physics)

    def before_step(self, action, physics):
        super().before_step(action, physics)
        self.reference_motion.step()

    def get_termination(self, physics):
        if self.reference_motion.next_done:
            return 0.0
        return super().get_termination(physics)

    @property
    def substeps(self):
        return self.reference_motion.substeps

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['goal'] = self.reference_motion.goals.astype(np.float32)
        obs['state'] = np.concatenate((obs['state'], obs['goal']))
        return obs
