# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from gym import core, spaces
import numpy as np
from dm_env import specs
import collections


def _spec_to_box(spec):
    """
    Helper function sourced from: https://github.com/denisyarats/dmc2gym
    """
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


class GymWrapper(core.Env):
    metadata = {"render.modes": ['rgb_array'], "video.frames_per_second": 25}

    def __init__(self, base_env):
        """
        Initializes 
        """
        self._base_env = base_env
        self._flat_dict = False

        # parses and stores action space
        self.action_space = _spec_to_box([base_env.action_spec()])
        # parses and stores (possibly nested) observation space
        if isinstance(base_env.observation_spec(), (dict, collections.OrderedDict)):
            obs_space = collections.OrderedDict()
            for k, v in base_env.observation_spec().items():
                obs_space[k] = _spec_to_box([v])
            self.observation_space = spaces.Dict(obs_space)
            if base_env.flat_obs:
                self.observation_space = self.observation_space['observations']
                self._flat_dict = True
        else:
            self.observation_space = _spec_to_box([base_env.observation_spec()])

    def reset(self):
        step = self._base_env.reset()
        obs = step.observation
        obs = obs['observations'] if self._flat_dict else obs
        return obs
    
    def step(self, action):
        step = self._base_env.step(action.astype(self.action_space.dtype))
        o = step.observation
        o = o['observations'] if self._flat_dict else o
        r = step.reward
        done = step.last()
        info = self._base_env.task.step_info
        return o, r, done, info

    def render(self, mode='rgb_array', height=240, width=320, camera_id=None):
        assert mode == 'rgb_array', "env only supports rgb_array rendering"
        if camera_id is None:
            camera_id = self._base_env.default_camera_id
        return self._base_env.physics.render(height=height, width=width, 
                                                camera_id=camera_id)

    @property
    def wrapped(self):
        return self._base_env
