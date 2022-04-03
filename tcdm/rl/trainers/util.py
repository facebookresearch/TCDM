# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools, gym
import numpy as np
from tcdm import suite
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, \
                                             VecVideoRecorder


class InfoCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        all_keys = {}
        for info in self.model.ep_info_buffer:
            for k, v in info.items():
                if k in ('r', 't', 'l'):
                    continue
                elif k not in all_keys:
                    all_keys[k] = []
                all_keys[k].append(v)

        for k, v in all_keys.items():
            self.model.logger.record(f'env/{k}', np.mean(v))
    
    def _on_step(self) -> bool:
        return True


class _ObsExtractor(gym.Wrapper):
    def __init__(self, env, state_keyword):
        super().__init__(env=env)
        self._state_keyword = state_keyword
        self.observation_space = env.observation_space[state_keyword]
    
    def step(self, action):
        o, r, done, info = self.env.step(action)
        return o[self._state_keyword], r, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[self._state_keyword]


class FallbackCheckpoint(BaseCallback):
    def __init__(self, checkpoint_freq=1, verbose=0):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0 or self.n_calls <= 1:
            self.model.save('restore_checkpoint')
        return True


def _env_maker(name, task_kwargs, env_kwargs, info_keywords, state_keyword):
    np.random.seed()
    domain, task = name.split('-')
    env = suite.load(domain, task, OmegaConf.to_container(task_kwargs), 
                     dict(env_kwargs), gym_wrap=True)
    env = Monitor(env, info_keywords=tuple(info_keywords))
    env = _ObsExtractor(env, state_keyword)
    return env


def make_env(multi_proc, n_envs, vid_freq, vid_length, **kwargs):
    env_maker = functools.partial(_env_maker, **kwargs)
    if multi_proc:
        env = SubprocVecEnv([env_maker for _ in range(n_envs)])
    else:
        env = DummyVecEnv([env_maker for _ in range(n_envs)])

    if vid_freq is not None:
        vid_freq = max(int(vid_freq // n_envs), 1)
        trigger = lambda x: x % vid_freq == 0 or x <= 1
        env = VecVideoRecorder(env, "videos/", 
                            record_video_trigger=trigger, 
                            video_length=vid_length)
    return env


def get_warm_start(env_args):
    e = _env_maker(env_args.name, env_args.task_kwargs, 
                   env_args.env_kwargs, env_args.info_keywords,
                   'zero_ac')
    zero_ac = e.reset().copy().astype(np.float32)
    zero_ac[1] += 0.2; zero_ac[6:] += 0.2
    return np.clip(zero_ac, -1, 1)


def make_policy_kwargs(policy_config):
    return OmegaConf.to_container(policy_config)
