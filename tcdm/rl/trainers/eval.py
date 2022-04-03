# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time, copy, imageio, wandb, os
import numpy as np
from tcdm.rl.trainers.util import make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


def make_eval_env(multi_proc, n_eval_envs, **kwargs):
    """
    Corrects environment kwargs for evaluation condition
    """
    eval_args = copy.deepcopy(kwargs)
    eval_args['multi_proc'] = multi_proc
    eval_args['n_envs'] = n_eval_envs
    eval_args['vid_freq'] = None
    if 'rand_reset_prob' in eval_args['task_kwargs']:
        eval_args['task_kwargs']['rand_reset_prob'] = 0
    env = make_env(**eval_args)
    env.has_multiproc = multi_proc
    return env


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, eval_env, verbose=0, n_eval_episodes=25):
        super().__init__(verbose)
        self._vid_log_dir = os.path.join(os.getcwd(), 'eval_videos/')
        if not os.path.exists(self._vid_log_dir):
            os.makedirs(self._vid_log_dir)
        self._eval_freq = eval_freq
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes

    def _info_callback(self, locals, _):
        if locals['i'] == 0:
            env = locals['env']
            if env.has_multiproc:
                pipe = env.remotes[0]
                pipe.send(("render", "rgb_array"))
                render = pipe.recv()
            else:
                render = env.envs[0].render('rgb_array')
            self._info_tracker['rollout_video'].append(render)

        if locals['done']:
            for k, v in locals['info'].items():
                if isinstance(v, (float, int)):
                    if k not in self._info_tracker:
                        self._info_tracker[k] = []
                    self._info_tracker[k].append(v)

    def _on_step(self, fps=25) -> bool:
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            self._info_tracker = dict(rollout_video=[])
            start_time = time.time()
            episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self._eval_env,
                    n_eval_episodes=self._n_eval_episodes,
                    render=False,
                    deterministic=False,
                    return_episode_rewards=True,
                    warn=True,
                    callback=self._info_callback,
                )
            end_time = time.time()

            mean_reward, mean_length = np.mean(episode_rewards), np.mean(episode_lengths)
            self.logger.record('eval/time', end_time - start_time)
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_length', mean_length)
            for k, v in self._info_tracker.items():
                if k == 'rollout_video':
                    path = 'eval-call-{}.mp4'.format(self.n_calls)
                    path = os.path.join(self._vid_log_dir, path)
                    writer = imageio.get_writer(path, fps=fps)
                    for i in v:
                        writer.append_data(i)
                    writer.close()
                    wandb.log({'eval/rollout_video': wandb.Video(path)})
                else:
                    self.logger.record('eval/mean_{}'.format(k), np.mean(v))
            self.logger.dump(self.num_timesteps)
        return True
