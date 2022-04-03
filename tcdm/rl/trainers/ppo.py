# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from stable_baselines3 import PPO
from tcdm.rl.trainers.util import make_env, make_policy_kwargs, \
                                   InfoCallback, FallbackCheckpoint, \
                                   get_warm_start
from wandb.integration.sb3 import WandbCallback
from tcdm.rl.models.policies import ActorCriticPolicy
from tcdm.rl.trainers.eval import make_eval_env, EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def ppo_trainer(config, resume_model=None):
    total_timesteps = config.total_timesteps
    eval_freq = int(config.eval_freq // config.n_envs)
    save_freq = int(config.save_freq // config.n_envs)
    restore_freq = int(config.restore_checkpoint_freq // config.n_envs)
    n_steps = int(config.agent.params.n_steps // config.n_envs)
    multi_proc = bool(config.agent.multi_proc)
    env = make_env(multi_proc=multi_proc, **config.env)

    if resume_model:
        model = PPO.load(resume_model, env)
        model._last_obs = None
        reset_num_timesteps = False
        total_timesteps -= model.num_timesteps
        if total_timesteps <= 0:
            return model
    else:
        model = PPO(
                        ActorCriticPolicy, 
                        env, verbose=1, 
                        tensorboard_log=f"logs/", 
                        n_steps=n_steps, 
                        gamma=config.agent.params.gamma,
                        gae_lambda=config.agent.params.gae_lambda,
                        learning_rate=config.agent.params.learning_rate,
                        ent_coef=config.agent.params.ent_coef,
                        vf_coef=config.agent.params.vf_coef,
                        clip_range=config.agent.params.clip_range,
                        batch_size=config.agent.params.batch_size,
                        n_epochs=config.agent.params.n_epochs,
                        policy_kwargs=make_policy_kwargs(config.agent.policy_kwargs)
                    )
        # initialize the agent with behavior cloning if desired
        if config.agent.params.warm_start_mean:
            warm_start = get_warm_start(config.env)
            bias = torch.from_numpy(warm_start)
            model.policy.set_action_bias(bias)
        reset_num_timesteps = True
    
    # initialize callbacks and train
    eval_env = make_eval_env(multi_proc, config.n_eval_envs, **config.env)
    eval_callback = EvalCallback(eval_freq, eval_env)
    restore_callback = FallbackCheckpoint(restore_freq)
    log_info = InfoCallback()
    checkpoint = CheckpointCallback(save_freq=save_freq, save_path=f'logs/', 
                                    name_prefix='rl_models')
    wandb = WandbCallback(model_save_path="models/", verbose=2)
    return model.learn(
                        total_timesteps=total_timesteps,
                        callback=[log_info, eval_callback, checkpoint, restore_callback, wandb],
                        reset_num_timesteps=reset_num_timesteps
                      )
