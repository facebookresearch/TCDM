# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from stable_baselines3 import SAC
from tcdm.rl.trainers.util import make_env, make_policy_kwargs, \
                                   InfoCallback, FallbackCheckpoint, \
                                   get_warm_start
from wandb.integration.sb3 import WandbCallback
from tcdm.rl.trainers.eval import make_eval_env, EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def sac_trainer(config, resume_model=None):
    total_timesteps = config.total_timesteps
    eval_freq = int(config.eval_freq // config.n_envs)
    save_freq = int(config.save_freq // config.n_envs)
    restore_freq = int(config.restore_checkpoint_freq // config.n_envs)
    multi_proc = bool(config.agent.multi_proc)
    env = make_env(multi_proc=multi_proc, **config.env)

    if resume_model:
        model = SAC.load(resume_model, env)
        model._last_obs = None
        reset_num_timesteps = False
        total_timesteps -= model.num_timesteps
        if total_timesteps <= 0:
            return model
    else:
        model = SAC(
                        'MlpPolicy', 
                        env, verbose=1, 
                        tensorboard_log=f"logs/", 
                        learning_rate=config.agent.params.learning_rate,
                        buffer_size=config.agent.params.buffer_size,
                        learning_starts=config.agent.params.learning_starts,
                        batch_size=config.agent.params.batch_size,
                        tau=config.agent.params.tau,
                        gamma=config.agent.params.gamma,
                        train_freq=config.agent.params.train_freq,
                        gradient_steps=config.agent.params.gradient_steps,
                        ent_coef=config.agent.params.ent_coef,
                        target_update_interval=config.agent.params.target_update_interval,
                        target_entropy=config.agent.params.target_entropy,
                        policy_kwargs=make_policy_kwargs(config.agent.policy_kwargs)
                    )
        
        # add an ortho initialization into the actor
        actor = model.policy.actor
        log_std_init = config.agent.policy_kwargs.get('log_std_init', 0)
        for layer, gain, fill in zip((actor.mu, actor.log_std), (1e-2, 1e-3), (0, log_std_init)):
            torch.nn.init.orthogonal_(layer.weight, gain)
            layer.bias.data.fill_(fill)

        # initialize the agent to hover around initial state if needed
        if config.agent.params.warm_start_mean:
            warm_start = get_warm_start(config.env)
            ac_bias = torch.from_numpy(warm_start)
            ac_bias = ac_bias.type(actor.mu.bias.dtype).to(actor.mu.bias.device)
            actor.mu.bias.data.copy_(ac_bias)
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
