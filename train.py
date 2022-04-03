# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import traceback
import hydra, os, wandb, yaml
from tcdm.rl import trainers
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f'{wandb_cfg.sweep_name_prefix}-{job_id}'
        notes = f'{override_dirname}'
    except:
        name, notes = None, None
    return wandb.init(
                        project=wandb_cfg.project,
                        config=job_config,
                        group=wandb_cfg.group,
                        sync_tensorboard=True, 
                        monitor_gym=True,
                        save_code=True,
                        name=name,
                        notes=notes,
                        id=run_id,
                        resume=run_id is not None
                  )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'experiments')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def train(cfg: DictConfig):
    try:
        cfg_yaml = OmegaConf.to_yaml(cfg)
        resume_model = cfg.resume_model
        if os.path.exists('exp_config.yaml'):
            old_config = yaml.load(open('exp_config.yaml', 'r'))
            params, wandb_id = old_config['params'], old_config['wandb_id']
            run = create_wandb_run(cfg.wandb, params, wandb_id)
            resume_model = 'restore_checkpoint.zip'
            assert os.path.exists(resume_model), 'restore_checkpoint.zip does not exist!'
        else:
            defaults = HydraConfig.get().runtime.choices
            params = yaml.safe_load(cfg_yaml)
            params['defaults'] = {k: defaults[k] for k in ('agent', 'env')}

            run = create_wandb_run(cfg.wandb, params)
            save_dict = dict(wandb_id=run.id, params=params)
            yaml.dump(save_dict, open('exp_config.yaml', 'w'))
            print('Config:')
            print(cfg_yaml)
        
        if cfg.agent.name == 'PPO':
            trainers.ppo_trainer(cfg, resume_model)
        else:
            raise NotImplementedError
        wandb.finish()
    except:
        traceback.print_exc(file=open('exception.log', 'w'))
        with open('exception.log', 'r') as f:
            print(f.read())


if __name__ == '__main__':
        train()
