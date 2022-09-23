
# TCDM task suite

This codebase contains an implementation of the TCDM benchmark and PGDM learning framework from our paper, [Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps](https://pregrasps.github.io/). 

## Citations
If you found this code useful in any way, please cite our paper:
```
@InProceedings{dasari2022pgdm,
            title={Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps},
            author={Dasari, Sudeep and Gupta, Abhinav and Kumar, Vikash},
            journal={arXiv preprint arXiv:2209.11221},
            year={2022}
          }
}
```

## Requirements
* Our code has been primarily tested on Ubuntu 20, but it should work on other versions of Linux
* We strongly recommend using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for setting up the Python environment. 
* You will need a [wandb](https://wandb.ai/) account, as our code uses it for logging results. 

## Installation Instructions

Please run the following commands to install our codebase:
```
git clone --recurse-submodules git@github.com:facebookresearch/TCDM.git && cd TCDM
conda env create -f environment.yml && conda activate tcdm
pip install -r requirements.txt
python setup.py develop
export MUJOCO_GL=egl        # NOTE: YOU PROBABLY WANT TO ADD THIS TO .bashrc
```

You should now be able to import our environment suite (python code below) and train policies. Happy experimenting!
```
>>> from tcdm import suite
>>> env = suite.load('hammer', 'use1')
```

## Training Examples
The following examples will show how to train dexterous manipulation policies in TCDM environments, using our PGDM framework. 
* Simply running `python train.py` will train an agent on the `hammer-use1` task. You may view the results (including videos of behaviors) on wandb.
* To train other tasks, specify on command line: `python train.py env.name=<task name>`. Checkout [TASKS.md](trajectories/TASKS.md) for a full list of our tasks, alongside goal visualizations.
* Note that you can change many aspects of our training pipeline, simply by overriding our [default config](experiments/config.yaml) with command line arguments! We use [hydra](https://hydra.cc/) to handle this -- please read those docs for more information.

### Sweeping TCDM
The following code snippet will run a sweep across all tasks in TCDM and report the results to wandb. **Note: our code assumes you have access to a [slurm](https://slurm.schedmd.com/overview.html) or [ray](https://docs.ray.io/en/latest/cluster/index.html) cluster! For alternate launchers check [here](https://hydra.cc/docs/plugins/joblib_launcher/#internaldocs-banner)**

```
python train.py hydra/launcher=<slurm/ray> exp_name=tcdm_sweep wandb.project=tcdm env.name=headphones-pass1,elephant-pass1,eyeglasses-pass1,flute-pass1,banana-pass1,hand-inspect1,binoculars-pass1,stanfordbunny-inspect1,toruslarge-inspect1,alarmclock-see1,fryingpan-cook2,airplane-fly1,cup-drink1,scissors-use1,cup-pour1,mug-drink3,waterbottle-shake1,flashlight-on2,wineglass-toast1,piggybank-use1,wineglass-drink2,lightbulb-pass1,wineglass-drink1,mouse-use1,knife-chop1,airplane-pass1,duck-inspect1,hammer-use1,stamp-stamp1,train-play1,toothpaste-lift,watch-lift,toothbrush-lift,stapler-lift,mouse-lift,waterbottle-lift,spheremedium-lift,alarmclock-lift,flashlight-lift,duck-lift,dhand-waterbottle,dhand-alarmclock,dhand-cup,dhand-elephant,dhand-binoculars,dmanus-crackerbox,dmanus-coffeecan,spheremedium-relocate,door-open,hammer-strike -m
```

### Sweeping TCDM-30
Similarly, you can sweep across all tasks in the TCDM-30 subset:

```
python train.py hydra/launcher=<slurm/ray> exp_name=tcdm_30_sweep wandb.project=tcdm env.name=headphones-pass1,elephant-pass1,eyeglasses-pass1,flute-pass1,banana-pass1,hand-inspect1,binoculars-pass1,stanfordbunny-inspect1,toruslarge-inspect1,alarmclock-see1,fryingpan-cook2,airplane-fly1,cup-drink1,scissors-use1,cup-pour1,mug-drink3,waterbottle-shake1,flashlight-on2,wineglass-toast1,piggybank-use1,wineglass-drink2,lightbulb-pass1,wineglass-drink1,mouse-use1,knife-chop1,airplane-pass1,duck-inspect1,hammer-use1,stamp-stamp1,train-play1 -m
```

## Pre-Trained Policies
Pre-trained policies are available [here](https://github.com/pregrasps/pregrasps.github.io/raw/master/resources/pretrained_agents.tar.gz). Please refer to `rollout.py` for an example on how to load and use these policies.


## Contributions
If you're interested in contributing to this codebase, please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License
<REPONAME> is MIT licensed, as found in [LICENSE](LICENSE.md). However, it does rely on other libraries, including [object_sim](https://github.com/vikashplus/object_sim), which each have their respective licenses that must be followed.
