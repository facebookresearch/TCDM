# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
from tcdm import suite
from stable_baselines3 import PPO
from argparse import ArgumentParser


"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--save_folder', default='pretrained_agents/hammer_use1/', 
                                     help="Save folder containing agent checkpoint/config")
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")


def render(writer, physics, AA=2, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def rollout(save_folder, writer):
    # get experiment config
    config =  yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    
    # build environment and load policy
    o, t = config['env']['name'].split('-')
    env = suite.load(o, t, config['env']['task_kwargs'], gym_wrap=True)
    policy = PPO.load(os.path.join(save_folder, 'checkpoint.zip'))

    # rollout the policy and print total reward
    s, done, total_reward = env.reset(), False, 0
    render(writer, env.wrapped.physics)
    while not done:
        action, _ = policy.predict(s['state'], deterministic=True)
        s, r, done, __ = env.step(action)
        render(writer, env.wrapped.physics)
        total_reward += r
    print('Total reward:', total_reward)


if __name__ == "__main__":
    args = parser.parse_args()

    # configure writer
    if args.render:
        writer = imageio.get_writer('rollout.mp4', fps=25)
        rollout(args.save_folder, writer)
        writer.close()
    else:
        rollout(args.save_folder, None)
