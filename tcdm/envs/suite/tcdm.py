# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import yaml
import numpy as np
from tcdm.envs import mj_models, traj_abspath
from tcdm.envs import asset_abspath
from dm_control.utils import containers
from tcdm.envs.control import Environment, ReferenceMotionTask
from tcdm.envs.reference import HandObjectReferenceMotion
from tcdm.envs.rewards import ObjectMimic
from tcdm.envs.mujoco import physics_from_mjcf


class ObjMimicTask(ReferenceMotionTask):
    def __init__(self, object_name, data_path, reward_kwargs, append_time, 
                       pregrasp_init_key):
        reference_motion = HandObjectReferenceMotion(object_name, data_path)
        reward_fn = ObjectMimic(**reward_kwargs)
        self._append_time = append_time
        super().__init__(reference_motion, [reward_fn], pregrasp_init_key)
    
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        
        # append time to observation if needed
        if self._append_time:
            t = self.reference_motion.time
            t = np.array([1, 4, 6, 8]) * t
            t = np.concatenate((np.sin(t), np.cos(t)))
            obs['state'] = np.concatenate((obs['state'], t))
        return obs


class Sim2RealMimicTask(ObjMimicTask):
    def initialize_episode(self, physics):
        friction = [self.U(0.3, 0.7), self.U(0.0001,0.005), self.U(0.00001,0.0002)]
        width = self.U(0.015, 0.033)
        physics.named.model.geom_priority['crackerbox/crackerbox_contact0'] = 10
        physics.named.model.geom_size['crackerbox/crackerbox_contact0'][0] = width
        physics.named.model.geom_friction['crackerbox/crackerbox_contact0'] = friction
        super().initialize_episode(physics)

    def U(self, low, high):
        return self._random.uniform(low, high)


def _obj_mimic_task_factory(name, object_class, robot_class, target_path):
    def task(append_time=True, pregrasp='initialized', reward_kwargs={}, environment_kwargs={}):
        # load target data and construct environment
        object_model = object_class()
        object_name = '{}/object'.format(object_model.mjcf_model.model)
        env = mj_models.TableEnv()
        env.attach(robot_class(limp=False))
        env.attach(object_model)

        # build task using reference motion data
        data_path = traj_abspath(target_path)
        task = ObjMimicTask(object_name, data_path, reward_kwargs, append_time, pregrasp)

        # build physics object and create environment
        physics = physics_from_mjcf(env)
        return Environment(physics, task,
                           n_sub_steps=task.substeps,
                           **environment_kwargs)
    task.__name__ = name
    return task


TCDM_DOMAINS = {}
TCDM_DOMAINS['door'] = containers.TaggedTasks()
with open(asset_abspath('task_trajs.yaml'), 'r') as g:
    _TCDM_TRAJS = yaml.safe_load(g)['obj_mimic']

for target_fname in _TCDM_TRAJS:
    target = target_fname.split('/')[-1][:-4]
    domain_name = target.split('_')[0]
    task_name = ''.join(target.split('_')[1:])
    
    if domain_name not in TCDM_DOMAINS:
        TCDM_DOMAINS[domain_name] = containers.TaggedTasks()
    try:
        object_class = mj_models.get_object(domain_name)
        robot_class = mj_models.Adroit
    except:
        object_class = mj_models.get_object(task_name)
        robot_class = mj_models.get_robot(domain_name)
    task = _obj_mimic_task_factory(task_name, object_class, 
                                   robot_class, target_fname)
    TCDM_DOMAINS[domain_name].add('mimic')(task)
DOOR_SUITE, HAMMER_SUITE, DMANUS_SUITE = [TCDM_DOMAINS[k] for k in ('door', 'hammer', 'dmanus')]


@DMANUS_SUITE.add('mimic')
def sim2real(append_time=True, pregrasp='initialized', reward_kwargs={}, environment_kwargs={}):
    # load target data and construct environment
    object_model = mj_models.CrackerBoxObject()
    object_name = '{}/object'.format(object_model.mjcf_model.model)
    env = mj_models.TableEnv()
    env.attach(mj_models.DManus(limp=False))
    env.attach(object_model)

    # build task using reference motion data
    data_path = traj_abspath('dmanus_sim2real.npz')
    task = Sim2RealMimicTask(object_name, data_path, reward_kwargs, append_time, pregrasp)

    # build physics object and create environment
    physics = physics_from_mjcf(env)
    return Environment(physics, task,
                        n_sub_steps=task.substeps,
                        **environment_kwargs)


@DOOR_SUITE.add('mimic')
def open(append_time=True, pregrasp='initialized', reward_kwargs={}, environment_kwargs={}):
    # load target data and construct environment
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=False))
    env.attach(mj_models.DoorObject())

    # build task using reference motion data
    data_path = traj_abspath('door_open.npz')
    task = ObjMimicTask('door/object', data_path, reward_kwargs, append_time, pregrasp)

    # build physics object, set constants, and create environment
    physics = physics_from_mjcf(env)
    physics.named.model.body_pos['door/frame'] = np.array([-0.149568, 0.243598, 0.381253])
    return Environment(physics, task, n_sub_steps=task.substeps, 
                       default_camera_id=-1, **environment_kwargs)


@HAMMER_SUITE.add('mimic')
def strike(append_time=True, pregrasp='initialized', reward_kwargs={}, environment_kwargs={}):
    # load target data and construct environment
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=False))
    env.attach(mj_models.DAPGHammerObject())
    env.attach(mj_models.NailObject())

    # adjust timestep for accurate collision detection w/ DAPGHammerObject
    env.mjcf_model.option.timestep = 0.002

    # build task using reference motion data
    data_path = traj_abspath('hammer_strike.npz')
    task = ObjMimicTask('dapghammer/mid_point', data_path, reward_kwargs, append_time, pregrasp)

    # build physics object, set constants, and create environment
    physics = physics_from_mjcf(env)
    physics.named.model.body_pos['nail/object'] = np.array([0.05, 0.0, 0.23002825])
    physics.named.model.body_quat['nail/object'] = np.array([0.584,0.583,-0.399,-0.399])
    return Environment(physics, task, n_sub_steps=task.substeps, 
                       default_camera_id=0, **environment_kwargs)
