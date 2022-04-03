# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from tcdm.envs import asset_abspath
from tcdm.envs.mujoco.physics import ROBOT_CONFIGS
from dm_control import mjcf
from .base import MjModel


class BaseRobot(MjModel):
    def __init__(self, robot_name, limp=False, vis=False):
        self._robot_name = robot_name
        self._bodies = ROBOT_CONFIGS[robot_name]['BODIES']

        base_model = mjcf.from_path(asset_abspath('robots/{}/base.xml'.format(robot_name)))
        self._attach_mocap = bool(limp)
        self._vis = vis
        if not self._attach_mocap:
            # adds motors to actuated model
            actuators = mjcf.from_path(asset_abspath('robots/{}/actuators.xml'.format(robot_name)))
            for a in actuators.actuator.all_children():
                a_data = {}
                if a.tag == 'general':
                    keys = ['name', 'class', 'joint', 'ctrllimited', 'ctrlrange',
                                            'biastype', 'gainprm', 'biasprm', 'forcerange']
                elif a.tag == 'position':
                    keys = ['name', 'class', 'joint', 'ctrllimited', 'ctrlrange', 'kp', 'forcerange']
                else:
                    raise NotImplementedError
                for key in keys:
                    val = getattr(a, key)
                    if val is not None:
                        a_data[key] = val
                base_model.actuator.add(a.tag, **a_data)

        # finished object initialization by passing final xml
        super().__init__(base_model)

    def _post_attach(self, env_mjfc):
        if self._attach_mocap:
            # adds mocaps to limp model
            with open(asset_abspath('robots/{}/mocaps.txt'.format(self._robot_name)), 'r') as f:
                mocap_locations = [[float(x) for x in l.split(' ')] for l in f.read().split('\n')]
            assert len(mocap_locations) == len(self._bodies), "Must specify mocap per body!"

            for i, p in enumerate(mocap_locations):
                # create body
                mocap_name = 'j{}_mocap'.format(i)
                mocap_body = env_mjfc.worldbody.add('body', mocap=True, name=mocap_name, pos=p)
                rgba = [0,0,1,1] if self._vis else [0,0,1,0]
                mocap_body.add('site', name='j{}'.format(i), size=(0.005,), rgba=rgba, pos=[0,0,0])

                # create equality constraint
                robot_body = '{}/{}'.format(self._robot_name, self._bodies[i])
                env_mjfc.equality.add('connect', body1=mocap_name, body2=robot_body, anchor=[0,0,0], 
                                                   solimp=[0.9,0.95,0.001], solref=[0.02,1])


class Adroit(BaseRobot):
    def __init__(self, limp=False, vis=False):
        super().__init__('adroit', limp, vis)


class DHand(BaseRobot):
    def __init__(self, limp=False, vis=False):
        super().__init__('dhand', limp, vis)


class DManus(BaseRobot):
    def __init__(self, limp=False, vis=False):
        super().__init__('dmanus', limp, vis)


class Franka(BaseRobot):
    def __init__(self, limp=False, vis=False):
        super().__init__('franka', limp, vis)


def get_robot(name):
    if name == 'adroit':
        return Adroit
    elif name == 'dhand':
        return DHand
    elif name == 'dmanus':
        return DManus
    elif name == 'franka':
        return Franka
    raise NotImplementedError
