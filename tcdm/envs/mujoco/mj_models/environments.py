# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from tcdm.envs import asset_abspath
from dm_control import mjcf
from .base import MjModel


class BaseEnv(MjModel):
    def __init__(self, xml_path):
        xml_path = asset_abspath(xml_path)
        mjcf_model = mjcf.from_path(xml_path)
        super().__init__(mjcf_model)


class TableEnv(BaseEnv):
    def __init__(self):
        super().__init__('environments/table.xml')


class EmptyEnv(BaseEnv):
    def __init__(self):
        super().__init__('environments/empty.xml')
