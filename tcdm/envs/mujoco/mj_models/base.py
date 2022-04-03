# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MjModel(object):
    """ """

    def __init__(self, mjcf_model):
        self._mjcf_model = mjcf_model

    def attach(self, other):
        """ """
        self.mjcf_model.attach(other.mjcf_model)
        other._post_attach(self.mjcf_model)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    def _post_attach(self, base_mjfc_model):
        """ """
