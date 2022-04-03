# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from typing import Tuple, Union, List
from stable_baselines3.common import distributions


class DiagGaussianDistribution(distributions.DiagGaussianDistribution):
    """
    Modified GaussianDistribution class from StableBaselines3: https://github.com/DLR-RM/stable-baselines3
       - Now includes support for vectors of standard deviations

    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    :param action_dim:  Dimension of the action space.
    """

    def proba_distribution_net(self, latent_dim: int, log_std_init: Union[float, List[float]] = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if isinstance(log_std_init, float):
            log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        else:
            assert len(log_std_init) == self.action_dim, "must supply std for each action dimension"
            log_std = nn.Parameter(torch.Tensor(log_std_init).float(), requires_grad=True)
        return mean_actions, log_std
