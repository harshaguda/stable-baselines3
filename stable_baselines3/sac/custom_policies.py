from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy

from .segnet import SegNet

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# class CustomNetwork(nn.Module):
#     def __init__(
#             self, 
#             features_dim: int,
#             last_layer_dim_pi: int = 64,
#             last_layer_dim_vf: int = 64,
#             ):
#         super().__init__()


class CustomActor(Actor):
    """
    Actor network (policy) for SAC.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        self.mu = SegNet(3,3)


class CustomContinousCritic(ContinuousCritic):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True
            ):
        super().__init(
            observation_space,
            action_space,
            features_extractor,
            normalize_images
        )
        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []

        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
            with th.set_grad_enabled(not self.share_features_extractor):
                features = self.features_extractor(obs)
            
            qvalue_input = th.cat([features, actions], dim=1)

            return tuple(qnet(qvalue_input) for qnet in self.q_networks)
        
        def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
            with th.no_grad():
                features = self.extract_features(obs)
            return self.q_networks[0](th.cat([features, actions], dim=1))
        

class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
            actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
            return CustomActor(**actor_kwargs).to(self.device)
        
        def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinousCritic:
            critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
            return CustomContinousCritic(**critic_kwargs).to(self.device)
