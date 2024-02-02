# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from stable_baselines3.ppo.custom_policies import CustomFeaturesExtractor
CustomACPolicy = CustomFeaturesExtractor
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
