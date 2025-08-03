import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy

from .r3m import R3MFeatureExtractor
from .vint import VINTFeatureExtractor
from .efficientnet import EfficientNetFeatureExtractor
from .cnn import CNNFeatureExtractor

FEATURE_EXTRACTORS = {
    'cnn': CNNFeatureExtractor,
    'r3m': R3MFeatureExtractor,
    'vint': VINTFeatureExtractor,
    'efficientnet': EfficientNetFeatureExtractor
}

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        feature_extractor_kwargs = kwargs.pop('feature_extractor')
        super(CustomPPOPolicy, self).__init__(*args, **kwargs, 
                                           features_extractor_class=FEATURE_EXTRACTORS[feature_extractor_kwargs['name']], 
                                           features_extractor_kwargs=feature_extractor_kwargs)

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        feature_extractor_kwargs = kwargs.pop('feature_extractor')
        super(CustomSACPolicy, self).__init__(*args, **kwargs, 
                                           features_extractor_class=FEATURE_EXTRACTORS[feature_extractor_kwargs['name']], 
                                           features_extractor_kwargs=feature_extractor_kwargs)
        
class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        feature_extractor_kwargs = kwargs.pop('feature_extractor')
        super(CustomTD3Policy, self).__init__(*args, **kwargs, 
                                           features_extractor_class=FEATURE_EXTRACTORS[feature_extractor_kwargs['name']], 
                                           features_extractor_kwargs=feature_extractor_kwargs)


