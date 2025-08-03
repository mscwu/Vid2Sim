import torch
import torch.nn as nn
import imageio
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .vint_model import ViNT
from einops import rearrange

class VINTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, 
                       ckpt_path,
                       vec_encoding_size=512,
                       **kwargs):
        super(VINTFeatureExtractor, self).__init__(observation_space, features_dim=32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViNT(vec_encoding_size=vec_encoding_size)
        self.model.load_state_dict(torch.load(ckpt_path), strict=False)
        self.model.to(self.device)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 1, 3, 1, 1)
        
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        for param in self.model.obs_encoder.parameters():
            param.requires_grad = False


    def forward(self, observations):
        img_obs = observations['image'].to(self.device)
        vec_obs = observations['vector'].to(self.device)
        
        B, N, C, H, W = img_obs.shape
        img_obs = (img_obs - self.mean) / self.std
        assert N == 6, "VinT requires 6 frames input (5 context frames + 1 current frame)"
        feats = self.model(img_obs, vec_obs)
        return feats
        