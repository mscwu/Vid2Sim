import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from r3m import load_r3m

class R3MFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space, 
                 r3m_type: str = "resnet34",
                 finetune: bool = False,
                 img_mlp_sizes: tuple = (256, 32),
                 vec_mlp_sizes: tuple = (256, 32),
                 activation_fn: nn.Module = nn.ReLU,
                 use_batch_norm: bool = True,
                 use_dropout: bool = False,
                 dropout_p: float = 0.5, 
                 **kwargs):
        super(R3MFeatureExtractor, self).__init__(observation_space, features_dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r3m = load_r3m(r3m_type) # resnet18, resnet34
        self.r3m.to(self.device)
        self.r3m.eval()
        
        # Freeze the model parameters
        if not finetune:
            for param in self.r3m.parameters():
                param.requires_grad = False

        # Compute the CNN output dimensions
        with torch.no_grad():
            N, _, H, W = observation_space['image'].shape
            zero_tensor = torch.zeros((1, 3, H, W), device=self.device)
            feats = self.r3m(zero_tensor)
            C = feats.shape[-1]
            vec_dim = observation_space['vector'].shape[0] // N

        # Image MLP
        img_mlp_layers = self._build_mlp(C*N, img_mlp_sizes, activation_fn, use_batch_norm, use_dropout, dropout_p)
        self.img_mlp = nn.Sequential(*img_mlp_layers)

        # Vector MLP
        vec_mlp_layers = self._build_mlp(vec_dim*N, vec_mlp_sizes, activation_fn, use_batch_norm, use_dropout, dropout_p)
        self.vec_mlp = nn.Sequential(*vec_mlp_layers)

        self._features_dim = (img_mlp_sizes[-1] + vec_mlp_sizes[-1]) 

    def _build_mlp(self, input_dim, layer_sizes, activation_fn, use_batch_norm, use_dropout, dropout_p):
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(activation_fn())
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = size
        return layers

    def forward(self, observations):
        # Extract image and vector observations
        img_obs = observations['image'].to(self.device)
        vec_obs = observations['vector'].to(self.device)

        if img_obs.max() <= 1:
            img_obs *= 255 # R3M expects input in [0, 255]

        B, N, C, H, W = img_obs.shape
        img_obs = img_obs.reshape(-1, C, H, W)
        vec_obs = vec_obs.reshape(B, -1)

        # Extract image features
        with torch.no_grad():
            feats = self.r3m(img_obs).reshape(B, -1) # 512 * N

        img_feats = self.img_mlp(feats)
        vec_feats = self.vec_mlp(vec_obs)

        final_feats = torch.cat([img_feats, vec_feats], dim=1)
        return final_feats
