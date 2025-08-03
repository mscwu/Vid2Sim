import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from efficientnet_pytorch import EfficientNet


class EfficientNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space, 
                 enet_type: str = "efficientnet-b0",
                 pretrained: bool = True,
                 finetune: bool = True,
                 activation_fn: type = nn.SiLU,
                 feats_dim: int = 1280,
                 reduced_dim: int = 64,
                 img_mlp_sizes: tuple = (128, 128),
                 vec_mlp_sizes: tuple = (16, 16),
                 **kwargs):
        super(EfficientNetFeatureExtractor, self).__init__(observation_space, features_dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enet = EfficientNet.from_pretrained(enet_type) if pretrained else EfficientNet.from_name(enet_type)
        self.enet.train()
        self.enet.to(self.device)

        # Freeze the model parameters
        if not finetune:
            for param in self.enet.parameters():
                param.requires_grad = False
            self.enet.eval()

        # Compute the output dimensions
        with torch.no_grad():
            N, C, H, W = observation_space['image'].shape
            if C != 3:
                self.enet.conv_stem = nn.Conv2d(
                    in_channels=C,
                    out_channels=self.enet.conv_stem.out_channels,
                    kernel_size=self.enet.conv_stem.kernel_size,
                    stride=self.enet.conv_stem.stride,
                    padding=self.enet.conv_stem.padding,
                    bias=True
                )
                nn.init.kaiming_normal_(self.enet.conv_stem.weight, mode='fan_out', nonlinearity='relu')
            
            vec_dim = observation_space['vector'].shape[0] // N
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_reduction = nn.Linear(feats_dim, reduced_dim)

        # Image MLP
        img_mlp_layers = self._build_mlp(reduced_dim*N, img_mlp_sizes, activation_fn)
        self.img_mlp = nn.Sequential(*img_mlp_layers)

        # Vector MLP
        vec_mlp_layers = self._build_mlp(vec_dim*N, vec_mlp_sizes, activation_fn)
        self.vec_mlp = nn.Sequential(*vec_mlp_layers)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)

        self._features_dim = (img_mlp_sizes[-1] + vec_mlp_sizes[-1]) 

    def _build_mlp(self, input_dim, layer_sizes, activation_fn):
        layers = []
        for size in layer_sizes:
            linear = nn.Linear(input_dim, size)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(linear.bias, 0)
            layers.append(linear)
            layers.append(activation_fn())
            input_dim = size
        return layers

    def forward(self, observations):
        # Extract image and vector observations
        img_obs = observations['image'].to(self.device)
        vec_obs = observations['vector'].to(self.device)

        B, N, C, H, W = img_obs.shape

        img_obs = img_obs.reshape(-1, C, H, W)
        vec_obs = vec_obs.reshape(B, -1)
        img_obs = (img_obs - self.mean) / self.std

        feats = self.enet.extract_features(img_obs)
        feats = self.pool(feats).reshape(B, N, -1)
        feats = self.dim_reduction(feats)
        feats = feats.reshape(B, -1)

        img_feats = self.img_mlp(feats)
        vec_feats = self.vec_mlp(vec_obs)

        final_feats = torch.cat([img_feats, vec_feats], dim=1)
        return final_feats
