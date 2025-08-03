import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.utils import adjust_hue

class CNNImageEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, encoding_dim=128):
        super(CNNImageEncoder, self).__init__()
        
        # Initial convolution layer with smaller kernel and stride
        self.conv_initial = nn.Conv2d(input_channels, base_channels, 
                                      kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*8, base_channels*16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*16, base_channels*16, kernel_size=4, stride=4, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_final = nn.Conv2d(base_channels*16, encoding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Initial layers
        x = self.conv_initial(x)
        x = self.relu(x)
        x = self.conv_net(x)
        x = self.conv_final(x)
        x = x.flatten(1)
        return x

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space, 
                 base_channel: int = 64,  # Hidden channel sizes for CNN
                 encode_dim: int = 64,  # Output channels for CNN
                 vec_mlp_sizes: tuple = (64, 256),
                 activation_fn=nn.ReLU,
                 use_batch_norm=False,
                 image_aug=True,
                 **kwargs):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N, C, H, W = observation_space['image'].shape
        vec_dim = observation_space['vector'].shape[0] // N

        # Dynamically building the CNN model based on input parameters
        self.model = CNNImageEncoder(input_channels=C,
                                        base_channels=base_channel,
                                        encoding_dim=encode_dim
                                      ).to(self.device)
        self.image_aug = image_aug

        # Determine the feature size from the CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, observation_space['image'].shape[2], observation_space['image'].shape[3]).to(self.device)
            cnn_out_dim = self.model(dummy_input).shape[1]  # Get CNN output dimension
        
        # Define the feature dimension
        self._features_dim = (cnn_out_dim + vec_dim)*N


    def forward(self, observations):
        # Extract image and vector observations
        img_obs = observations['image'].to(self.device)
        vec_obs = observations['vector'].to(self.device) # N * 2 

        B, N, C, H, W = img_obs.shape
        img_obs = img_obs.reshape(-1, C, H, W)  # Reshape to (B*N, C, H, W) for CNN processing
        vec_obs = vec_obs.reshape(B, -1)  # Reshape vector obs

        if self.image_aug and C == 3:
            brightness = torch.empty(B*N).uniform_(0.5, 1.5).to(self.device).view(B*N, 1, 1, 1)
            contrast = torch.empty(B*N).uniform_(0.8, 1.2).to(self.device).view(B*N, 1, 1, 1)
            hue_shift = torch.empty(B*N).uniform_(-0.1, 0.1).to(self.device).view(B*N, 1, 1, 1)

            # Adjust Brightness
            img_obs = img_obs * brightness
            img_obs = torch.clamp(img_obs, 0.0, 1.0)

            # Adjust Contrast
            mean = img_obs.mean(dim=[2,3], keepdim=True)
            img_obs = (img_obs - mean) * contrast + mean
            img_obs = torch.clamp(img_obs, 0.0, 1.0)

            # Adjust Hue
            img_obs = adjust_hue(img_obs, hue_shift)
            img_obs = torch.clamp(img_obs, 0.0, 1.0)

        # Extract image features
        cnn_feats = self.model(img_obs).reshape(B, -1)  # Apply CNN and reshape to (B, cnn_out_dim * N)
        # Concatenate image and vector features
        final_feats = torch.cat([cnn_feats, vec_obs], dim=1)
        return final_feats
