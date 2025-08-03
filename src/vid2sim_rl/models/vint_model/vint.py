import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet

from .base_model import BaseModel
from .self_attention import MultiLayerDecoder

class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        vis_encoding_size: Optional[int] = 512,
        vec_encoding_size: Optional[int] = 32,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 4,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            vis_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(ViNT, self).__init__(context_size)
        self.vis_encoding_size = vis_encoding_size
        self.goal_encoding_size = vis_encoding_size
        self.vec_encoding_size = vec_encoding_size

        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
        self.num_obs_features = self.obs_encoder._fc.in_features
        
        self.goal_vec_encoder = nn.Sequential(
            nn.Linear(2*(context_size+1), self.vec_encoding_size),
            nn.ReLU(),
            nn.Linear(self.vec_encoding_size, self.vec_encoding_size),
            nn.ReLU(),
        )
        self.num_goal_features = self.obs_encoder._fc.in_features + self.vec_encoding_size
        self.goal_combine_encoder = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        
        if self.num_obs_features != self.vis_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.vis_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.vis_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

    def forward(
        self, 
        obs_img: torch.tensor, 
        obs_goal: torch.tensor,
    ):
        B, N, C, H, W = obs_img.shape
        assert N == self.context_size + 1
        obs_img = obs_img.reshape(B*N, C, H, W)
        vis_encoding = self.obs_encoder.extract_features(obs_img)
        vis_encoding = self.obs_encoder._avg_pooling(vis_encoding)
        if self.obs_encoder._global_params.include_top:
            vis_encoding = vis_encoding.flatten(start_dim=1)
            vis_encoding = self.obs_encoder._dropout(vis_encoding)

        goal_encoding = self.goal_vec_encoder(obs_goal)
        vis_feats = vis_encoding.detach().clone()
        vis_feats = vis_feats.reshape(B, N, self.num_obs_features)[:, -1]
        goal_encoding = self.goal_combine_encoder(torch.cat((vis_feats, goal_encoding), dim=1))
        assert goal_encoding.shape[-1] == self.goal_encoding_size
        
        obs_encoding = self.compress_obs_enc(vis_encoding)
        obs_encoding = obs_encoding.reshape((B, N, self.vis_encoding_size))
        goal_encoding = goal_encoding.reshape((B, 1, self.goal_encoding_size))

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)  # batch_size, 32
        return final_repr
    
if __name__ == "__main__":
    model = ViNT()
    model.eval()

    ckpt_path = '/home/ziyangxie/Code/Video2Sim-RL/checkpoints/vint_model.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt, strict=False)

    obs = torch.randn(5, 6, 3, 85, 40)
    goal = torch.randn(5, 6, 2)
    feat = model(obs, goal)
    import pdb; pdb.set_trace()