import torch
import torch.nn as nn

from typing import List, Dict, Optional, Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_obs: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_img (torch.Tensor): batch of observations
            goal_img (torch.Tensor): batch of goals
        Returns:
            dist_pred (torch.Tensor): predicted distance to goal
            action_pred (torch.Tensor): predicted action
        """
        raise NotImplementedError
