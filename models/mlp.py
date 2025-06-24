import gymnasium as gym
import numpy as np
import logging

import torch
import torch.nn as nn

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical

FLOAT_MIN = -3.4e38

class BaseHierarchicalRLM(TorchRLModule):
    """
    基礎 RLModule，共享 Critic 網路層。
    使用新的 RLModule API。
    """
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
        # Dynamically determine input dimension from observation space
        obs_space = self.config.observation_space
        if isinstance(obs_space, gym.spaces.Dict) and "regular_obs" in obs_space.spaces:
            regular_obs_space = obs_space["regular_obs"]
            if isinstance(regular_obs_space, gym.spaces.Box):
                input_dim = regular_obs_space.shape[0]
            else:
                # Fallback to hardcoded value if structure is unexpected
                input_dim = 152
        else:
            # Fallback for non-hierarchical case
            if isinstance(obs_space, gym.spaces.Box):
                input_dim = obs_space.shape[0]
            else:
                input_dim = 152
        
        self.shared_vf_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.vf_head = nn.Linear(256, 1)

    def _get_regular_obs(self, batch: Dict):
        """Extract and concatenate regular observations from batch."""
        obs = batch[SampleBatch.OBS]
        regular_obs = obs["regular_obs"]
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # If regular_obs is already a tensor (from Grid2Op's BoxGymnasiumObsSpace), use it directly
        if isinstance(regular_obs, torch.Tensor):
            return regular_obs.to(device)
        elif isinstance(regular_obs, np.ndarray):
            return torch.tensor(regular_obs, device=device, dtype=torch.float32)
        else:
            # If it's a dictionary, concatenate the values
            obs_list = []
            for val in regular_obs.values():
                if isinstance(val, torch.Tensor):
                    obs_list.append(val.to(device))
                else:
                    obs_list.append(torch.tensor(val, device=device, dtype=torch.float32))
            return torch.cat(obs_list, dim=1)


class ChooseSubstationModel(BaseHierarchicalRLM):
    """高層決策者 - 選擇變電站。"""
    
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
        super().setup()
        
        # Get action space dimension
        self.num_outputs = self.config.action_space.n
        
        # Dynamically determine input dimension
        obs_space = self.config.observation_space
        if isinstance(obs_space, gym.spaces.Dict) and "regular_obs" in obs_space.spaces:
            regular_obs_space = obs_space["regular_obs"]
            if isinstance(regular_obs_space, gym.spaces.Box):
                input_dim = regular_obs_space.shape[0]
            else:
                input_dim = 152
        else:
            if isinstance(obs_space, gym.spaces.Box):
                input_dim = obs_space.shape[0]
            else:
                input_dim = 152
        
        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)
        
        # Initialize action distribution
        self._action_dist_class = TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        """Return the action distribution class for training."""
        return self._action_dist_class

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return the action distribution class for exploration."""
        return self._action_dist_class

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return the action distribution class for inference."""
        return self._action_dist_class

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        regular_obs = self._get_regular_obs(batch)
        
        # Actor forward pass
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        
        # Critic forward pass
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for inference (deterministic actions)."""
        regular_obs = self._get_regular_obs(batch)
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
        }

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for exploration (stochastic actions)."""
        regular_obs = self._get_regular_obs(batch)
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
        }


class ChooseActionModel(BaseHierarchicalRLM):
    """低層執行者 - 在選定的變電站內選擇動作。"""
    
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
        super().setup()
        
        # Get action space dimension
        self.num_outputs = self.config.action_space.n
        
        # Determine num_substations from observation space
        obs_space = self.config.observation_space
        if isinstance(obs_space, gym.spaces.Dict) and "chosen_substation" in obs_space.spaces:
            self.num_substations = obs_space["chosen_substation"].n
        else:
            logging.warning("Could not determine num_substations from observation_space, defaulting to 8.")
            self.num_substations = 8 
        
        # Dynamically determine input dimension
        if isinstance(obs_space, gym.spaces.Dict) and "regular_obs" in obs_space.spaces:
            regular_obs_space = obs_space["regular_obs"]
            if isinstance(regular_obs_space, gym.spaces.Box):
                input_dim_regular_obs = regular_obs_space.shape[0]
            else:
                input_dim_regular_obs = 152
        else:
            input_dim_regular_obs = 152
            
        input_dim_actor = input_dim_regular_obs + self.num_substations
        
        self.separate_actor_base = nn.Sequential(
            nn.Linear(input_dim_actor, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)
        
        # Initialize action distribution
        self._action_dist_class = TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        """Return the action distribution class for training."""
        return self._action_dist_class

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return the action distribution class for exploration."""
        return self._action_dist_class

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return the action distribution class for inference."""
        return self._action_dist_class
        
    def _get_actor_input_and_mask(self, batch: Dict[str, torch.Tensor]):
        """Prepare actor input and action mask."""
        obs = batch[SampleBatch.OBS]
        regular_obs = self._get_regular_obs(batch)
        
        # Get device
        device = next(self.parameters()).device
        
        # Process chosen_substation
        chosen_sub_tensor = obs["chosen_substation"]
        if not isinstance(chosen_sub_tensor, torch.Tensor):
            chosen_sub_tensor = torch.tensor(chosen_sub_tensor, device=device)
        else:
            chosen_sub_tensor = chosen_sub_tensor.to(device)
        chosen_sub_tensor = chosen_sub_tensor.long()
        
        # One-hot encode the chosen substation
        chosen_sub_one_hot = torch.nn.functional.one_hot(
            chosen_sub_tensor, num_classes=self.num_substations
        ).float()
        
        # Process action mask
        action_mask = obs["action_mask"]
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, device=device, dtype=torch.float32)
        else:
            action_mask = action_mask.to(device=device, dtype=torch.float32)

        # Concatenate regular obs with one-hot encoded substation
        actor_input = torch.cat([regular_obs, chosen_sub_one_hot], dim=1)
        
        return actor_input, action_mask

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        # Actor forward pass
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        # Critic forward pass (uses only regular_obs)
        regular_obs = self._get_regular_obs(batch)
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for inference."""
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        # Actor forward pass
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
        }

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for exploration."""
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        # Actor forward pass
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
        }