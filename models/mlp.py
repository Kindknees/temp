# 檔案: models/mlp.py
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
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

FLOAT_MIN = -3.4e38

# ADDED: 創建一個共享的 Actor 基礎網路，以模擬舊專案的 SHARED_ACTOR 行為
# 兩個 agent 將共享同一個實例
SHARED_ACTOR_BASE = nn.Sequential(
    nn.Linear(152, 256), nn.ReLU(inplace=True),
    nn.Linear(256, 256), nn.ReLU(inplace=True),
    nn.Linear(256, 256), nn.ReLU(inplace=True),
    nn.Linear(256, 256), nn.ReLU(inplace=True),
)

class BaseHierarchicalRLM(TorchRLModule, ValueFunctionAPI):
    """
    基礎 RLModule，共享 Critic 網路層。
    """
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
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
        
        self.shared_vf_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.vf_head = nn.Linear(256, 1)

    def _get_regular_obs(self, batch: Dict):
        """Extract and concatenate regular observations from batch."""
        obs = batch.get(Columns.OBS, batch.get(SampleBatch.OBS))
        
        if isinstance(obs, dict) and "regular_obs" in obs:
            regular_obs = obs["regular_obs"]
        else:
            regular_obs = obs
        
        device = next(self.parameters()).device
        
        if isinstance(regular_obs, torch.Tensor):
            return regular_obs.to(device)
        elif isinstance(regular_obs, np.ndarray):
            return torch.tensor(regular_obs, device=device, dtype=torch.float32)
        else:
            obs_list = []
            for val in regular_obs.values():
                if isinstance(val, torch.Tensor):
                    obs_list.append(val.to(device))
                else:
                    obs_list.append(torch.tensor(val, device=device, dtype=torch.float32))
            return torch.cat(obs_list, dim=1)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict, **kwargs):
        """
        Computes the value function estimates for the given batch of observations.
        """
        regular_obs = self._get_regular_obs(batch)
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return vf_preds


class ChooseSubstationModel(BaseHierarchicalRLM):
    """高層決策者 - 選擇變電站。"""
    
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
        super().setup()
        
        self.num_outputs = self.config.action_space.n
        
        # CHANGED: 使用共享的 Actor Base
        self.actor_base = SHARED_ACTOR_BASE
        self.logits_head = nn.Linear(256, self.num_outputs)
        
        self._action_dist_class = TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self._action_dist_class

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self._action_dist_class

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self._action_dist_class

    def _common_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Common forward pass for all modes."""
        regular_obs = self._get_regular_obs(batch)
        
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: vf_preds,
        }

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)


class ChooseActionModel(BaseHierarchicalRLM):
    """低層執行者 - 在選定的變電站內選擇動作。"""
    
    @override(RLModule)
    def setup(self):
        """Setup the neural network components."""
        super().setup()
        
        self.num_outputs = self.config.action_space.n
        
        obs_space = self.config.observation_space
        if isinstance(obs_space, gym.spaces.Dict) and "chosen_substation" in obs_space.spaces:
            self.num_substations = obs_space["chosen_substation"].n
        else:
            logging.warning("Could not determine num_substations from observation_space, defaulting to 8.")
            self.num_substations = 8 
        
        # CHANGED: 使用共享的 Actor Base
        self.actor_base = SHARED_ACTOR_BASE
        # CHANGED: 輸出層的輸入維度現在是 actor 特徵(256) + one-hot決策(8)
        self.logits_head = nn.Linear(256 + self.num_substations, self.num_outputs)
        
        self._action_dist_class = TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self._action_dist_class

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self._action_dist_class

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self._action_dist_class
        
    def _get_actor_input_and_mask(self, batch: Dict[str, torch.Tensor]):
        """Prepare actor input and action mask."""
        obs = batch.get(Columns.OBS, batch.get(SampleBatch.OBS))
        regular_obs = self._get_regular_obs(batch)
        
        device = next(self.parameters()).device
        
        chosen_sub_tensor = obs["chosen_substation"]
        if not isinstance(chosen_sub_tensor, torch.Tensor):
            chosen_sub_tensor = torch.tensor(chosen_sub_tensor, device=device)
        else:
            chosen_sub_tensor = chosen_sub_tensor.to(device)
        chosen_sub_tensor = chosen_sub_tensor.long()
        
        chosen_sub_one_hot = torch.nn.functional.one_hot(
            chosen_sub_tensor, num_classes=self.num_substations
        ).float()
        
        action_mask = obs["action_mask"]
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, device=device, dtype=torch.float32)
        else:
            action_mask = action_mask.to(device=device, dtype=torch.float32)

        return regular_obs, chosen_sub_one_hot, action_mask
    
    def _common_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """A common forward pass for all modes."""
        # CHANGED: 此處的分解變更，以匹配舊的邏輯
        regular_obs, chosen_sub_one_hot, action_mask = self._get_actor_input_and_mask(batch)
        
        # Actor forward pass
        # 1. 先從 regular_obs 提取特徵
        actor_features = self.actor_base(regular_obs)
        # 2. 再將特徵與高階決策串接
        combined_features = torch.cat([actor_features, chosen_sub_one_hot], dim=1)
        # 3. 最後通過輸出層
        action_logits = self.logits_head(combined_features)
        
        # Apply action mask
        masked_action_logits = torch.where(
            action_mask.bool(),
            action_logits,
            torch.tensor(FLOAT_MIN, device=action_logits.device)
        )
        
        # Critic forward pass (uses only regular_obs)
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            Columns.ACTION_DIST_INPUTS: masked_action_logits,
            Columns.VF_PREDS: vf_preds,
        }

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._common_forward(batch)