# mlp.py (已修正為 RLModule API)

import gymnasium as gym
import numpy as np
import logging

import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
# from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType

# 適合用作 -inf logit 的極小值
FLOAT_MIN = -3.4e38

class BaseHierarchicalRLM(TorchRLModule, nn.Module):
    """
    基礎 RLModule，共享 Critic 網路層。
    建構子已更新至最新 API。
    """
    # [修正] 更新 __init__ 簽名以符合新版 API
    def __init__(self, *, observation_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space, 
                 model_config: ModelConfigDict, 
                 **kwargs):
        nn.Module.__init__(self)
        # [修正] 使用新的參數來呼叫父類別建構子
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs
        )
        
        input_dim = 152 
        
        self.shared_vf_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.vf_head = nn.Linear(256, 1)

    def _get_regular_obs(self, batch: Dict):
        obs = batch[SampleBatch.OBS]
        regular_obs_dict = obs["regular_obs"]
        return torch.concat([val for val in regular_obs_dict.values()], dim=1)


class ChooseSubstationModel(BaseHierarchicalRLM):
    """高層決策者，建構子已更新。"""
    # [修正] 更新 __init__ 簽名
    def __init__(self, *, observation_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space, 
                 model_config: ModelConfigDict, 
                 **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs
        )
        # [修正] 直接從 action_space 獲取輸出維度
        self.num_outputs = action_space.n
        input_dim = 152

        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)

    @override
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        # ... forward 邏輯不變 ...
        regular_obs = self._get_regular_obs(batch)
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }


class ChooseActionModel(BaseHierarchicalRLM):
    """低層執行者，建構子已更新。"""
    # [修正] 更新 __init__ 簽名
    def __init__(self, *, observation_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space, 
                 model_config: ModelConfigDict, 
                 **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs
        )
        # [修正] 直接從 action_space 獲取輸出維度
        self.num_outputs = action_space.n
        
        input_dim = 160
        self.separate_actor_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)
        
    @override
    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        # ... forward 邏輯不變 ...
        obs = batch[SampleBatch.OBS]
        regular_obs = self._get_regular_obs(batch)
        chosen_sub_one_hot = torch.nn.functional.one_hot(obs["chosen_substation"].long(), num_classes=8).float()
        action_mask = obs["action_mask"]
        actor_input = torch.cat([regular_obs, chosen_sub_one_hot], dim=1)
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        action_logits += inf_mask
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }