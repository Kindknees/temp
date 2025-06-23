import gymnasium as gym
import numpy as np
import logging

import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType
# from ray.rllib.models.torch.torch_action_dist import TorchCategorical # Not strictly needed if relying on self.dist_class

FLOAT_MIN = -3.4e38

class BaseHierarchicalRLM(TorchRLModule, nn.Module):
    """
    基礎 RLModule，共享 Critic 網路層。
    建構子已更新至最新 API。
    """
    def __init__(self, *, observation_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space, 
                 model_config: ModelConfigDict, 
                 **kwargs):
        nn.Module.__init__(self)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs
        )
        
        # Assuming regular_obs after concatenation has this dimension
        # This should match the sum of dimensions of individual components of regular_obs
        # Example: if regular_obs = {"comp1": Box(shape=(100,)), "comp2": Box(shape=(52,))} -> input_dim = 152
        # This needs to be robustly calculated or configured if obs structure changes.
        # For now, we keep the hardcoded value as in the original file.
        input_dim = 152 
        
        self.shared_vf_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.vf_head = nn.Linear(256, 1)

    def _get_regular_obs(self, batch: Dict):
        # batch[SampleBatch.OBS] contains the full observation dictionary from the env
        # For high-level agent: obs = {"regular_obs": self.cur_obs, "chosen_action": ...}
        # For low-level agent: obs = {"action_mask": ..., "regular_obs": self.cur_obs, "chosen_substation": ...}
        # self.cur_obs itself is a dict like {"rho": ..., "p_mw": ..., ...}
        obs = batch[SampleBatch.OBS]
        regular_obs_dict = obs["regular_obs"] 
        # Ensure correct device
        return torch.concat([val.to(self.vf_head.weight.device) if isinstance(val, torch.Tensor) else torch.tensor(val, device=self.vf_head.weight.device) for val in regular_obs_dict.values()], dim=1)


class ChooseSubstationModel(BaseHierarchicalRLM):
    """高層決策者，建構子已更新。"""
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
        self.num_outputs = action_space.n
        input_dim = 152 # As per BaseHierarchicalRLM for regular_obs

        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)

    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        regular_obs = self._get_regular_obs(batch)
        actor_features = self.actor_base(regular_obs)
        action_logits = self.logits_head(actor_features)
        
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }

    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        regular_obs = self._get_regular_obs(batch) # Input processing
        actor_features = self.actor_base(regular_obs) # Actor features
        action_logits = self.logits_head(actor_features) # Logits
        
        # Get deterministic actions using the distribution class from RLModule
        action_dist = self.dist_class(action_logits) 
        actions = action_dist.deterministic_sample()
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits, # Optional, but good for consistency
            SampleBatch.ACTIONS: actions,
        }

    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        regular_obs = self._get_regular_obs(batch) # Input processing
        actor_features = self.actor_base(regular_obs) # Actor features
        action_logits = self.logits_head(actor_features) # Logits
        
        # Get stochastic actions using the distribution class from RLModule
        action_dist = self.dist_class(action_logits)
        actions = action_dist.sample()
        
        # Logp is not strictly needed by PPO for sampling env, but can be included if desired
        # action_logp = action_dist.logp(actions)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits, # Optional, good for consistency
            SampleBatch.ACTIONS: actions,
            # SampleBatch.ACTION_LOGP: action_logp,
        }


class ChooseActionModel(BaseHierarchicalRLM):
    """低層執行者，建構子已更新。"""
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
        self.num_outputs = action_space.n
        
        # Determine num_substations dynamically from observation_space
        # obs_space for this agent is Dict({"action_mask": ..., "regular_obs": ..., "chosen_substation": Discrete(N)})
        if isinstance(observation_space, gym.spaces.Dict) and "chosen_substation" in observation_space.spaces:
            self.num_substations = observation_space["chosen_substation"].n
        else:
            # Fallback or error, assuming 8 if not found, but this should be configured robustly
            logging.warning("Could not determine num_substations from observation_space, defaulting to 8. Ensure 'chosen_substation' is in obs space.")
            self.num_substations = 8 
            
        input_dim_regular_obs = 152 # Dimension of concatenated regular_obs
        input_dim_actor = input_dim_regular_obs + self.num_substations # regular_obs + one_hot_chosen_substation
        
        self.separate_actor_base = nn.Sequential(
            nn.Linear(input_dim_actor, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Linear(256, self.num_outputs)
        
    def _get_actor_input_and_mask(self, batch: Dict[str, torch.Tensor]):
        obs = batch[SampleBatch.OBS]
        regular_obs = self._get_regular_obs(batch) # This gets from obs["regular_obs"]
        
        # Ensure chosen_substation is on the correct device and is long type
        chosen_sub_tensor = obs["chosen_substation"]
        if not isinstance(chosen_sub_tensor, torch.Tensor):
            chosen_sub_tensor = torch.tensor(chosen_sub_tensor, device=regular_obs.device)
        chosen_sub_tensor = chosen_sub_tensor.long()

        chosen_sub_one_hot = torch.nn.functional.one_hot(chosen_sub_tensor, num_classes=self.num_substations).float()
        
        action_mask = obs["action_mask"]
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, device=regular_obs.device, dtype=torch.float32)
        else:
            action_mask = action_mask.to(device=regular_obs.device, dtype=torch.float32)

        actor_input = torch.cat([regular_obs, chosen_sub_one_hot], dim=1)
        return actor_input, action_mask

    def _forward_train(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        # Critic uses only regular_obs
        regular_obs = self._get_regular_obs(batch) # Re-extract or pass from _get_actor_input_and_mask
        vf_features = self.shared_vf_base(regular_obs)
        vf_preds = self.vf_head(vf_features).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
            SampleBatch.VF_PREDS: vf_preds,
        }

    def _forward_inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        action_dist = self.dist_class(masked_action_logits)
        actions = action_dist.deterministic_sample()
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
            SampleBatch.ACTIONS: actions,
        }

    def _forward_exploration(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        actor_input, action_mask = self._get_actor_input_and_mask(batch)
        
        actor_features = self.separate_actor_base(actor_input)
        action_logits = self.logits_head(actor_features)
        
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_action_logits = action_logits + inf_mask
        
        action_dist = self.dist_class(masked_action_logits)
        actions = action_dist.sample()
        
        # action_logp = action_dist.logp(actions)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: masked_action_logits,
            SampleBatch.ACTIONS: actions,
            # SampleBatch.ACTION_LOGP: action_logp,
        }