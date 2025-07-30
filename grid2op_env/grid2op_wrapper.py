# kindknees-temp/grid2op_env/grid2op_wrapper.py (替換全部內容)

import numpy as np
from typing import Any, Dict as TypingDict
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

from ray.rllib.env import MultiAgentEnv

# --- MODIFIED: Import the new GridGym ---
from grid2op_env.grid_to_gym import GridGym
from grid2op_env.utils import get_sub_id_to_action

import logging
logger = logging.getLogger(__name__)

class SerializableHierarchicalGridGym(MultiAgentEnv):
    def __init__(self, env_config: TypingDict[str, Any] = None):
        super().__init__()

        self.high_level_agent_id = "choose_substation_agent"
        self.low_level_agent_id = "choose_action_agent"
        self._agent_ids = {self.high_level_agent_id, self.low_level_agent_id}
        
        self.agents = []
        self.possible_agents = [self.high_level_agent_id, self.low_level_agent_id]
            
        self._skip_env_checking = True
        
        self.env_config = env_config if env_config is not None else {}
        self._initialized = False
      
        # Placeholders remain the same
        placeholder_num_substations = 14
        placeholder_num_actions = 106
        placeholder_obs_shape = (152,)

        regular_obs_space = Box(
            low=-np.inf, high=np.inf, shape=placeholder_obs_shape, dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            self.high_level_agent_id: gym.spaces.Dict({
                "regular_obs": regular_obs_space,
                "chosen_action": Discrete(placeholder_num_actions)
            }),
            self.low_level_agent_id: gym.spaces.Dict({
                "action_mask": Box(0.0, 1.0, shape=(placeholder_num_actions,), dtype=np.float32),
                "regular_obs": regular_obs_space,
                "chosen_substation": Discrete(placeholder_num_substations)
            })
        })
 
        self.action_space = gym.spaces.Dict({
            self.high_level_agent_id: Discrete(placeholder_num_substations),
            self.low_level_agent_id: Discrete(placeholder_num_actions)
        })

    def _lazy_init(self):
        if self._initialized:
            return

        # --- MODIFIED: Instantiate GridGym directly ---
        # GridGym now encapsulates create_gym_env and the reward accumulation logic
        self.env_gym = GridGym(self.env_config)
        
        self.do_nothing_action_idx = self.env_gym.do_nothing_action_idx
        
        # Get mappings from the underlying CustomGymEnv
        underlying_custom_env = self.env_gym.env_gym
        self.sub_id_to_action_num = get_sub_id_to_action(
            underlying_custom_env.action_space.converter.all_actions, return_action_ix=True
        )
        self.num_to_sub = {i: k for i, k in enumerate(self.sub_id_to_action_num.keys())}
        
        self._initialized = True
        logger.info(f"Hierarchical env initialized, using GridGym as the base.")

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        if not self._initialized:
            self._lazy_init()
        
        self.current_obs, _ = self.env_gym.reset(seed=seed, options=options)
        
        self.agents = [self.high_level_agent_id]
        
        obs = {
            self.high_level_agent_id: {
                "regular_obs": self.current_obs,
                "chosen_action": np.int64(self.do_nothing_action_idx)
            }
        }
        return obs, {}

    def step(self, action_dict: TypingDict[str, Any]) -> tuple[TypingDict, TypingDict, TypingDict, TypingDict, TypingDict]:
        if self.high_level_agent_id in action_dict:
            return self._high_level_step(action_dict[self.high_level_agent_id])
        elif self.low_level_agent_id in action_dict:
            return self._low_level_step(action_dict[self.low_level_agent_id])
        else:
            raise ValueError(f"Unknown agent in action_dict: {action_dict.keys()}")

    def _high_level_step(self, high_level_action: int) -> tuple[TypingDict, TypingDict, TypingDict, TypingDict, TypingDict]:
        self.high_level_action = high_level_action
        self.agents = [self.low_level_agent_id]
        
        action_mask = np.zeros(self.action_space[self.low_level_agent_id].n, dtype=np.float32)
        selected_sub_id = self.num_to_sub.get(high_level_action)
        if selected_sub_id is not None:
            available_actions = self.sub_id_to_action_num.get(selected_sub_id, [])
            if available_actions:
                 action_mask[available_actions] = 1.0

        obs = {
            self.low_level_agent_id: {
                "action_mask": action_mask,
                "regular_obs": self.current_obs,
                "chosen_substation": np.int64(high_level_action),
            }
        }
        
        rewards = {self.low_level_agent_id: 0.0}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        infos = {self.high_level_agent_id: {}, self.low_level_agent_id: {}}
        return obs, rewards, terminateds, truncateds, infos

    # --- MODIFIED: Simplified _low_level_step ---
    def _low_level_step(self, low_level_action: int) -> tuple[TypingDict, TypingDict, TypingDict, TypingDict, TypingDict]:
        # The while loop is now inside self.env_gym.step()
        obs, reward, terminated, truncated, info = self.env_gym.step(low_level_action)
        self.current_obs = obs
        
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}

        rewards = {}
        next_obs = {}

        if not (terminated or truncated):
            self.agents = [self.high_level_agent_id]
            rewards = {self.high_level_agent_id: reward}
            next_obs = {
                self.high_level_agent_id: {
                    "regular_obs": self.current_obs,
                    "chosen_action": np.int64(low_level_action)
                }
            }
        else:
            self.agents = []
            rewards = {
                self.high_level_agent_id: reward,
                self.low_level_agent_id: reward
            }
        
        infos = {
            self.high_level_agent_id: {"steps_in_episode": info.get("steps_in_episode", 0)}, 
            self.low_level_agent_id: info
        }
        return next_obs, rewards, terminateds, truncateds, infos

    def close(self):
        if hasattr(self, 'env_gym') and self.env_gym is not None:
            self.env_gym.close()