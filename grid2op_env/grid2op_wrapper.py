# 檔案: kindknees-temp/grid2op_env/grid2op_wrapper.py

import numpy as np
import grid2op
from typing import Any, Dict as TypingDict, Tuple, Set
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

from ray.rllib.env import MultiAgentEnv
from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action, opponent_kwargs
from grid2op_env.rewards import ScaledL2RPNReward
from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import BoxGymnasiumObsSpace

import logging
logger = logging.getLogger(__name__)


def _recursively_define_float32_space(space: gym.Space) -> gym.Space:
    if isinstance(space, Box):
        return Box(
           low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32
        )
    elif isinstance(space, Dict):
        return Dict({
            k: _recursively_define_float32_space(v) for k, v in space.spaces.items()
        })
    elif isinstance(space, gym.spaces.Tuple):
         return gym.spaces.Tuple(
            tuple(_recursively_define_float32_space(s) for s in space.spaces)
        )
    return space

# ####################################################################
# REPLACED: 將原本不夠穩健的轉換函式，替換為能處理陣列、純量和列表的更強版本
# ####################################################################
def _convert_obs_to_float32(obs: Any) -> Any:
    """
    遞迴地走訪觀測值(字典、列表/元組或陣列)，並將所有浮點數Numpy物件轉換為float32。
    這個版本比先前更穩健。
    """
    if isinstance(obs, dict):
        # 處理字典結構的觀測值
        return {k: _convert_obs_to_float32(v) for k, v in obs.items()}
    elif isinstance(obs, (np.ndarray, np.generic)):
        # 如果是Numpy陣列或純量 (np.float64, np.int64 等)
        if np.issubdtype(obs.dtype, np.floating):
            # 並且其資料型態為浮點數，則轉換為float32
            return obs.astype(np.float32)
        else:
            # 如果是整數等其他型態，則保持不變
            return obs
    elif isinstance(obs, (list, tuple)):
        # 如果是列表或元組，遞迴處理其內部每一個元素
        return type(obs)(_convert_obs_to_float32(v) for v in obs)
    else:
        # 其他型態（如原生 Python int, float）保持不變
        return obs
# ####################################################################
# END REPLACEMENT
# ####################################################################


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

        env_name = self.env_config.get("env_name", "rte_case14_realistic")
        with_opponent = self.env_config.get("with_opponent", False)
        
        keep_observations = self.env_config.get("keep_observations", None)
        disable_line = self.env_config.get("disable_line", -1)

        backend = LightSimBackend()
        self.grid2op_env = grid2op.make(
            env_name, reward_class=ScaledL2RPNReward, test=False, backend=backend,
            **(opponent_kwargs if with_opponent else {})
        )

        actions_with_redundant, ref_indices, dict_with_redundant = create_action_space(
            self.grid2op_env, return_actions_dict=True
        )
        all_actions, do_nothing_ids, _ = remove_redundant_actions(
            actions_with_redundant, ref_indices, self.grid2op_env.sub_info, dict_with_redundant
        )
        
        converter = IdToAct(self.grid2op_env.action_space)
        converter.init_converter(all_actions=all_actions)
        self.action_converter = CustomDiscreteActions(converter=converter)
        
        real_num_actions = len(self.action_converter.converter.all_actions)
        if real_num_actions != self.action_space.spaces[self.low_level_agent_id].n:
            raise ValueError(
                f"Real number of actions ({real_num_actions}) does not match placeholder "
                f"size ({self.action_space.spaces[self.low_level_agent_id].n}). "
                "Update placeholder_num_actions in __init__."
            )

        if keep_observations is None:
            self.gym_obs_space = BoxGymnasiumObsSpace(self.grid2op_env.observation_space)
        else:
            self.gym_obs_space = BoxGymnasiumObsSpace(
                self.grid2op_env.observation_space, attr_to_keep=keep_observations
            )
        
        self.do_nothing_action_idx = int(do_nothing_ids[0]) if do_nothing_ids else 0
        self.disable_line = disable_line
        self.reconnect_line = None
        
        self.sub_id_to_action_num = get_sub_id_to_action(
            self.action_converter.converter.all_actions, return_action_ix=True
        )
        self.num_to_sub = {i: k for i, k in enumerate(self.sub_id_to_action_num.keys())}
        
        self._initialized = True
        logger.info(f"Grid2Op environment initialized with {len(self.num_to_sub)} substations and {real_num_actions} actions")

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] |
 None = None) -> tuple[TypingDict, TypingDict]:
        if not self._initialized:
            self._lazy_init()
        
        if seed is not None:
            self.grid2op_env.seed(seed)
            
        if self.disable_line != -1:
            done = True
            i = 0
            while done:
                g2op_obs = self.grid2op_env.reset()
                g2op_obs, _, done, info = self.grid2op_env.step(
                    self.grid2op_env.action_space({"set_line_status": (self.disable_line, -1)})
                )
                i += 1
            if i > 1:
                logger.info(f"Skipped {i-1} chronic(s) to get a valid start after disabling line {self.disable_line}")
        else:
            g2op_obs = self.grid2op_env.reset()
        
        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        self.current_obs = _convert_obs_to_float32(gym_obs)
        self.steps_in_episode = 0
        
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
        
        action_mask = np.zeros(self.action_space.spaces[self.low_level_agent_id].n, dtype=np.float32)
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
        
        rewards = {self.high_level_agent_id: 0.0}
        
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        infos = {self.high_level_agent_id: {}}
        return obs, rewards, terminateds, truncateds, infos

    def _low_level_step(self, low_level_action: int) -> tuple[TypingDict, TypingDict, TypingDict, TypingDict, TypingDict]:
        g2op_act = self.action_converter.from_gym(low_level_action)
        
        if self.reconnect_line is not None:
            reconnect_act = self.grid2op_env.action_space({"set_line_status": (self.reconnect_line, 1)})
            g2op_act = g2op_act + reconnect_act
            self.reconnect_line = None
            
        g2op_obs, reward, done, info = self.grid2op_env.step(g2op_act)
        self.steps_in_episode += 1

        run_until_threshold = self.env_config.get("run_until_threshold", False)
        rho_threshold = self.env_config.get("rho_threshold", 0.95) - 1e-5
        
        cum_reward = reward
        if run_until_threshold and not done:
            current_rho = np.asarray(g2op_obs.rho)
            do_nothing_g2op_act = self.action_converter.from_gym(self.do_nothing_action_idx)
            
            while (np.max(current_rho) < rho_threshold) and not done:
                g2op_obs, reward_do_nothing, done, info = self.grid2op_env.step(do_nothing_g2op_act)
                cum_reward += reward_do_nothing
                self.steps_in_episode += 1
                current_rho = np.asarray(g2op_obs.rho)
        
        final_reward = cum_reward
        
        if isinstance(info.get("opponent_attack_line"), np.ndarray):
            if info.get("opponent_attack_duration") == 1:
                line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()
                if len(line_id_attacked) > 0:
                    self.reconnect_line = line_id_attacked[0]
        
        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        self.current_obs = _convert_obs_to_float32(gym_obs)
        
        terminated = done
        truncated = info.get('is_truncated', False)
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}

        rewards = {
            self.high_level_agent_id: final_reward,
            self.low_level_agent_id: final_reward
        }

        next_obs = {}
        if not (terminated or truncated):
            self.agents = [self.high_level_agent_id]
            
            next_obs = {
                self.high_level_agent_id: {
                    "regular_obs": self.current_obs,
                    "chosen_action": np.int64(low_level_action)
                }
            }
        else:
            self.agents = []
        
        infos = {
            self.high_level_agent_id: {"steps_in_episode": self.steps_in_episode}, 
            self.low_level_agent_id: info
        }
        return next_obs, rewards, terminateds, truncateds, infos

    def close(self):
        if hasattr(self, 'grid2op_env') and self.grid2op_env is not None:
            self.grid2op_env.close()