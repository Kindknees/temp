import numpy as np
import grid2op
import os
import logging
from typing import Any, Dict as TypingDict, Tuple

# 使用現代的 Gymnasium API
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend
from ray.rllib.env import MultiAgentEnv

# Import Grid2Op's gymnasium compatibility classes
from grid2op.gym_compat import BoxGymnasiumObsSpace, DiscreteActSpace, GymEnv

# 專案模組導入
from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action, opponent_kwargs

# 設定日誌
logger = logging.getLogger(__name__)

def _recursively_define_float32_space(space: gym.Space) -> gym.Space:
    """
    遞迴地將 observation space 中的所有 Box 空間的 dtype 設定為 float32。
    """
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

def _convert_obs_to_float32(obs: Any) -> Any:
    """遞迴地將觀測字典中的 numpy 陣列轉換為 float32。"""
    if isinstance(obs, dict):
        converted_obs = {}
        for k, v in obs.items():
            converted_obs[k] = _convert_obs_to_float32(v)
        return converted_obs
    elif isinstance(obs, np.ndarray) and obs.dtype != np.float32:
        return obs.astype(np.float32)
    else:
        return obs


class CustomGymEnv(gym.Env):
    """
    一個基礎的 Gymnasium 環境包裝器，用於 Grid2Op。
    使用 Grid2Op 的內建 gymnasium 相容性。
    """
    def __init__(self, env_init: grid2op.Environment, action_space_converter: CustomDiscreteActions, 
                 disable_line: int = -1, keep_observations: list = None):
        super().__init__()
        self.init_env = env_init
        self.disable_line = disable_line
        self.reconnect_line = None
        
        # 設定動作空間
        self.action_space = action_space_converter
        
        # 使用 Grid2Op 的 BoxGymnasiumObsSpace 來創建觀測空間
        # 根據 keep_observations 參數來決定要保留哪些觀測
        if keep_observations is None:
            # 使用所有可用的觀測
            self.gym_obs_space = BoxGymnasiumObsSpace(self.init_env.observation_space)
        else:
            # 只保留指定的觀測
            self.gym_obs_space = BoxGymnasiumObsSpace(
                self.init_env.observation_space,
                attr_to_keep=keep_observations
            )
        
        self.observation_space = _recursively_define_float32_space(self.gym_obs_space)

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        super().reset(seed=seed)
        
        if seed is not None:
            self.init_env.seed(seed)

        # 根據 disable_line 參數重設環境
        if self.disable_line != -1:
            done = True
            i = 0
            while done:
                g2op_obs = self.init_env.reset()
                g2op_obs, _, done, info = self.init_env.step(
                    self.init_env.action_space({"set_line_status": (self.disable_line, -1)})
                )
                i += 1
            if i > 1:
                logging.info(f"Skipped {i-1} chronic(s) to get a valid start after disabling line {self.disable_line}")
        else:
            g2op_obs = self.init_env.reset()

        # 使用 Grid2Op 的轉換方法
        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        return _convert_obs_to_float32(gym_obs), {}

    def step(self, gym_action: Any) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        g2op_act = self.action_space.from_gym(gym_action)

        # 如果有待重新連接的線路，將其加入本次動作
        if self.reconnect_line is not None:
            reconnect_act = self.init_env.action_space({"set_line_status": (self.reconnect_line, 1)})
            g2op_act = g2op_act + reconnect_act
            self.reconnect_line = None
                
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)

        # 檢查對手攻擊
        if isinstance(info.get("opponent_attack_line"), np.ndarray):
            if info.get("opponent_attack_duration") == 1:
                line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()
                if len(line_id_attacked) > 0:
                    self.reconnect_line = line_id_attacked[0]

        # 使用 Grid2Op 的轉換方法
        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        terminated = done
        truncated = info.get('is_truncated', False)
        
        return _convert_obs_to_float32(gym_obs), float(reward), terminated, truncated, info
    
    def close(self):
        self.init_env.close()


class GridGym(gym.Env):
    """
    在 CustomGymEnv 之上增加一層邏輯，主要用於實驗中的 "run_until_threshold" 功能。
    """
    def __init__(self, env_config: TypingDict[str, Any]):
        super().__init__()
        self.env_gym, self.do_nothing_action_idx, _, _ = create_gym_env(**env_config)
        
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5
        self.run_until_threshold = env_config.get("run_until_threshold", False)
        self.steps = 0
        
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        obs, info = self.env_gym.reset(seed=seed, options=options)
        self.steps = 0 
        return obs, info

    def step(self, action: Any) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        obs, reward, terminated, truncated, info = self.env_gym.step(action)
        self.steps += 1
        
        # 如果啟用，自動執行 "do-nothing" 直到 rho 超過閾值
        if self.run_until_threshold and not (terminated or truncated):
            cum_reward = reward
            current_rho = np.asarray(obs["rho"])
            while (np.max(current_rho) < self.rho_threshold) and not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env_gym.step(self.do_nothing_action_idx)
                cum_reward += reward
                self.steps += 1
                current_rho = np.asarray(obs["rho"])
            reward = cum_reward

        info["steps_in_episode"] = self.steps
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env_gym.close()


class HierarchicalGridGym(MultiAgentEnv):
    """
    將電網控制問題轉換為一個階層式的多智慧體環境。
    """
    def __init__(self, env_config: TypingDict[str, Any]):
        super().__init__()
        self._skip_env_checking = True
        
        # 建立底層的單一智慧體環境
        self.env_gym, self.do_nothing_action_idx, _, all_actions_dict = create_gym_env(**env_config)
        
        # 定義高層和低層智慧體的名稱
        self.high_level_agent_id = "choose_substation_agent"
        self.low_level_agent_id = "choose_action_agent"
        self._agent_ids = {self.high_level_agent_id, self.low_level_agent_id}
        
        # Required by RLlib: define possible agents and current agents
        self.possible_agents = [self.high_level_agent_id, self.low_level_agent_id]
        self.agents = []  # Will be updated during reset and step

        # 建立從變電站 ID 到其對應動作索引的映射
        self.sub_id_to_action_num = get_sub_id_to_action(
            self.env_gym.action_space.converter.all_actions, 
            return_action_ix=True
        )
        self.num_to_sub = {i: k for i, k in enumerate(self.sub_id_to_action_num.keys())}
        num_substations = len(self.num_to_sub)
        num_actions = self.env_gym.action_space.n

        # 狀態變數
        self.current_obs = None
        self.high_level_action = None
        self.steps_in_episode = 0

        # 定義階層式智慧體的觀測與動作空間
        regular_obs_space = self.env_gym.observation_space

        self.observation_space = Dict({
            self.high_level_agent_id: Dict({
                "regular_obs": regular_obs_space,
                "chosen_action": Discrete(num_actions)
            }),
            self.low_level_agent_id: Dict({
                "action_mask": Box(0.0, 1.0, shape=(num_actions,), dtype=np.float32),
                "regular_obs": regular_obs_space,
                "chosen_substation": Discrete(num_substations)
            })
        })
        
        self.action_space = Dict({
            self.high_level_agent_id: Discrete(num_substations),
            self.low_level_agent_id: self.env_gym.action_space
        })

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        self.current_obs, _ = self.env_gym.reset(seed=seed, options=options)
        self.steps_in_episode = 0
        
        # Episode 開始時，只有高層智慧體需要動作
        # IMPORTANT: Update agents list to indicate high-level agent is active
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
        """處理高層智慧體的動作：選擇變電站"""
        self.high_level_action = high_level_action
        
        # Update current agents - now low-level agent is active
        self.agents = [self.low_level_agent_id]
        
        # 建立 action_mask
        action_mask = np.zeros(self.env_gym.action_space.n, dtype=np.float32)
        selected_sub_id = self.num_to_sub.get(high_level_action)
        if selected_sub_id is not None:
            available_actions = self.sub_id_to_action_num.get(selected_sub_id, [])
            action_mask[available_actions] = 1.0

        # 準備給低層智慧體的觀測
        obs = {
            self.low_level_agent_id: {
                "action_mask": action_mask,
                "regular_obs": self.current_obs,
                "chosen_substation": np.int64(high_level_action),
            }
        }
        
        rewards = {self.low_level_agent_id: 0.0, self.high_level_agent_id: 0.0}
        terminateds, truncateds = {"__all__": False}, {"__all__": False}
        
        return obs, rewards, terminateds, truncateds, {}

    def _low_level_step(self, low_level_action: int) -> tuple[TypingDict, TypingDict, TypingDict, TypingDict, TypingDict]:
        """處理低層智慧體的動作：在給定變電站內選擇具體動作並與環境互動"""
        obs, reward, terminated, truncated, info = self.env_gym.step(low_level_action)
        self.current_obs = obs
        self.steps_in_episode += 1

        # 兩個智慧體都獲得相同的獎勵
        rewards = {
            self.high_level_agent_id: reward,
            self.low_level_agent_id: reward
        }
        
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}

        # If episode is not done, update agents for next step
        if not (terminated or truncated):
            # Update current agents - high-level agent is active again
            self.agents = [self.high_level_agent_id]
            
            # 準備下一個時間步給高層智慧體的觀測
            next_obs = {
                self.high_level_agent_id: {
                    "regular_obs": self.current_obs,
                    "chosen_action": np.int64(low_level_action)
                }
            }
        else:
            # Episode is done, no active agents
            self.agents = []
            next_obs = {}
        
        infos = {
            self.high_level_agent_id: {"steps_in_episode": self.steps_in_episode},
            self.low_level_agent_id: {}
        }

        return next_obs, rewards, terminateds, truncateds, infos


def create_gym_env(**env_config: Any) -> tuple:
    """
    環境工廠函式：初始化 Grid2Op，並使用 medha_action_space 建立自訂動作空間。
    """
    env_name = env_config.get("env_name", "rte_case14_realistic")
    with_opponent = env_config.get("with_opponent", False)
    keep_observations = env_config.get("keep_observations", None)

    # 根據是否需要對手來建立 Grid2Op 環境
    backend = LightSimBackend()
    if with_opponent:
        env = grid2op.make(env_name, reward_class=L2RPNReward(), test=False, backend=backend, **opponent_kwargs)
    else:
        env = grid2op.make(env_name, reward_class=L2RPNReward(), test=False, backend=backend)
    
    logging.info(f"Grid2Op environment created with {len(env.chronics_handler.subpaths)} chronics.")

    # 使用 medha_actions 建立自訂動作空間
    all_actions, do_nothing_ids, all_actions_dict = [], [], {}
    if env_config.get("medha_actions", True):
        logging.info("Creating custom action space using Medha's method.")
        actions_with_redundant, ref_indices, dict_with_redundant = create_action_space(
            env, return_actions_dict=True
        )
        all_actions, do_nothing_ids, all_actions_dict = remove_redundant_actions(
            actions_with_redundant, ref_indices, env.sub_info, dict_with_redundant
        )
    
    # Grid2Op 的 IdToAct 轉換器
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)
    custom_action_space = CustomDiscreteActions(converter=converter)

    env_gym = CustomGymEnv(
        env,
        disable_line=env_config.get("disable_line", -1),
        action_space_converter=custom_action_space,
        keep_observations=keep_observations
    )

    logging.info("Environment successfully wrapped into Gymnasium-compatible format.")
    
    # 確保 do_nothing_ids 是一個單一整數
    do_nothing_action_idx = int(do_nothing_ids[0]) if do_nothing_ids else 0

    return env_gym, do_nothing_action_idx, env, all_actions_dict