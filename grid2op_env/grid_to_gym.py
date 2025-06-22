import numpy as np
import grid2op
import os
import logging
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import random
from typing import Any, Dict as TypingDict, Tuple

from grid2op.gym_compat import GymEnv # grid2op 內部 wrapper，繼續使用
from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action, opponent_kwargs
from ray.rllib.env import MultiAgentEnv

logger = logging.getLogger(__name__)

def convert_obs_to_float32(obs):
    """
    Recursively casts all numpy arrays in the observation dict to float32.
    This is crucial for compatibility with Gymnasium and RLlib.
    """
    if isinstance(obs, dict):
        new_obs = type(obs)()
        for k, v in obs.items():
            new_obs[k] = convert_obs_to_float32(v)
        return new_obs
    elif isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    else:
        return obs

class CustomGymEnv(GymEnv):
    """
    A custom GymEnv wrapper that accepts an action space converter at initialization
    and handles the crucial dtype conversion for observations.
    """
    def __init__(self, env_init, disable_line: int = -1, action_space_converter=None):
        super().__init__(env_init)
        
        if action_space_converter is not None:
            self.action_space = action_space_converter
        
        self.disable_line = disable_line
        self.reconnect_line = None

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple:
        if seed is not None:
            self.init_env.seed(seed)

        if self.disable_line == -1:
            g2op_obs = self.init_env.reset()
        else:
            done = True
            i = -1
            while done:
                g2op_obs = self.init_env.reset()
                g2op_obs, _, done, info = self.init_env.step(self.init_env.action_space({"set_line_status": (self.disable_line, -1)}))
                i += 1
            if i!= 0:
                logging.info(f"Had to skip {i} times to get a valid observation")
        
        gym_obs = self.observation_space.to_gym(g2op_obs)

        # [核心修正] 在觀測值返回給上層 (Gymnasium/RLlib) 之前，立刻轉換其 dtype
        gym_obs_float32 = convert_obs_to_float32(gym_obs)

        return gym_obs_float32, {}

    def step(self, gym_action: Any) -> tuple[Any, float, bool, bool, TypingDict[str, Any]]:
        g2op_act = self.action_space.from_gym(gym_action)
        if self.reconnect_line is not None:
            reconnect_act = self.init_env.action_space({"set_line_status": (self.reconnect_line, 1)})
            g2op_act = g2op_act + reconnect_act
            self.reconnect_line = None
                
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)

        if isinstance(info.get("opponent_attack_line"), np.ndarray):
            if info.get("opponent_attack_duration") == 1:
                line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()
                if len(line_id_attacked) > 0:
                    self.reconnect_line = line_id_attacked

        gym_obs = self.observation_space.to_gym(g2op_obs)
        
        # [核心修正] 在觀測值返回給上層 (Gymnasium/RLlib) 之前，立刻轉換其 dtype
        gym_obs_float32 = convert_obs_to_float32(gym_obs)

        terminated = done
        truncated = info.get('is_truncated', False)
        
        return gym_obs_float32, float(reward), terminated, truncated, info
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()

class Grid_Gym(gym.Env):
    def __init__(self, env_config: TypingDict[str, Any]):
        self.env_gym, self.do_nothing_actions, self.org_env, self.all_actions_dict = create_gym_env(**env_config)
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5
        self.run_until_threshold = env_config.get("run_until_threshold", False)
        self.steps = 0
        
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space
        if isinstance(self.do_nothing_actions, (np.ndarray, np.number)):
            self.do_nothing_actions = int(np.asarray(self.do_nothing_actions).item())

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None):
        # [修正] 移除這裡的轉換，因為已在 CustomGymEnv 中完成
        obs, info = self.env_gym.reset(seed=seed, options=options)
        self.steps = 0 
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env_gym.step(action)
        
        if self.run_until_threshold:
            cum_reward = reward
            # 確保 obs["rho"] 是 numpy array 以使用 .max()
            current_rho = np.asarray(obs["rho"])
            while (np.max(current_rho) < self.rho_threshold) and not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env_gym.step(self.do_nothing_actions)
                cum_reward += reward
                self.steps += 1
                current_rho = np.asarray(obs["rho"])
            reward = cum_reward
            if terminated or truncated:
                info["steps"] = self.steps
        
        # [修正] 移除這裡的轉換，因為已在 CustomGymEnv 中完成
        return obs, reward, terminated, truncated, info

class HierarchicalGridGym(MultiAgentEnv):
    def __init__(self, env_config: TypingDict[str, Any]):
        super().__init__()
        self._skip_env_checking = True
        self.env_gym = Grid_Gym(env_config)
        
        self.low_level_agent_id = "choose_action_agent"
        self.high_level_agent_id = "choose_substation_agent"
        
        self._agent_ids = {self.low_level_agent_id, self.high_level_agent_id}

        self.sub_id_to_action_num = get_sub_id_to_action(self.env_gym.action_space.converter.all_actions, return_action_ix=True)
        self.num_to_sub = {i: k for i, k in enumerate(self.sub_id_to_action_num.keys())}
        
        self.cur_obs = None
        self.high_level_pred = None
        self.info = {"steps": 0}

        # [核心修正] 確保 observation space 的 dtype 明確為 float32
        # 這一步對於讓 RLlib 的模型初始化和檢查通過至關重要
        regular_obs_space_items = {}
        for k, v in self.env_gym.observation_space.items():
            if isinstance(v, gym.spaces.Box):
                regular_obs_space_items[k] = gym.spaces.Box(
                    low=v.low.astype(np.float32), 
                    high=v.high.astype(np.float32), 
                    shape=v.shape, 
                    dtype=np.float32
                )
            else:
                regular_obs_space_items[k] = v
        regular_obs_space = gym.spaces.Dict(regular_obs_space_items)


        self.observation_space = Dict({
            self.high_level_agent_id: Dict({
                "regular_obs": regular_obs_space,
                "chosen_action": Discrete(self.env_gym.action_space.n)
            }),
            self.low_level_agent_id: Dict({
                "action_mask": Box(0, 1, shape=(self.env_gym.action_space.n,), dtype=np.float32),
                "regular_obs": regular_obs_space,
                "chosen_substation": Discrete(len(self.num_to_sub))
            })
        })
        
        self.action_space = Dict({
            self.high_level_agent_id: Discrete(len(self.num_to_sub)),
            self.low_level_agent_id: self.env_gym.action_space
        })

    def map_sub_to_mask(self) -> np.ndarray:
        action_mask = np.zeros(self.env_gym.action_space.n, dtype=np.float32)
        if self.high_level_pred is not None:
            modified_sub = self.num_to_sub.get(self.high_level_pred)
            if modified_sub is not None:
                aval_actions = self.sub_id_to_action_num.get(modified_sub, [])
                action_mask[aval_actions] = 1.0
        return action_mask

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None):
        self.agents = [self.high_level_agent_id]
        
        self.cur_obs, _ = self.env_gym.reset(seed=seed, options=options)
        self.high_level_pred = None
        
        obs = {
            self.high_level_agent_id: {
                "regular_obs": self.cur_obs,
                "chosen_action": np.int64(0) 
            }
        }
        infos = {self.high_level_agent_id: {}}
        return obs, infos

    def step(self, action_dict: TypingDict[str, Any]):
        agent_id = list(action_dict.keys())[0]
        
        if agent_id == self.high_level_agent_id:
            self.agents = [self.low_level_agent_id]
            return self._high_level_step(action_dict[self.high_level_agent_id])
        else:
            self.agents = [self.high_level_agent_id]
            return self._low_level_step(action_dict[self.low_level_agent_id])

    def _high_level_step(self, action: int):
        self.high_level_pred = action
        action_mask = self.map_sub_to_mask()
        
        obs = {
            self.low_level_agent_id: {
                "action_mask": action_mask,
                "regular_obs": self.cur_obs,
                "chosen_substation": np.int64(self.high_level_pred),
            }
        }
        
        rewards = {self.low_level_agent_id: 0.0}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        infos = {self.low_level_agent_id: {}}
        
        return obs, rewards, terminateds, truncateds, infos

    def _low_level_step(self, action: int):
        f_obs, f_rew, f_terminated, f_truncated, f_info = self.env_gym.step(action)
        self.info["steps"] = f_info.get("steps", 0)
        self.cur_obs = f_obs

        terminateds = {"__all__": f_terminated}
        truncateds = {"__all__": f_truncated}

        rewards = {
            self.low_level_agent_id: f_rew,
            self.high_level_agent_id: f_rew
        }
        
        obs = {
            self.high_level_agent_id: {
                "regular_obs": f_obs,
                "chosen_action": np.int64(action)
            }
        }
        infos = {self.high_level_agent_id: self.info.copy()}
        
        return obs, rewards, terminateds, truncateds, infos

def create_gym_env(**env_config: Any) -> tuple:
    env_name = env_config.get("env_name", "rte_case14_realistic")
    with_opponent = env_config.get("with_opponent", False)

    if with_opponent:
        env = grid2op.make(env_name, reward_class=L2RPNReward, test=False, backend=LightSimBackend(), **opponent_kwargs)
    else:
        env = grid2op.make(env_name, reward_class=L2RPNReward, test=False, backend=LightSimBackend())
    
    logging.info(f"The environment has {len(env.chronics_handler.subpaths)} chronics.")

    custom_action_space = None
    all_actions_dict = {}
    do_nothing_actions = None
    if env_config.get("medha_actions", True):
        logging.info("Using the action space defined by Medha!")
        all_actions_with_redundant, ref_indices, all_actions_dict_with_redundant = create_action_space(env, return_actions_dict=True)
        all_actions, do_nothing_actions, all_actions_dict = remove_redundant_actions(all_actions_with_redundant, ref_indices, env.sub_info, all_actions_dict_with_redundant)
        converter = IdToAct(env.action_space)
        converter.init_converter(all_actions=all_actions)
        custom_action_space = CustomDiscreteActions(converter=converter)

    env_gym = CustomGymEnv(env, 
                           disable_line=env_config.get("disable_line", -1),
                           action_space_converter=custom_action_space)

    logging.info("Environment successfully converted to Gym")
    
    return env_gym, do_nothing_actions, env, all_actions_dict