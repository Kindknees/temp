# kindknees-temp/grid2op_env/grid_to_gym.py (替換全部內容)

import numpy as np
import grid2op
import os
import logging
from typing import Any, Dict as TypingDict, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Dict

from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import BoxGymnasiumObsSpace

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.rewards import ScaledL2RPNReward
from grid2op_env.utils import CustomDiscreteActions, opponent_kwargs

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
    return space

def _convert_obs_to_float32(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {k: _convert_obs_to_float32(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray) and obs.dtype != np.float32:
        return obs.astype(np.float32)
    return obs

class CustomGymEnv(gym.Env):
    def __init__(self, env_init: grid2op.Environment, action_space_converter: CustomDiscreteActions, 
                 disable_line: int = -1, keep_observations: list = None):
        super().__init__()
        self.init_env = env_init
        self.disable_line = disable_line
        self.reconnect_line = None
        self.action_space = action_space_converter
        
        if keep_observations is None:
            self.gym_obs_space = BoxGymnasiumObsSpace(self.init_env.observation_space)
        else:
            self.gym_obs_space = BoxGymnasiumObsSpace(
                self.init_env.observation_space,
                attr_to_keep=keep_observations
            )
        
        self.observation_space = _recursively_define_float32_space(self.gym_obs_space)
        
        # --- FIX: Initialize last_rho attribute ---
        self.last_rho = None

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        super().reset(seed=seed)
        if seed is not None: self.init_env.seed(seed)

        if self.disable_line != -1:
            done = True
            i = 0
            while done:
                g2op_obs = self.init_env.reset()
                g2op_obs, _, done, info = self.init_env.step(
                    self.init_env.action_space({"set_line_status": (self.disable_line, -1)})
                )
                i += 1
            if i > 1: logging.info(f"Skipped {i-1} chronics for valid start.")
        else:
            g2op_obs = self.init_env.reset()

        # --- FIX: Store rho before flattening ---
        self.last_rho = g2op_obs.rho
        
        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        return _convert_obs_to_float32(gym_obs), {}

    def step(self, gym_action: Any) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        g2op_act = self.action_space.from_gym(gym_action)

        if self.reconnect_line is not None:
            reconnect_act = self.init_env.action_space({"set_line_status": (self.reconnect_line, 1)})
            g2op_act = g2op_act + reconnect_act
            self.reconnect_line = None
                
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)

        # --- FIX: Store rho before flattening ---
        self.last_rho = g2op_obs.rho

        if isinstance(info.get("opponent_attack_line"), np.ndarray):
            if info.get("opponent_attack_duration") == 1:
                line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()
                if len(line_id_attacked) > 0: self.reconnect_line = line_id_attacked[0]

        gym_obs = self.gym_obs_space.to_gym(g2op_obs)
        terminated = done
        truncated = info.get('is_truncated', False)
        
        return _convert_obs_to_float32(gym_obs), float(reward), terminated, truncated, info
    
    def close(self):
        self.init_env.close()

class GridGym(gym.Env):
    def __init__(self, env_config: TypingDict[str, Any]):
        super().__init__()
        self.env_gym, self.do_nothing_action_idx, _, _ = create_gym_env(**env_config)
        
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5
        self.run_until_threshold = env_config.get("run_until_threshold", False)
        self.steps_in_episode = 0
        
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple[TypingDict, TypingDict]:
        obs, info = self.env_gym.reset(seed=seed, options=options)
        self.steps_in_episode = 0 
        return obs, info

    def step(self, action: Any) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        obs, reward, terminated, truncated, info = self.env_gym.step(action)
        self.steps_in_episode += 1
        
        if self.run_until_threshold and not (terminated or truncated):
            cum_reward = reward
            
            # --- FIX: Access rho from the inner env's attribute ---
            current_rho = np.asarray(self.env_gym.last_rho)
            
            while (np.max(current_rho) < self.rho_threshold) and not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env_gym.step(self.do_nothing_action_idx)
                cum_reward += reward
                self.steps_in_episode += 1
                
                # --- FIX: Update rho from the inner env's attribute ---
                current_rho = np.asarray(self.env_gym.last_rho)
                
            reward = cum_reward

        info["steps_in_episode"] = self.steps_in_episode
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env_gym.close()

def create_gym_env(**env_config: Any) -> tuple:
    env_name = env_config.get("env_name", "rte_case14_realistic")
    with_opponent = env_config.get("with_opponent", False)
    keep_observations = env_config.get("keep_observations", None)

    backend = LightSimBackend()
    env = grid2op.make(
        env_name, reward_class=ScaledL2RPNReward, test=False, backend=backend, 
        **(opponent_kwargs if with_opponent else {})
    )
    
    actions_with_redundant, ref_indices, dict_with_redundant = create_action_space(
        env, return_actions_dict=True
    )
    all_actions, do_nothing_ids, all_actions_dict = remove_redundant_actions(
        actions_with_redundant, ref_indices, env.sub_info, dict_with_redundant
    )
    
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)
    custom_action_space = CustomDiscreteActions(converter=converter)

    env_gym = CustomGymEnv(
        env,
        disable_line=env_config.get("disable_line", -1),
        action_space_converter=custom_action_space,
        keep_observations=keep_observations
    )

    do_nothing_action_idx = int(do_nothing_ids[0]) if do_nothing_ids else 0

    return env_gym, do_nothing_action_idx, env, all_actions_dict