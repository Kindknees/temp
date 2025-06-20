import numpy as np
import grid2op
import os
import logging
# [修正] 導入 gymnasium
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import random
import time
from collections import OrderedDict
from typing import Any, Dict as TypingDict, Tuple, List as TypingList

from grid2op.PlotGrid import PlotMatplot
from grid2op.gym_compat import GymEnv # grid2op 內部 wrapper，繼續使用
from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action, opponent_kwargs
from ray.rllib.env import MultiAgentEnv

logger = logging.getLogger(__name__)

class CustomGymEnv(GymEnv):
    """
    A custom GymEnv wrapper that accepts an action space converter at initialization.
    """
    def __init__(self, env_init, disable_line: int = -1, action_space_converter=None):
        # 呼叫父類別的建構函式，但不傳入不被接受的 'action_space_converter' 參數
        super().__init__(env_init)
        
        # 在父類別初始化後，手動設定 action_space
        if action_space_converter is not None:
            self.action_space = action_space_converter
        
        self.disable_line = disable_line
        self.reconnect_line = None

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None) -> tuple:
        # Gymnasium API requires reset to accept seed and options
        super().reset(seed=seed, options=options)
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
        # Gymnasium API requires reset to return obs and info
        return gym_obs, {}

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
        terminated = done
        # [確認] 新增 truncated 旗標
        truncated = info.get('is_truncated', False)
        # [確認] 回傳 (obs, reward, terminated, truncated, info) tuple
        return gym_obs, float(reward), terminated, truncated, info
    
    def close(self):
        """
        Close the underlying grid2op environment.
        This overrides the parent close method to avoid calling close() on the
        observation space, which does not exist.
        """
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

    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None):
        obs, info = self.env_gym.reset(seed=seed, options=options)
        
        if self.run_until_threshold:
            terminated = False
            truncated = False
            self.steps = 0
            # 修正: obs['rho'] 可能是 numpy array
            while (np.max(obs["rho"]) < self.rho_threshold) and not (terminated or truncated):
                obs, _, terminated, truncated, _ = self.env_gym.step(self.do_nothing_actions)
                self.steps += 1
        return obs, info

    # [確認] 'step' 方法回傳值已符合 gymnasium API
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env_gym.step(action)
        
        if self.run_until_threshold:
            cum_reward = reward
            while (np.max(obs["rho"]) < self.rho_threshold) and not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env_gym.step(self.do_nothing_actions)
                cum_reward += reward
                self.steps += 1
            reward = cum_reward
            if terminated or truncated:
                info["steps"] = self.steps
        
        return obs, reward, terminated, truncated, info

class HierarchicalGridGym(MultiAgentEnv):
    """
    一個分層的多智能體環境。

    這個環境中有兩個智能體：
    1. high_level_agent (高層決策者): 它的任務是選擇一個變電站 (substation)。
    2. low_level_agent (低層執行者): 在高層選定變電站後，它從該變電站的可用動作中選擇一個具體動作來執行。

    工作流程：
    1. reset() -> high_level_agent 收到觀測，輪到它行動。
    2. high_level_agent 執行動作 (選擇變電站) -> low_level_agent 收到觀測 (包含可用動作的 mask)，輪到它行動。
    3. low_level_agent 執行動作 (選擇具體操作) -> 環境(Grid2Op)前進一步，episode 可能結束，也可能回到步驟 1，輪到 high_level_agent 為下一步做決策。
    """
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

        # [確認] 使用 gymnasium.spaces
        self.observation_space = Dict({
            self.high_level_agent_id: Dict({
                "regular_obs": self.env_gym.observation_space,
                "chosen_action": Discrete(self.env_gym.action_space.n)
            }),
            self.low_level_agent_id: Dict({
                "action_mask": Box(0, 1, shape=(self.env_gym.action_space.n,), dtype=np.float32),
                "regular_obs": self.env_gym.observation_space,
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

    # [確認] 'reset' 方法簽名和回傳值已符合 MultiAgentEnv API
    def reset(self, *, seed: int | None = None, options: TypingDict[str, Any] | None = None):
        self.cur_obs, _ = self.env_gym.reset(seed=seed, options=options)
        self.high_level_pred = None
        
        obs = {
            self.high_level_agent_id: {
                "regular_obs": self.cur_obs,
                "chosen_action": 0
            }
        }
        # [確認] 回傳 (obs_dict, info_dict)
        infos = {self.high_level_agent_id: {}}
        return obs, infos

    # [確認] 'step' 方法簽名和回傳值已符合 MultiAgentEnv API
    def step(self, action_dict: TypingDict[str, Any]):
        assert len(action_dict) == 1, "分層環境每次只接受一個智能體的動作"
        
        if self.high_level_agent_id in action_dict:
            return self._high_level_step(action_dict[self.high_level_agent_id])
        else:
            return self._low_level_step(action_dict[self.low_level_agent_id])

    def _high_level_step(self, action: int):
        self.high_level_pred = action
        action_mask = self.map_sub_to_mask()
        
        obs = {
            self.low_level_agent_id: {
                "action_mask": action_mask,
                "regular_obs": self.cur_obs,
                "chosen_substation": self.high_level_pred,
            }
        }
        
        rewards = {self.low_level_agent_id: 0.0, self.high_level_agent_id: 0.0}
        # [確認] 使用 terminateds 和 truncateds 字典
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        infos = {self.low_level_agent_id: {}, self.high_level_agent_id: {}}
        
        # [確認] 回傳值順序正確: obs, rewards, terminateds, truncateds, infos
        return obs, rewards, terminateds, truncateds, infos

    def _low_level_step(self, action: int):
        f_obs, f_rew, f_terminated, f_truncated, f_info = self.env_gym.step(action)
        self.info["steps"] = f_info.get("steps", 0)
        self.cur_obs = f_obs

        # [確認] 使用 terminateds 和 truncateds 字典
        terminateds = {"__all__": f_terminated}
        truncateds = {"__all__": f_truncated}

        rewards = {
            self.low_level_agent_id: f_rew,
            self.high_level_agent_id: f_rew
        }
        
        obs = {
            self.high_level_agent_id: {
                "regular_obs": f_obs,
                "chosen_action": action
            }
        }
        infos = {self.high_level_agent_id: self.info.copy(), self.low_level_agent_id: {}}
        
        # [確認] 回傳值順序正確: obs, rewards, terminateds, truncateds, infos
        return obs, rewards, terminateds, truncateds, infos

def create_gym_env(**env_config: Any) -> tuple:
    env_name = env_config.get("env_name", "rte_case14_realistic")
    with_opponent = env_config.get("with_opponent", False)

    if with_opponent:
        env = grid2op.make(env_name, reward_class=L2RPNReward, test=False, backend=LightSimBackend(), **opponent_kwargs)
    else:
        env = grid2op.make(env_name, reward_class=L2RPNReward, test=False, backend=LightSimBackend())
    
    logging.info(f"The environment has {len(env.chronics_handler.subpaths)} chronics.")

    # ... (建立 custom_action_space 的部分不變) ...
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

    # [修正] 移除覆寫 observation_space 的程式碼。
    # 必須保留 env_gym.observation_space 物件的原貌，因為它包含了 grid2op
    # 特有的轉換方法，例如 .to_gym()。將其轉換為純粹的 gymnasium.spaces.Dict
    # 會導致這些方法遺失，從而引發 AttributeError。
    # d = {k: v for k, v in env_gym.observation_space.items()}
    # env_gym.observation_space = gym.spaces.Dict(d)

    return env_gym, do_nothing_actions, env, all_actions_dict