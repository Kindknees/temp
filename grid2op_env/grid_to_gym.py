import numpy as np
import grid2op
import os
import logging
import gymnasium as gym
from collections import OrderedDict
from typing import Any, Dict as TypingDict, Tuple, List as TypingList

from grid2op.PlotGrid import PlotMatplot
from grid2op.gym_compat import GymEnv, MultiToTupleConverter, DiscreteActSpace, ScalerAttrConverter
from grid2op.Reward import L2RPNReward, CombinedReward
from grid2op.Converter import IdToAct
from grid2op.dtypes import dt_float
from lightsim2grid import LightSimBackend
from gymnasium.spaces import Box, Discrete

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action, opponent_kwargs
from grid2op_env.rewards import ScaledL2RPNReward, CloseToOverflowReward, LinesReconnectedReward
from models.utils import get_sub_id_to_elem_id
from definitions import ROOT_DIR
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog

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

    def step(self, gym_action: Any) -> tuple:
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
        # Gymnasium API: done is split into terminated and truncated
        terminated = done
        truncated = info.get('is_truncated', False) # Assume info might contain truncation info
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
            while (max(obs["rho"]) < self.rho_threshold) and not (terminated or truncated):
                obs, _, terminated, truncated, _ = self.env_gym.step(self.do_nothing_actions)
                self.steps += 1
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env_gym.step(action)
        
        if self.run_until_threshold:
            cum_reward = reward
            while (max(obs["rho"]) < self.rho_threshold) and not (terminated or truncated):
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
        # RLlib 在較新版本中會自動進行環境檢查，若自訂環境複雜，可設為 True 跳過
        self._skip_env_checking = True
        
        # 假設 Grid_Gym 是一個已經符合 gymnasium API 的單智能體環境
        # 並且 get_sub_id_to_action 是您專案中的一個輔助函式
        # from grid2op_env.grid_to_gym import Grid_Gym as Grid_Gym
        # from experiments.utils import get_sub_id_to_action
        self.env_gym = Grid_Gym(env_config)
        
        # 定義智能體 ID
        self.low_level_agent_id = "choose_action_agent"
        self.high_level_agent_id = "choose_substation_agent"
        self._agent_ids = {self.low_level_agent_id, self.high_level_agent_id}

        # 動作空間相關的映射
        self.sub_id_to_action_num = get_sub_id_to_action(self.env_gym.action_space.converter.all_actions, return_action_ix=True)
        self.num_to_sub = {i: k for i, k in enumerate(self.sub_id_to_action_num.keys())}
        
        # 狀態變數
        self.cur_obs = None
        self.high_level_pred = None
        self.info = {"steps": 0}

        # 定義觀測空間 (Observation Space)
        self.observation_space = gym.spaces.Dict({
            self.high_level_agent_id: gym.spaces.Dict({
                "regular_obs": self.env_gym.observation_space,
                # 低層智能體上一步選擇的具體動作
                "chosen_action": gym.spaces.Discrete(self.env_gym.action_space.n)
            }),
            self.low_level_agent_id: gym.spaces.Dict({
                # 可用動作的遮罩 (mask)
                "action_mask": gym.spaces.Box(0, 1, shape=(self.env_gym.action_space.n,), dtype=np.float32),
                "regular_obs": self.env_gym.observation_space,
                # 高層智能體選擇的變電站
                "chosen_substation": gym.spaces.Discrete(len(self.num_to_sub))
            })
        })
        
        # 定義動作空間 (Action Space)
        self.action_space = gym.spaces.Dict({
            self.high_level_agent_id: gym.spaces.Discrete(len(self.num_to_sub)),
            self.low_level_agent_id: self.env_gym.action_space
        })

    def map_sub_to_mask(self) -> np.ndarray:
        """根據高層智能體的預測，生成低層智能體的動作遮罩。"""
        action_mask = np.zeros(self.env_gym.action_space.n, dtype=np.float32)
        if self.high_level_pred is not None:
            modified_sub = self.num_to_sub.get(self.high_level_pred)
            if modified_sub is not None:
                aval_actions = self.sub_id_to_action_num.get(modified_sub, [])
                action_mask[aval_actions] = 1.0
        return action_mask

    def reset(self, *, seed=None, options=None):
        """
        重置環境，符合 gymnasium API。
        返回高層智能體的初始觀測和資訊。
        """
        self.cur_obs, _ = self.env_gym.reset(seed=seed, options=options)
        self.high_level_pred = None
        
        # 在 episode 開始時，只有高層智能體需要行動
        obs = {
            self.high_level_agent_id: {
                "regular_obs": self.cur_obs,
                "chosen_action": 0  # 初始時沒有已選擇的動作，設為0
            }
        }
        # info 也應對應提供觀測的智能體
        infos = {self.high_level_agent_id: {}}
        return obs, infos

    def step(self, action_dict: TypingDict[str, Any]):
        """
        根據傳入的 action_dict 執行一步。
        action_dict 中每次只應包含一個智能體的動作。
        """
        assert len(action_dict) == 1, "分層環境每次只接受一個智能體的動作"
        
        if self.high_level_agent_id in action_dict:
            # 如果是高層智能體在行動
            return self._high_level_step(action_dict[self.high_level_agent_id])
        else:
            # 否則就是低層智能體在行動
            return self._low_level_step(action_dict[self.low_level_agent_id])

    def _high_level_step(self, action: int):
        """處理高層智能體的動作 (選擇變電站)，並返回低層智能體的觀測。"""
        self.high_level_pred = action
        action_mask = self.map_sub_to_mask()
        
        # 高層行動後，輪到低層行動
        obs = {
            self.low_level_agent_id: {
                "action_mask": action_mask,
                "regular_obs": self.cur_obs,
                "chosen_substation": self.high_level_pred,
            }
        }
        
        # 在這個中間步驟，通常沒有獎勵，episode 也不會結束
        rewards = {self.low_level_agent_id: 0.0, self.high_level_agent_id: 0.0}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        infos = {self.low_level_agent_id: {}, self.high_level_agent_id: {}} # 為兩個智能體都提供 info 字典
        
        return obs, rewards, terminateds, truncateds, infos

    def _low_level_step(self, action: int):
        """處理低層智能體的動作 (執行具體操作)，並返回下一個狀態的高層觀測。"""
        # 將動作傳遞給底層的單智能體環境
        f_obs, f_rew, f_terminated, f_truncated, f_info = self.env_gym.step(action)
        self.info["steps"] = f_info.get("steps", 0)
        self.cur_obs = f_obs

        # episode 是否結束的標誌
        terminateds = {"__all__": f_terminated}
        truncateds = {"__all__": f_truncated}

        # 獎勵同時給予兩個智能體
        rewards = {
            self.low_level_agent_id: f_rew,
            self.high_level_agent_id: f_rew
        }
        
        # 如果 episode 沒有結束，則輪到高層智能體為下一步做決策
        # 如果 episode 結束了，RLlib 會自動呼叫 reset，這個 obs 會被忽略
        obs = {
            self.high_level_agent_id: {
                "regular_obs": f_obs,
                "chosen_action": action
            }
        }
        infos = {self.high_level_agent_id: self.info.copy(), self.low_level_agent_id: {}} # 提供對應的 info
        
        return obs, rewards, terminateds, truncateds, infos

def create_gym_env(**env_config: Any) -> tuple:
    env_name = env_config.get("env_name", "rte_case14_realistic")
    with_opponent = env_config.get("with_opponent", False)

    if with_opponent:
        env = grid2op.make(env_name, reward_class=ScaledL2RPNReward, test=False, backend=LightSimBackend(), **opponent_kwargs)
    else:
        env = grid2op.make(env_name, reward_class=ScaledL2RPNReward, test=False, backend=LightSimBackend())
    
    logging.info(f"The environment has {len(env.chronics_handler.subpaths)} chronics.")

    # Create the custom action space converter BEFORE initializing the GymEnv
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

    # Now, create the GymEnv wrapper, passing the custom action space converter with the correct keyword
    env_gym = CustomGymEnv(env, 
                           disable_line=env_config.get("disable_line", -1),
                           action_space_converter=custom_action_space)

    logging.info("Environment successfully converted to Gym")

    # Process observation space
    d = {k: v for k, v in env_gym.observation_space.items()}
    env_gym.observation_space = gym.spaces.Dict(d)

    return env_gym, do_nothing_actions, env, all_actions_dict