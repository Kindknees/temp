import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import io
import logging
import torch
# FIX: Import RLlibCallback from its new location and remove DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env import BaseEnv
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, EnvID, PolicyID
from ray.tune.logger import TBXLogger
from ray.util.debug import log_once
from ray.tune.result import (TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL)
from ray._private.dict import flatten_dict
from ray.rllib.models import ModelCatalog

from gymnasium.spaces import Discrete
from tensorboardX import SummaryWriter
from typing import TYPE_CHECKING, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64] # from rllib

# Plotting settings
matplotlib.rcParams["figure.dpi"] = 200
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.figure(figsize=(14,10), tight_layout=True)

# FIX: Subclass from RLlibCallback instead of DefaultCallbacks
class LogDistributionsCallback(RLlibCallback):
    """
    Custom callback to log extra information about
    the distribution of observations and actions.
    Updated for Ray RLlib 2.x API.
    """
    
    def on_episode_end(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode, Exception],
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """Called when an episode ends."""
        if isinstance(episode, Exception):
            return
            
        try:
            if isinstance(episode, MultiAgentEpisode):
                steps_in_episode = None
                agent_id_with_info = None
                agent_ids = episode.get_agents()

                # Find an agent that has the 'steps_in_episode' information
                if "choose_substation_agent" in agent_ids:
                    agent_infos = episode.get_infos("choose_substation_agent")
                    if agent_infos and len(agent_infos) > 0 and isinstance(agent_infos[-1], dict):
                        steps_in_episode = agent_infos[-1].get("steps_in_episode") or agent_infos[-1].get("steps")
                        if steps_in_episode is not None:
                            agent_id_with_info = "choose_substation_agent"
                
                if steps_in_episode is None and agent_ids:
                    for agent_id in agent_ids:
                        agent_infos = episode.get_infos(agent_id)
                        if agent_infos and len(agent_infos) > 0 and isinstance(agent_infos[-1], dict):
                            steps_in_episode = agent_infos[-1].get("steps_in_episode") or agent_infos[-1].get("steps")
                            if steps_in_episode is not None:
                                agent_id_with_info = agent_id
                                break
                
                # FIX: Attach metric to the specific SingleAgentEpisode's user_data
                target_agent_id = agent_id_with_info or next(iter(agent_ids), None)
                if target_agent_id:
                    value = steps_in_episode if steps_in_episode is not None else len(episode)
                    episode.agent_episodes[target_agent_id].user_data["num_env_steps"] = value

            elif isinstance(episode, SingleAgentEpisode):
                steps_in_episode = None
                infos = episode.get_infos()
                if infos and len(infos) > 0 and isinstance(infos[-1], dict):
                    steps_in_episode = infos[-1].get("steps_in_episode") or infos[-1].get("steps")
                
                value = steps_in_episode if steps_in_episode is not None else len(episode)
                episode.user_data["num_env_steps"] = value
                
        except Exception as e:
            logger.warning(f"Error processing episode in callback: {e}")
            
    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs
    ) -> None:
        """Called after a training iteration."""
        # Process learner results for each policy
        if "info" in result and "learner" in result["info"]:
            learner_info = result["info"]["learner"]
            
            # Handle both single-policy and multi-policy cases
            if isinstance(learner_info, dict):
                for policy_id, policy_info in learner_info.items():
                    if isinstance(policy_info, dict) and "learner_stats" in policy_info:
                        self._process_policy_stats(policy_id, policy_info["learner_stats"], result)
                    
    def _process_policy_stats(self, policy_id: str, learner_stats: dict, result: dict) -> None:
        """Process statistics for a specific policy."""
        # This is where we would process action distributions if they were collected
        # during training. For now, we'll leave this as a placeholder.
        pass
        
    def on_learn_on_batch(
        self,
        *,
        policy: Policy,
        train_batch: SampleBatch,
        result: dict,
        **kwargs
    ) -> None:
        """
        Log the action distribution and extra information about the observation.
        Note that everything in result[something] is logged by Ray.
        """
        
        # Convert tensors to numpy arrays if needed
        if torch.is_tensor(train_batch["obs"]):
            train_batch_obs = train_batch["obs"].numpy()
        else:
            train_batch_obs = train_batch["obs"]
            
        if torch.is_tensor(train_batch["new_obs"]):
            train_batch_new_obs = train_batch["new_obs"].numpy()
        else:
            train_batch_new_obs = train_batch["new_obs"]
            
        if torch.is_tensor(train_batch["actions"]):
            train_batch_actions = train_batch["actions"].numpy()
        else:
            train_batch_actions = train_batch["actions"]

        # Handle different observation formats
        if isinstance(train_batch_obs, dict):
            # For hierarchical environments, extract the regular observations
            if "regular_obs" in train_batch_obs:
                # This is a hierarchical observation
                # We need to handle this differently
                return self._process_hierarchical_batch(
                    train_batch_obs, train_batch_new_obs, train_batch_actions, result, policy
                )
        
        # For non-hierarchical observations, process normally
        self._process_standard_batch(
            train_batch_obs, train_batch_new_obs, train_batch_actions, result, policy
        )
        
    def _process_hierarchical_batch(
        self,
        train_batch_obs: Union[dict, np.ndarray],
        train_batch_new_obs: Union[dict, np.ndarray],
        train_batch_actions: np.ndarray,
        result: dict,
        policy: Policy
    ) -> None:
        """Process batch data for hierarchical agents."""
        # For hierarchical agents, we might want different metrics
        # For now, just track action distribution
        unique, counts = np.unique(train_batch_actions, return_counts=True)
        action_distr_dic = {}
        
        for action in range(policy.action_space.n):
            action_distr_dic[str(action)] = 0
            
        total_actions = len(train_batch_actions)
        for action, count in zip(unique, counts):
            action_distr_dic[str(action)] = count / total_actions
            
        result["action_distr"] = action_distr_dic
        result["num_unique_actions"] = len(unique)
        
    def _process_standard_batch(
        self,
        train_batch_obs: np.ndarray,
        train_batch_new_obs: np.ndarray,
        train_batch_actions: np.ndarray,
        result: dict,
        policy: Policy
    ) -> None:
        """Process batch data for standard agents."""
        # Assuming the last 56 elements represent topology vector
        topo_vector_size = 56
        
        # Check if observations have enough dimensions for topology vector
        if train_batch_obs.shape[-1] >= topo_vector_size:
            # Extract topology vectors
            old_topo = train_batch_obs[..., -topo_vector_size:]
            new_topo = train_batch_new_obs[..., -topo_vector_size:]
            
            # Check which actions changed the topology
            changed_topo_vec = np.any(new_topo != old_topo, axis=-1)
            not_changed_topo_vec = 1 - changed_topo_vec
            
            # Check for non-terminal states (where topology is not all -1)
            non_terminal_actions = np.any(new_topo != -np.ones_like(new_topo), axis=-1)
            
            num_non_zero_actions = np.sum(changed_topo_vec)
            
            # Log the proportion of actions that change the topology
            if train_batch_actions.shape[0] > 0:
                result["prop_topo_action_change"] = num_non_zero_actions / train_batch_actions.shape[0]
            else:
                result["prop_topo_action_change"] = 0.0
                
            # Log proportion of explicit do-nothing actions
            if np.sum(not_changed_topo_vec * non_terminal_actions) > 0:
                result["prop_explicit_do_nothing"] = (
                    np.sum((train_batch_actions == 0) & (non_terminal_actions == 1)) / 
                    np.sum(not_changed_topo_vec * non_terminal_actions)
                )
            else:
                result["prop_explicit_do_nothing"] = 0.0
        else:
            # Fallback metrics when topology vector is not available
            result["prop_topo_action_change"] = 0.0
            result["prop_explicit_do_nothing"] = 0.0
            changed_topo_vec = np.ones_like(train_batch_actions, dtype=bool)
            num_non_zero_actions = len(train_batch_actions)
        
        # Count action distribution
        if num_non_zero_actions > 0:
            unique, counts = np.unique(train_batch_actions[changed_topo_vec], return_counts=True)
        else:
            unique, counts = np.array([]), np.array([])
            
        action_distr_dic = {}
        for action in range(policy.action_space.n):
            action_distr_dic[str(action)] = 0
            
        for action, count in zip(unique, counts):
            if num_non_zero_actions > 0:
                action_distr_dic[str(action)] = count / num_non_zero_actions
            
        result["action_distr"] = action_distr_dic
        result["num_non_zero_actions_tried"] = sum([1 for val in action_distr_dic.values() if val > 0])


class CustomTBXLogger(TBXLogger):
    """
    Custom TBX logger that logs the action distribution and extra information about the actions
    taken by the agent. Updated for Ray RLlib 2.x.
    """

    def on_result(self, result: Dict) -> None:
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to log these
        
        flat_result = flatten_dict(tmp, delimiter="/")
        
        # Try to find action distribution in the results
        action_distr_dic = None
        
        # Look for action distribution in different possible locations
        if "info" in result and "learner" in result["info"]:
            learner_info = result["info"]["learner"]
            
            # Check for single policy case
            if "default_policy" in learner_info:
                policy_info = learner_info["default_policy"]
                if "custom_metrics" in policy_info and "action_distr" in policy_info["custom_metrics"]:
                    action_distr_dic = policy_info["custom_metrics"]["action_distr"]
                elif "action_distr" in policy_info:
                    action_distr_dic = policy_info["action_distr"]
            
            # Check for multi-policy case
            else:
                for policy_id, policy_info in learner_info.items():
                    if isinstance(policy_info, dict):
                        if "custom_metrics" in policy_info and "action_distr" in policy_info["custom_metrics"]:
                            action_distr_dic = policy_info["custom_metrics"]["action_distr"]
                            break
                        elif "action_distr" in policy_info:
                            action_distr_dic = policy_info["action_distr"]
                            break
        
        # Log action distribution if found
        if action_distr_dic:
            try:
                bar_arr = plot_to_array(bar_graph_from_dict(action_distr_dic))
                self._custom_file_writer = SummaryWriter(self.logdir, flush_secs=30)
                self._custom_file_writer.add_image("Action_distribution", bar_arr, step, dataformats="HWC")
                self._custom_file_writer.close()
            except Exception as e:
                if log_once("action_distr_plot_error"):
                    logger.warning(f"Failed to plot action distribution: {e}")
        
        # Log scalar values
        path = ["ray", "tune"]
        valid_result = {}
        
        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if (isinstance(value, tuple(VALID_SUMMARY_TYPES))
                    and not np.isnan(value)):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(
                    full_attr, value, global_step=step)
            elif ((isinstance(value, list) and len(value) > 0)
                  or (isinstance(value, np.ndarray) and value.size > 0)):
                valid_result[full_attr] = value

                # Must be video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._file_writer.add_video(
                        full_attr, value, global_step=step, fps=20)
                    continue

                try:
                    self._file_writer.add_histogram(
                        full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        logger.warning(
                            "You are trying to log an invalid value ({}={}) "
                            "via {}!".format(full_attr, value,
                                             type(self).__name__))
        
        self.last_result = valid_result
        self._file_writer.flush()


# Utility plotting functions
def bar_graph_from_dict(dic):
    """
    Given a dictionary, return matplotlib bar graph figure.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    # Sort keys for consistent ordering
    sorted_items = sorted(dic.items(), key=lambda x: int(x[0]))
    keys, values = zip(*sorted_items) if sorted_items else ([], [])
    
    ax.bar(keys, values)
    ax.set_xticklabels(keys, rotation=90, fontsize=4)
    ax.set_xlabel('Action')
    ax.set_ylabel('Proportion of all non-zero actions')
    fig.tight_layout()
    
    return fig


def plot_to_array(fig):
    """
    Transform the matplotlib figure to a numpy array.
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im[:, :, :3]  # skip the alpha channel in rgba
    plt.close(fig)  # Close the figure to free memory
    return im