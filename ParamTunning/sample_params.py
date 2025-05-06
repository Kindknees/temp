from typing import Any, Dict
import optuna


def sample_params(trial, agent_name, middle_agent):
    """
    Sample hyperparameters for the specified agent and middle agent.
    
    Args:
        trial (optuna.Trial or None): Optuna trial for hyperparameter sampling, or None for defaults.
        agent_name (str): Name of the agent (e.g., 'ppo', 'ippo').
        middle_agent (str): Name of the middle agent (e.g., 'ppo', 'fixed').
    
    Returns:
        dict: Sampled or default hyperparameters.
    """
    sampled_hyperparams = {}
    
    # Agent-specific parameters
    if agent_name == "ppo":
        sampled_hyperparams.update(sample_ppo_params(trial))
    elif agent_name == "ippo":
        sampled_hyperparams.update(sample_ppo_params(trial))  # IMARL with BasePPO
    # Add other agent types as needed (e.g., sac, sacd)
    
    # Middle agent parameters
    if middle_agent == "ppo":
        sampled_hyperparams.update(sample_middle_ppo_params(trial))
    
    return sampled_hyperparams


def sample_ppo_params(trial):
    """
    Sample PPO hyperparameters or return defaults if trial is None.
    
    Args:
        trial (optuna.Trial or None): Optuna trial or None.
    
    Returns:
        dict: PPO hyperparameters.
    """
    if trial is None:
        return {
            "lr": 5e-3,  # Matches args.lr default in test.py
            "batch_size": 8,  # Matches args.batch_size default
            "gamma": 0.995,  # Matches args.gamma default
            "epsilon": 0.2,
            "lambda": 0.95,
            "hidden_dim": 64,
        }
    
    # Optuna-based sampling
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.3)
    lambda_ = trial.suggest_float("lambda", 0.8, 1.0)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    
    return {
        "lr": lr,
        "batch_size": batch_size,
        "gamma": gamma,
        "epsilon": epsilon,
        "lambda": lambda_,
        "hidden_dim": hidden_dim,
    }

# def sample_ppo_middle_params(trial: optuna.Trial) -> Dict[str, Any]:
#     batch_size = trial.suggest_categorical("middle_batch_size", [4, 8, 16, 32, 64])
#     update_start = trial.suggest_int("middle_update_start", low=1, high=8, step=1)
#     gamma = 1.0 - trial.suggest_float("middle_gamma", 0.001, 0.1, log=True)
#     lr = trial.suggest_float("middle_lr", 1e-5, 0.01, log=True)
#     entropy = trial.suggest_float("middle_entropy", 1e-9, 0.1, log=True)
#     epsilon = trial.suggest_float("middle_epsilon", 0.1, 0.3)
#     gae_lambda = trial.suggest_float("middle_gae_lambda", 0.8, 0.99)
#     trial.set_user_attr("middle_gamma_", gamma)
#     return {
#         "middle_update_start": update_start,
#         "middle_batch_size": batch_size,
#         "middle_gamma": gamma,
#         "middle_lr": lr,
#         "middle_entropy": entropy,
#         "middle_epsilon": epsilon,
#         "middle_gae_lambda": gae_lambda,
#     }

def sample_middle_ppo_params(trial):
    """
    Sample PPO middle agent hyperparameters or return defaults if trial is None.
    
    Args:
        trial (optuna.Trial or None): Optuna trial or None.
    
    Returns:
        dict: Middle agent PPO hyperparameters.
    """
    if trial is None:
        return {
            "middle_lr": 3e-4,  # Matches PPOMiddleAgent default
            "middle_batch_size": 64,  # Matches PPOMiddleAgent default
            "middle_gamma": 0.995,  # Matches PPOMiddleAgent default
            "middle_epsilon": 0.2,  # Matches PPOMiddleAgent default
            "middle_lambda": 0.95,  # Matches PPOMiddleAgent default
            "middle_hidden_dim": 64,  # Matches PPOMiddleAgent default
        }
    
    # Optuna-based sampling
    middle_lr = trial.suggest_float("middle_lr", 1e-5, 1e-3, log=True)
    middle_batch_size = trial.suggest_categorical("middle_batch_size", [32, 64, 128, 256])
    middle_gamma = trial.suggest_float("middle_gamma", 0.9, 0.999)
    middle_epsilon = trial.suggest_float("middle_epsilon", 0.1, 0.3)
    middle_lambda = trial.suggest_float("middle_lambda", 0.8, 1.0)
    middle_hidden_dim = trial.suggest_categorical("middle_hidden_dim", [32, 64, 128])
    
    return {
        "middle_lr": middle_lr,
        "middle_batch_size": middle_batch_size,
        "middle_gamma": middle_gamma,
        "middle_epsilon": middle_epsilon,
        "middle_lambda": middle_lambda,
        "middle_hidden_dim": middle_hidden_dim,
    }

def sample_dppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DPPO hyperparams.
        Used as a referenc: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    update_start = trial.suggest_int("update_start", low=1, high=6, step=1)
    # discount factor between 0.9 and 0.999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.01, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    entropy = trial.suggest_float("entropy", 1e-6, 1e-3, log=True)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.15)
    gae_lambda = trial.suggest_float("gae_lambda", 0.825, 0.925)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "entropy": entropy,
        "epsilon": epsilon,
        "gae_lambda": gae_lambda,
    }


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC(D) hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    update_start = trial.suggest_int("update_start", low=2, high=8, step=2)
    # discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    target_update = trial.suggest_int("target_update", low=1, high=2, step=1)
    target_entropy_scale = trial.suggest_float("target_entropy_scale", 0.95, 0.99)
    tau = trial.suggest_float("tau", 1e-3, 1e-2)
    update_freq = trial.suggest_int("update_freq", low=1, high=5, step=1)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "target_update": target_update,
        "target_entropy_scale": target_entropy_scale,
        "tau": tau,
        "update_freq": update_freq
    }


def sample_dsacd_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SACD hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    update_start = trial.suggest_int("update_start", low=1, high=4, step=1)
    # discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    target_update = trial.suggest_int("target_update", low=1, high=2, step=1)
    target_entropy_scale = trial.suggest_float("target_entropy_scale", 0.95, 0.99)
    tau = trial.suggest_float("tau", 1e-3, 1e-2)
    update_freq = trial.suggest_int("update_freq", low=1, high=5, step=1)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "target_update": target_update,
        "target_entropy_scale": target_entropy_scale,
        "tau": tau,
        "update_freq": update_freq
    }