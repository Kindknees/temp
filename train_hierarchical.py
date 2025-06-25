import logging
import os
import random
import yaml
import argparse
import numpy as np
import torch
import gymnasium as gym

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.logger import pretty_print
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import RunConfig, CheckpointConfig

from dotenv import load_dotenv

# Import custom RLModule models
from models.mlp import ChooseSubstationModel, ChooseActionModel
# Use the serializable wrapper instead
from grid2op_env.grid2op_wrapper import SerializableHierarchicalGridGym
from experiments.stopper import MaxNotImprovedStopper
from experiments.callback import LogDistributionsCallback

# Initialize logging and environment variables
load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
logging.basicConfig(
    format='[INFO]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

current_directory = os.path.dirname(os.path.abspath(__file__))
new_folder_name = "log_files"
log_files_absolute_path = os.path.join(current_directory, new_folder_name)
LOCAL_DIR = log_files_absolute_path

def main():
    # Set random seeds for reproducibility
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)
    
    # 偵測是否有可用的 GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info("GPU is available, configuring Ray to use GPU resources.")
    else:
        logging.info("GPU is not available, using CPU resources.")

    # 根據是否有 GPU 來設定資源
    # Learner 是訓練的核心，如果用 GPU，就給它一整張卡
    num_gpus_for_learner = 1 if use_gpu else 0
    # 如果 worker 數量多，可以設定 0.1, 0.2 等小數，讓大家共享 GPU
    # num_gpus_per_worker = 0.1 if use_gpu else 0
    num_gpus_per_worker = 0

    # Register environment with the serializable wrapper
    # register_env("HierarchicalGridGym", lambda config: SerializableHierarchicalGridGym(config))

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a hierarchical agent on the Grid2Op environment")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm to use", choices=["ppo"])
    parser.add_argument("--algorithm_config_path", type=str, default="experiments/hierarchical/full_mlp_share_critic.yaml", help="Path to config file for the algorithm")
    parser.add_argument("--use_tune", action=argparse.BooleanOptionalAction, default=True, help="Use Tune to train the agent")
    parser.add_argument("--project_name", type=str, default="hierarchical_grid_control", help="Name of the project to be saved in WandB")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations to train the agent for.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to use for training.")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Number of iterations between checkpoints.")
    parser.add_argument("--group", type=str, default=None, help="Group to use for training.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume training from a checkpoint.")
    parser.add_argument("--grace_period", type=int, default=400, help="Minimum number of iterations before a trial can be early stopped.")
    parser.add_argument("--num_iters_no_improvement", type=int, default=200, help="Number of iterations with no improvement before stopping.")
    parser.add_argument("--with_opponent", action=argparse.BooleanOptionalAction, default=False, help="Whether to use an opponent or not.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of rollout workers to use. Set to 0 for Colab/low-resource environments.")

    args = parser.parse_args()

    logging.info("Training the agent with the following parameters:")
    for arg, value in vars(args).items():
        logging.info(f"{arg.upper()}: {value}")

    # Load YAML configuration
    with open(args.algorithm_config_path) as f:
        yaml_config = yaml.safe_load(f)

    # Create base PPOConfig
    config = PPOConfig()
    tune_params = yaml_config['tune_config']

    # Configure environment
    env_config_train = yaml_config['env_config_train']
    env_config_train['with_opponent'] = args.with_opponent
    config.environment(
        env="HierarchicalGridGym", 
        env_config=env_config_train,
        disable_env_checking=True
    )

    # Configure multi-agent policies
    config.multi_agent(
        policies=["choose_substation_agent", "choose_action_agent"],
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: 
                           "choose_action_agent" if agent_id.startswith("choose_action_agent") else "choose_substation_agent")
    )

    # Configure RLModule
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "choose_substation_agent": RLModuleSpec(
                    module_class=ChooseSubstationModel
                ),
                "choose_action_agent": RLModuleSpec(
                    module_class=ChooseActionModel
                ),
            }
        ),
    )

    # Configure framework and resources
    config.framework("torch")
    config.env_runners(
        num_env_runners=args.num_workers,
        rollout_fragment_length=tune_params['rollout_fragment_length'],
        num_gpus_per_env_runner=num_gpus_per_worker,
        add_default_connectors_to_env_to_module_pipeline=True,
        add_default_connectors_to_module_to_env_pipeline=True,
        batch_mode="complete_episodes"
    )
    
    config.fault_tolerance(restart_failed_env_runners=True)

    config.api_stack(enable_rl_module_and_learner=True,
                     enable_env_runner_and_connector_v2=True)
    config.experimental(_validate_config=False)

    config.resources(
        num_gpus=num_gpus_per_worker
    )

    config.learners(
        num_gpus_per_learner=num_gpus_for_learner
    )

    # Configure training hyperparameters
    # Note: In newer RLlib versions, PPO-specific parameters are set directly
    config.training(
        train_batch_size=tune_params['train_batch_size'],
        lr=tune_params.get('lr', 5e-5),
        num_epochs=tune_params['num_epochs'],
        minibatch_size=tune_params['sgd_minibatch_size'],
        lambda_=tune_params['lambda_'],
        kl_coeff=tune_params['kl_coeff'],
        entropy_coeff=tune_params['entropy_coeff'],
        clip_param=0.2,  # Standard PPO clip parameter
        vf_loss_coeff=tune_params['vf_loss_coeff'],
        vf_clip_param=float(tune_params['vf_clip_param']),
        # from community
        use_gae=True,
        use_critic=True
    )
    
    # Configure evaluation
    env_config_val = yaml_config['env_config_val']
    env_config_val['with_opponent'] = args.with_opponent
    config.evaluation(
        evaluation_interval=args.checkpoint_freq,
        evaluation_num_env_runners=1,
        evaluation_config={"env_config": env_config_val}
    )
    
    config.callbacks(LogDistributionsCallback)
    
    # Handle hyperparameter search
    param_space = config.to_dict()
    param_space['lr'] = tune.loguniform(1e-6, 1e-4)
    param_space['seed'] = yaml_config["tune_config"]["seed"]

    if args.use_tune:
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter=args.num_iters),
            MaxNotImprovedStopper(
                metric="evaluation/env_runners/episode_return_mean",
                grace_period=args.grace_period,
                num_iters_no_improvement=args.num_iters_no_improvement
            )
        )

        tune_config = tune.TuneConfig(
            metric="evaluation/env_runners/episode_return_mean",
            mode="max",
            num_samples=args.num_samples,
        )
        
        run_config = RunConfig(
            name=args.group or "hierarchical_training_run",
            storage_path=LOCAL_DIR,
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="evaluation/env_runners/episode_return_mean",
                checkpoint_score_order="max",
                checkpoint_frequency=args.checkpoint_freq,
            ),
            callbacks=[WandbLoggerCallback(api_key=WANDB_API_KEY, project=args.project_name)] if WANDB_API_KEY else []
        )

        tuner = tune.Tuner(
            "PPO", 
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config
        )
        
        results = tuner.fit()
        
        # Handle the case where no results completed
        try:
            best_result = results.get_best_result()
            logging.info(f"Best checkpoint found was: {best_result.checkpoint}")
        except RuntimeError as e:
            logging.error(f"No results completed: {e}")
            logging.info("Check the logs above for error details.")

    else: # Single training run without Tune
        algo = config.build()
        for i in range(args.num_iters):
            result = algo.train()
            print(pretty_print(result))
            if (i + 1) % args.checkpoint_freq == 0:
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")
        algo.stop()

if __name__ == "__main__":
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    
    # Register the serializable environment
    register_env("HierarchicalGridGym", lambda config: SerializableHierarchicalGridGym(config))
    main()
    ray.shutdown()