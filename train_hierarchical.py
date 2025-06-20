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
from ray.rllib.policy.policy import PolicySpec
# [新增] 導入 RLModule 相關的類別
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.logger import pretty_print
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import RunConfig, CheckpointConfig

from dotenv import load_dotenv

# [修正] 導入的是我們修正後的 RLModule 模型
from models.mlp import ChooseSubstationModel, ChooseActionModel
from grid2op_env.grid_to_gym import HierarchicalGridGym
from experiments.stopper import MaxNotImprovedStopper
from experiments.callback import LogDistributionsCallback

# 初始化日誌和環境變數
load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
logging.basicConfig(
    format='[INFO]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

LOCAL_DIR = "/Users/chunyu/Desktop/HRL_python311/log_files"

def main():
    # 設置隨機種子以保證可複現性
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)

    # 註冊自訂模型和環境
    # 雖然原始碼中有註冊其他模型，但根據 train_hierarchical.py 的邏輯，只用到以下兩個
    ray.rllib.models.ModelCatalog.register_custom_model("choose_substation_model", ChooseSubstationModel)
    ray.rllib.models.ModelCatalog.register_custom_model("choose_action_model", ChooseActionModel)
    register_env("HierarchicalGridGym", lambda config: HierarchicalGridGym(config))

    # 解析命令列參數
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
    
    args = parser.parse_args()

    logging.info("Training the agent with the following parameters:")
    for arg, value in vars(args).items():
        logging.info(f"{arg.upper()}: {value}")

    # 載入基礎 YAML 設定
    with open(args.algorithm_config_path) as f:
        yaml_config = yaml.safe_load(f)

    # --- 開始全面修正 ---

    # 使用現代化的 AlgorithmConfig API
    # 1. 創建一個基礎的 PPOConfig 物件
    config = PPOConfig()
    tune_params = yaml_config['tune_config']

    # 2. 設置環境
    env_config_train = yaml_config['env_config_train']
    env_config_train['with_opponent'] = args.with_opponent
    config.environment(env="HierarchicalGridGym", env_config=env_config_train)

    # 臨時創建環境以獲取觀察和動作空間
    # temp_env = HierarchicalGridGym(env_config_train)
    
    # 3. 設置多智慧體策略
    config.multi_agent(
        policies=["choose_substation_agent", "choose_action_agent"],
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: 
                           "choose_action_agent" if agent_id.startswith("choose_action_agent") else "choose_substation_agent")
    )

     # 4. [核心修改] 配置 RLModule
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "choose_substation_agent": RLModuleSpec(
                    module_class=ChooseSubstationModel # 直接傳遞類別
                ),
                "choose_action_agent": RLModuleSpec(
                    module_class=ChooseActionModel # 直接傳遞類別
                ),
            }
        )
    )

    # 5. [核心修改] 啟用 RLModule API
    # config.training(_enable_rl_module_api=True)

    # 6. 設置框架、資源和 Rollout 配置
    config.framework("torch")
    config.env_runners(
        num_env_runners=tune_params['num_workers'],
        rollout_fragment_length=tune_params['rollout_fragment_length']
    )
    
    config.fault_tolerance(restart_failed_env_runners=True)

    # 7. 設置訓練超參數
    learner_hyperparameters = {
    "lambda_": tune_params['lambda_'],
    "kl_coeff": tune_params['kl_coeff'],
    "entropy_coeff": tune_params['entropy_coeff'],
    "clip_param": 0.2,
    "vf_loss_coeff": tune_params['vf_loss_coeff'],
    "vf_clip_param": float(tune_params['vf_clip_param']),
    }

    # [修正] 使用新的參數名稱和結構來呼叫 config.training()
    config.training(
        train_batch_size=tune_params['train_batch_size'],
        num_sgd_iter=tune_params['num_epochs'],
        
        # [修正] 'sgd_minibatch_size' 已更名為 'minibatch_size'
        minibatch_size=tune_params['sgd_minibatch_size'], 
        
        # --- 演算法專屬參數 (透過 learner_config_dict 傳入) ---
        learner_config_dict=learner_hyperparameters
    )
    
    # 8. 設置評估
    env_config_val = yaml_config['env_config_val']
    env_config_val['with_opponent'] = args.with_opponent
    config.evaluation(
        evaluation_interval=args.checkpoint_freq,
        evaluation_num_workers=1,
        evaluation_config={"env_config": env_config_val}
    )
    
    config.callbacks(LogDistributionsCallback)
    
     # 處理超參數搜索
    param_space = config.to_dict()
    param_space['seed'] = tune.randint(0, 1000)

    if args.use_tune:
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter=args.num_iters),
            MaxNotImprovedStopper(
                metric="evaluation/episode_reward_mean",
                grace_period=args.grace_period,
                num_iters_no_improvement=args.num_iters_no_improvement
            )
        )

        tune_config = tune.TuneConfig(
            metric="evaluation/episode_reward_mean",
            mode="max",
            num_samples=args.num_samples,
        )
        
        run_config = RunConfig(
            name=args.group or "hierarchical_training_run",
            storage_path=LOCAL_DIR,
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="evaluation/episode_reward_mean",
                checkpoint_score_order="max",
                checkpoint_frequency=args.checkpoint_freq,
            ),
            callbacks=[WandbLoggerCallback(api_key=WANDB_API_KEY)] if WANDB_API_KEY else []
        )

        tuner = tune.Tuner(
            "PPO", # 使用字串 "PPO"
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config
        )
        
        results = tuner.fit()
        logging.info(f"Best checkpoint found was: {results.get_best_result().checkpoint}")

    else:
        algo = config.build()
        for i in range(args.num_iters):
            result = algo.train()
            print(pretty_print(result))
            if (i + 1) % args.checkpoint_freq == 0:
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")
        algo.stop()

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    # [移除] 不再需要註冊自訂模型
    # ray.rllib.models.ModelCatalog.register_custom_model(...)
    register_env("HierarchicalGridGym", lambda config: HierarchicalGridGym(config))
    main()
    ray.shutdown()