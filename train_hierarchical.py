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
from ray.tune.logger import pretty_print
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from dotenv import load_dotenv

# 專案模組導入
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

LOCAL_DIR = "log_files"

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

    # 使用現代化的 AlgorithmConfig API
    # 1. 創建一個基礎的 PPOConfig 物件
    config = PPOConfig()

    # 2. 設置環境
    env_config_train = yaml_config['env_config_train']
    env_config_train['with_opponent'] = args.with_opponent
    config.environment(env="HierarchicalGridGym", env_config=env_config_train)
    
    # 臨時創建環境以獲取觀察和動作空間，用於策略定義
    temp_env = HierarchicalGridGym(env_config_train)
    
    # 3. 設置多智慧體策略
    policies = {
        "choose_substation_agent": PolicySpec(
            observation_space=temp_env.observation_space["choose_substation_agent"],
            action_space=temp_env.action_space["choose_substation_agent"],
            config=yaml_config['tune_config']['multiagent']['policies']['choose_substation_agent']['config']
        ),
        "choose_action_agent": PolicySpec(
            observation_space=temp_env.observation_space["choose_action_agent"],
            action_space=temp_env.action_space["choose_action_agent"],
            config=yaml_config['tune_config']['multiagent']['policies']['choose_action_agent']['config']
        )
    }
    
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: 
                           "choose_action_agent" if agent_id.startswith("choose_action_agent") else "choose_substation_agent")
    )
    
    # 4. 設置框架、資源和 Rollout 配置
    config.framework("torch")
    config.rollouts(num_rollout_workers=yaml_config['tune_config']['num_workers'])
    
    # 5. 設置訓練超參數
    training_params = {
        key: val for key, val in yaml_config['tune_config'].items() 
        if key not in ["env", "env_config", "multiagent", "num_workers", "callbacks", "evaluation_interval", "evaluation_num_episodes", "evaluation_config", "log_level", "framework", "seed"]
    }
    config.training(**training_params)
    
    # 6. 設置評估
    env_config_val = yaml_config['env_config_val']
    env_config_val['with_opponent'] = args.with_opponent
    config.evaluation(
        evaluation_interval=args.checkpoint_freq, # 評估頻率與檢查點頻率一致
        evaluation_num_workers=1,
        evaluation_config={"env_config": env_config_val}
    )
    
    # 7. 設置回呼函式
    config.callbacks(LogDistributionsCallback)
    
    # 8. 設置日誌和種子
    config.reporting(metrics_num_episodes_for_smoothing=5)
    
    # 處理超參數搜索
    param_space = config.to_dict()
    # 在這裡可以像舊方法一樣使用 tune.choice, tune.grid_search 等
    # 例如，從 YAML 中讀取搜索空間定義或直接在此處硬編碼
    param_space['seed'] = tune.choice(list(range(16))) # 來自 full_mlp_share_critic.yaml 的示例

    if args.use_tune:
        # 定義停止器
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter=args.num_iters),
            MaxNotImprovedStopper(
                metric="evaluation/episode_reward_mean",
                grace_period=args.grace_period,
                num_iters_no_improvement=args.num_iters_no_improvement,
                no_stop_if_val=5500 # 來自舊 stopper 的邏輯
            )
        )

        # 定義 TuneConfig
        tune_config = tune.TuneConfig(
            metric="evaluation/episode_reward_mean",
            mode="max",
            num_samples=args.num_samples,
        )

        # 定義 Tuner
        tuner = tune.Tuner(
            args.algorithm.upper(),
            param_space=param_space,
            tune_config=tune_config,
            run_config=ray.air.RunConfig(
                name=args.group or "hierarchical_training_run",
                local_dir=LOCAL_DIR,
                stop=stopper,
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=args.checkpoint_freq,
                    num_to_keep=5,
                    checkpoint_score_attribute="evaluation/episode_reward_mean",
                    checkpoint_score_order="max"
                ),
                callbacks= WANDB_API_KEY if WANDB_API_KEY else None
            )
        )
        
        # 執行調優
        results = tuner.fit()
        logging.info(f"Best checkpoint found was: {results.get_best_result().checkpoint}")

    else:
        # 直接訓練單一實例，不使用 Tune
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
    main()
    ray.shutdown()