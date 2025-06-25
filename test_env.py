#!/usr/bin/env python3
"""
Test script to verify SerializableHierarchicalGridGym works correctly.
"""

import pickle
import yaml
import ray
from ray.tune.registry import register_env

# Add path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid2op_env.grid2op_wrapper import SerializableHierarchicalGridGym

def test_basic_functionality():
    """Test basic environment functionality."""
    print("1. Testing basic environment functionality...")
    
    # Load config
    with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
        yaml_config = yaml.safe_load(f)
    
    env_config = yaml_config['env_config_train']
    env_config['with_opponent'] = False
    
    try:
        # Create environment
        env = SerializableHierarchicalGridGym(env_config)
        print("✓ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful, initial agent: {env.agents}")
        
        # Test a few steps
        for i in range(3):
            if env.agents:
                agent_id = env.agents[0]
                if agent_id == env.high_level_agent_id:
                    action = env.action_space[agent_id].sample()
                else:
                    # For low-level agent, choose a valid action
                    action_mask = obs[agent_id]["action_mask"]
                    valid_actions = [j for j in range(len(action_mask)) if action_mask[j] > 0]
                    action = valid_actions[0] if valid_actions else 0
                
                action_dict = {agent_id: action}
                obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
                print(f"✓ Step {i+1} successful, next agents: {env.agents}")
                
                if terminateds["__all__"]:
                    break
        
        env.close()
        print("✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_serialization():
    """Test that the environment can be pickled."""
    print("\n2. Testing serialization...")
    
    with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
        yaml_config = yaml.safe_load(f)
    
    env_config = yaml_config['env_config_train']
    env_config['with_opponent'] = False
    
    try:
        # Create environment
        env = SerializableHierarchicalGridGym(env_config)
        
        # Try to pickle it
        pickled = pickle.dumps(env)
        print("✓ Environment pickled successfully")
        
        # Try to unpickle it
        env2 = pickle.loads(pickled)
        print("✓ Environment unpickled successfully")
        
        # Test that unpickled env works
        obs, _ = env2.reset()
        print("✓ Unpickled environment can reset")
        
        env.close()
        env2.close()
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_ray():
    """Test environment with Ray."""
    print("\n3. Testing with Ray...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Register environment
        register_env("SerializableHierarchicalGridGym", 
                    lambda config: SerializableHierarchicalGridGym(config))
        
        # Load config
        with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
            yaml_config = yaml.safe_load(f)
        
        env_config = yaml_config['env_config_train']
        env_config['with_opponent'] = False
        
        # Create environment through Ray registration
        from ray.rllib.env.env_context import EnvContext
        env_context = EnvContext(env_config, worker_index=0, num_workers=1)
        
        # Test creating environment in a remote function
        @ray.remote
        def create_and_test_env(config):
            env = SerializableHierarchicalGridGym(config)
            obs, _ = env.reset()
            env.close()
            return True
        
        # Execute remote function
        result = ray.get(create_and_test_env.remote(env_context))
        if result:
            print("✓ Remote environment creation successful")
        
        # Test with multiple workers
        results = ray.get([create_and_test_env.remote(env_context) for _ in range(2)])
        if all(results):
            print("✓ Multiple remote workers successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        ray.shutdown()

def test_with_ppo():
    """Test environment with PPO algorithm."""
    print("\n4. Testing with PPO algorithm...")
    
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from models.mlp import ChooseSubstationModel, ChooseActionModel
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Register environment
        register_env("SerializableHierarchicalGridGym", 
                    lambda config: SerializableHierarchicalGridGym(config))
        
        # Load config
        with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
            yaml_config = yaml.safe_load(f)
        
        # Create PPO config
        config = PPOConfig()
        
        env_config_train = yaml_config['env_config_train']
        env_config_train['with_opponent'] = False
        
        config.environment(
            env="SerializableHierarchicalGridGym", 
            env_config=env_config_train,
            disable_env_checking=True
        )
        
        config.multi_agent(
            policies=["choose_substation_agent", "choose_action_agent"],
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: 
                               "choose_action_agent" if agent_id.startswith("choose_action_agent") 
                               else "choose_substation_agent")
        )
        
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
        
        config.api_stack(enable_rl_module_and_learner=True,
                         enable_env_runner_and_connector_v2=True)
        config.framework("torch")
        config.env_runners(num_env_runners=0)  # Local only for testing
        
        # Try to build the algorithm
        algo = config.build_algo()
        print("✓ PPO algorithm built successfully")
        
        # Try one training iteration
        result = algo.train()
        print("✓ One training iteration completed")
        print(f"  Episode reward mean: {result.get('env_runners', {}).get('episode_reward_mean', 'N/A')}")
        
        algo.stop()
        ray.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ PPO test failed: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()
        return False

if __name__ == "__main__":
    print("Testing SerializableHierarchicalGridGym...\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Serialization", test_serialization),
        ("Ray Integration", test_with_ray),
        ("PPO Algorithm", test_with_ppo),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        results.append((test_name, test_func()))
    
    print(f"\n{'='*50}")
    print("Test Summary:")
    print('='*50)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed! The environment is ready for training.")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before training.")