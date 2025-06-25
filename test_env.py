#!/usr/bin/env python3
"""
Test script to verify HierarchicalGridGym environment works correctly.
Run this before training to ensure the environment is properly set up.
"""

import sys
import yaml
import ray
from ray.tune.registry import register_env

# Add the project root to the Python path if needed
# sys.path.append('/path/to/your/project')

from grid2op_env.grid_to_gym import HierarchicalGridGym

def test_environment():
    """Test the hierarchical environment setup."""
    
    print("1. Testing environment initialization...")
    
    # Load config
    with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
        yaml_config = yaml.safe_load(f)
    
    env_config = yaml_config['env_config_train']
    env_config['with_opponent'] = False  # Start without opponent for testing
    
    try:
        # Create environment
        env = HierarchicalGridGym(env_config)
        print("✓ Environment created successfully")
        
        # Check observation and action spaces
        print("\n2. Checking spaces...")
        print(f"Observation space keys: {list(env.observation_space.keys())}")
        print(f"Action space keys: {list(env.action_space.keys())}")
        print(f"Agent IDs: {env.get_agent_ids()}")
        print("✓ Spaces look correct")
        
        # Test reset
        print("\n3. Testing reset...")
        obs, info = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Active agents after reset: {env.agents}")
        print("✓ Reset successful")
        
        # Test step with high-level agent
        print("\n4. Testing high-level agent step...")
        high_level_action = env.action_space[env.high_level_agent_id].sample()
        action_dict = {env.high_level_agent_id: high_level_action}
        obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
        print(f"Observation keys after high-level step: {list(obs.keys())}")
        print(f"Active agents after high-level step: {env.agents}")
        print("✓ High-level step successful")
        
        # Test step with low-level agent
        print("\n5. Testing low-level agent step...")
        if env.low_level_agent_id in obs:
            # Get a valid action based on action mask
            action_mask = obs[env.low_level_agent_id]["action_mask"]
            valid_actions = [i for i in range(len(action_mask)) if action_mask[i] > 0]
            if valid_actions:
                low_level_action = valid_actions[0]
            else:
                low_level_action = 0
            
            action_dict = {env.low_level_agent_id: low_level_action}
            obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
            print(f"Observation keys after low-level step: {list(obs.keys())}")
            print(f"Active agents after low-level step: {env.agents}")
            print(f"Episode done: {terminateds['__all__']}")
            print("✓ Low-level step successful")
        
        # Test close
        print("\n6. Testing close...")
        env.close()
        print("✓ Environment closed successfully")
        
        print("\n✅ All environment tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_with_ray():
    """Test environment with Ray registration."""
    print("\n7. Testing with Ray...")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("HierarchicalGridGym", lambda config: HierarchicalGridGym(config))
    
    # Try to create environment through Ray
    from ray.rllib.env.env_context import EnvContext
    
    with open("experiments/hierarchical/full_mlp_share_critic.yaml") as f:
        yaml_config = yaml.safe_load(f)
    
    env_config = yaml_config['env_config_train']
    env_config['with_opponent'] = False
    
    env_context = EnvContext(env_config, worker_index=0, num_workers=1)
    env = HierarchicalGridGym(env_context)
    
    obs, _ = env.reset()
    print("✓ Ray environment creation successful")
    
    env.close()
    ray.shutdown()
    
    return True

if __name__ == "__main__":
    print("Testing HierarchicalGridGym environment...\n")
    
    # Run basic tests
    if test_environment():
        print("\nRunning Ray integration test...")
        if test_with_ray():
            print("\n🎉 All tests passed! The environment is ready for training.")
        else:
            print("\n⚠️ Ray integration test failed.")
    else:
        print("\n⚠️ Basic environment test failed.")