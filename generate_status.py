# precompute_norm_stats.py
import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import grid2op
from lightsim2grid import LightSimBackend
import warnings
 
# --- Try to import from user's project structure ---
# This assumes precompute_norm_stats.py is in the same root directory as test.py
try:
    # Import necessary functions and constants from test.py
    from test import ENV_CASE, DATA_SPLIT, make_envs
    # Import ObsConverter to verify dimensions if needed, though calculation uses obs.to_vect()
    from converters import ObsConverter
    print("Successfully imported from local 'test.py' and 'converters.py'")
except ImportError as e:
    print(f"Error importing from local 'test.py' or 'converters.py': {e}")
    print("Please ensure 'precompute_norm_stats.py' is in the root directory of the project,")
    print("or that the project directory is in your PYTHONPATH.")
    # Define necessary constants/functions here only as a fallback if really needed
    # ENV_CASE = {"5": "rte_case5_example", ...}
    # DATA_SPLIT = {"5": (...), ...}
    # You would need to copy the make_envs function definition here.
    # You would need to copy the ObsConverter class definition here.
    exit("Import failed, cannot proceed.")
 
# --- Suppress Grid2Op warnings during data collection ---
warnings.filterwarnings("ignore", category=UserWarning, module='grid2op')
 
# --- Command Line Interface ---
def cli():
    parser = ArgumentParser(description="Precompute normalization statistics (mean/std) for Grid2Op observations based on training chronics.")
    parser.add_argument(
        "-d", "--dir", type=str, default="",
        help="Base directory where 'data' folder is located (default: current directory)"
    )
    parser.add_argument(
        "-c", "--case", type=str, required=True, choices=["5", "14"],
        help="Grid case identifier (e.g., 5 for rte_case5_example)"
    )
    parser.add_argument(
        "-i", "--input", type=str, nargs="+",
        default=["p_i", "r", "o", "d", "m"], # Should match the features used in training
        help="Input features used in the agent's ObsConverter (used for dimension check)"
    )
    # --- Include ALL arguments expected by test.py's make_envs ---
    # Even if not directly used for stats calculation, make_envs might need them.
    # Add default values matching test.py where possible.
    parser.add_argument("-n", "--name", type=str, default="precompute") # Dummy
    parser.add_argument("-la", "--load_agent", type=str, default="") # Dummy
    parser.add_argument("-s", "--seed", type=int, default=0) # Dummy
    parser.add_argument("-rw", "--reward", type=str, default="margin", choices=["loss", "margin"])
    parser.add_argument("-gpu", "--gpuid", type=int, default=0) # Dummy
    parser.add_argument("-ml", "--memlen", type=int, default=50000) # Dummy
    parser.add_argument("-ns", "--nb_steps", type=int, default=100) # Dummy
    parser.add_argument("-ev", "--eval_steps", type=int, default=50) # Dummy
    parser.add_argument("-m", "--mask", type=int, default=3)
    parser.add_argument("-mr", "--max_reward", type=int, default=10)
    parser.add_argument("-fc", "--forecast", type=int, default=0, help="Forecast steps (must match training)")
    parser.add_argument("-dg", "--danger", type=float, default=0.9, help="Danger threshold (must match training)")
    parser.add_argument("-ma", "--middle_agent", type=str, default="capa", choices=["fixed", "random", "capa"]) # Dummy
    parser.add_argument("-a", "--agent", type=str, default="ppo") # Dummy
    parser.add_argument("-nn", "--network", type=str, default="lin") # Dummy
    parser.add_argument("-hn", "--head_number", type=int, default=8) # Dummy
    parser.add_argument("-sd", "--state_dim", type=int, default=128) # Dummy
    parser.add_argument("-nh", "--n_history", type=int, default=6) # Dummy
    parser.add_argument("-do", "--dropout", type=float, default=0.0) # Dummy
    parser.add_argument("-nl", "--n_layers", type=int, default=3) # Dummy
    parser.add_argument("-lr", "--lr", type=float, default=5e-3) # Dummy
    parser.add_argument("-g", "--gamma", type=float, default=0.995) # Dummy
    parser.add_argument("-bs", "--batch_size", type=int, default=8) # Dummy
    parser.add_argument("-u", "--update_start", type=int, default=2) # Dummy
    parser.add_argument("-r", "--rule", type=str, default="c", choices=["c", "d", "o", "f"]) # Dummy
    parser.add_argument("-thr", "--threshold", type=float, default=0.1) # Dummy
    parser.add_argument("-tu", "--target_update", type=int, default=1) # Dummy
    parser.add_argument("--tau", type=float, default=1e-3) # Dummy
    parser.add_argument("-te", "--target_entropy_scale", type=float, default=0.98) # Dummy
    parser.add_argument("-ep", "--epsilon", type=float, default=0.2) # Dummy
    parser.add_argument("-en", "--entropy", type=float, default=0.01) # Dummy
    parser.add_argument("-l", "--lambda", dest='gae_lambda', type=float, default=0.95) # Dummy, use dest
 
    args = parser.parse_args()
    return args
 
# --- Main Execution ---
if __name__ == "__main__":
    args = cli()
    print("Starting normalization statistics precomputation...")
    print(f"Configuration: Case={args.case}, BaseDir='{args.dir if args.dir else '.'}', Features='{args.input}'")
    # The script will use obs.to_vect() for stats, features list is for verification.
 
    # --- 1. Setup Environment ---
    try:
        # Use the exact same function as in training script to ensure environment consistency
        env, test_env_dummy, env_path_from_make_envs = make_envs(args)
        print(f"Environment '{ENV_CASE[args.case]}' created successfully using make_envs.")
 
        # Determine the correct directory to save the stats files
        my_dir = args.dir if args.dir else "."
        # Construct the expected path based on args.dir and ENV_CASE
        save_dir = os.path.join(my_dir, "data", ENV_CASE[args.case])
 
        # Verify the save directory exists
        if not os.path.isdir(save_dir):
            print(f"Error: Target save directory '{save_dir}' does not exist.")
            print(f"Attempted path based on --dir='{args.dir}' and case='{args.case}'.")
            print("Please create the directory structure (e.g., './data/rte_case5_example/') or check the --dir argument.")
            exit(1)
        print(f"Normalization statistics will be saved in: {save_dir}")
 
        # Close the dummy test environment if it was created
        if test_env_dummy:
             test_env_dummy.close()
             del test_env_dummy
 
    except Exception as e:
        print(f"Error during environment creation via make_envs: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure Grid2Op is installed and the case data is accessible.")
        exit(1)
 
    # --- 2. Select Training Chronics ---
    try:
        train_chronics_ids = DATA_SPLIT[args.case][0]
        if not train_chronics_ids:
            print(f"Error: No training chronic IDs found for case '{args.case}' in DATA_SPLIT.")
            exit(1)
        print(f"Using {len(train_chronics_ids)} training chronics defined in DATA_SPLIT: {train_chronics_ids}")
    except KeyError:
        print(f"Error: Case '{args.case}' not found in DATA_SPLIT definition in test.py.")
        exit(1)
    except IndexError:
         print(f"Error: DATA_SPLIT[{args.case}] does not seem to have the expected structure (list of lists/tuples).")
         exit(1)
 
 
    # --- 3. Collect Observation Vectors ---
    all_obs_vects = []
    print("Collecting observation vectors (obs.to_vect()) from training chronics...")
    total_steps_processed = 0
 
    for chron_id in tqdm(train_chronics_ids, desc="Processing Chronics"):
        try:
            # Set the environment to use the specific chronic
            # Note: The actual chronic loaded might be the *next* one after setting ID,
            # depending on Grid2Op internal handling. We assume set_id targets the desired one for reset.
            env.set_id(chron_id)
            obs = env.reset()
            # Double-check the chronic name after reset if possible/needed
            # current_chronic_name = env.chronics_handler.get_name()
            # print(f"Reset done. Current chronic: {current_chronic_name}") # Debug
 
            if obs is None:
                print(f"Warning: env.reset() returned None for chronic id {chron_id}. Skipping.")
                continue
        except Exception as e:
            print(f"Error resetting environment for chronic id {chron_id}: {e}. Skipping.")
            continue
 
        done = False
        step = 0
        max_steps = env.chronics_handler.max_timestep() # Max steps for THIS chronic
 
        while not done and step < max_steps:
            try:
                # Get the full observation vector as used by state_normalize
                obs_vect = obs.to_vect()
                all_obs_vects.append(torch.from_numpy(obs_vect).float())
                total_steps_processed += 1
            except Exception as e:
                print(f"Error getting obs.to_vect() at step {step} in chronic {chron_id}: {e}")
                # Depending on the error, you might want to stop processing this chronic
                break # Stop this chronic on error
 
            try:
                # Advance the environment state with a do-nothing action
                action = env.action_space() # Creates a do-nothing action
                obs, reward, done, info = env.step(action)
                step += 1
 
                # Check for simulation issues
                if info.get("is_illegal", False) or info.get("is_ambiguous", False) or info["exception"] is not None:
                    # print(f"Info: Simulation ended or issue encountered at step {step} in chronic {chron_id}. Info: {info}")
                    done = True # Stop collecting data for this chronic if issues arise
            except Exception as e:
                print(f"Error stepping environment at step {step} in chronic {chron_id}: {e}")
                done = True # Stop this chronic on error
 
    if not all_obs_vects:
        print("\nError: No observation vectors were collected. Cannot compute statistics.")
        print("Check chronic data and environment setup.")
        env.close()
        exit(1)
 
    # --- 4. Compute Mean and Std ---
    print(f"\nCollected {total_steps_processed} observations across all chronics.")
    print("Calculating mean and standard deviation...")
    stacked_obs = torch.stack(all_obs_vects, dim=0)
    print(f"Stacked observations tensor shape: {stacked_obs.shape}") # Should be [total_steps, obs_dim]
 
    # Get the actual dimension from the collected data
    actual_dim = stacked_obs.shape[1]
    print(f"Determined observation dimension from data (obs.to_vect()): {actual_dim}")
 
    # Optional: Compare with environment's stated dimension
    try:
        env_obs_dim = env.observation_space.shape[0]
        print(f"Dimension according to env.observation_space: {env_obs_dim}")
        if actual_dim != env_obs_dim:
            print(f"Warning: Collected dimension ({actual_dim}) differs from env.observation_space ({env_obs_dim}). Using {actual_dim}.")
    except Exception as e:
        print(f"Warning: Could not get dimension from env.observation_space: {e}")
 
    # Calculate statistics
    mean_val = torch.mean(stacked_obs, dim=0)
    std_val = torch.std(stacked_obs, dim=0)
 
    # Handle potential zero standard deviation (replace with 1.0 like in the agent)
    zero_std_mask = std_val < 1e-6
    if torch.any(zero_std_mask):
        print(f"Warning: Found {torch.sum(zero_std_mask)} dimensions with std < 1e-6. Replacing std with 1.0 for these dimensions.")
        std_val = std_val.masked_fill(zero_std_mask, 1.0)
 
    print(f"Calculated mean tensor shape: {mean_val.shape}")
    print(f"Calculated std tensor shape: {std_val.shape}")
 
    # --- 5. Save Results ---
    mean_path = os.path.join(save_dir, "mean.pt")
    std_path = os.path.join(save_dir, "std.pt")
 
    try:
        # Save the tensors
        torch.save(mean_val, mean_path)
        torch.save(std_val, std_path)
        print(f"\nSuccessfully saved mean tensor to: {mean_path}")
        print(f"Successfully saved std tensor to: {std_path}")
    except Exception as e:
        print(f"\nError saving .pt files to {save_dir}: {e}")
        env.close()
        exit(1)
 
    # --- Cleanup ---
    env.close()
    print("\nPrecomputation complete.")