import grid2op
import os

env_name = "rte_case14_realistic"  # or any other...
env = grid2op.make(env_name)

# retrieve the names of the chronics:
full_path_data = env.chronics_handler.subpaths
chron_names = [os.path.split(el)[-1] for el in full_path_data]

# splitting into training / test, keeping the "last" 10 chronics to the test set
nm_env_train, m_env_val, nm_env_test = env.train_val_split(test_scen_id=chron_names[-10:],  # last 10 in test set
                                                           add_for_test="test",
                                                           val_scen_id=chron_names[-20:-10],  # last 20 to last 10 in val test
                                                           )

env_train = grid2op.make(env_name+"_train")
env_val = grid2op.make(env_name+"_val")
env_test = grid2op.make(env_name+"_test")