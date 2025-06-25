import grid2op
import numpy as np

env_name = "rte_case14_realistic"
# 載入基礎環境
env = grid2op.make(env_name, test=True)

# 假設 .npy 檔案路徑正確
val_chron = np.load("grid2op_env/train_val_test_split/val_chronics.npy")
test_chron = np.load("grid2op_env/train_val_test_split/test_chronics.npy")

# 步驟 1: 執行分割。此函式會在硬碟上建立檔案，並「回傳新環境的名稱」
print("正在分割環境...")
train_env_name, val_env_name, test_env_name = env.train_val_split(
    test_scen_id=test_chron,
    add_for_test="test",
    val_scen_id=val_chron
)
print(f"分割完成，取得環境名稱: '{train_env_name}', '{val_env_name}', '{test_env_name}'")


# 步驟 2: 使用上一步回傳的「準確名稱」來呼叫 grid2op.make()，以載入完整的環境物件
print("\n正在載入分割後的環境...")
env_train = grid2op.make(train_env_name, test=True)
env_val = grid2op.make(val_env_name, test=True)
env_test = grid2op.make(test_env_name, test=True)

# 現在，env_train, env_val, env_test 都是功能齊全的環境物件了
print("\n訓練環境已成功載入:", env_train)
print("驗證環境已成功載入:", env_val)
print("測試環境已成功載入:", env_test)


# 步驟 3: 現在 env_train 是一個真正的環境物件，可以進行操作
try:
    obs_train = env_train.reset()
    print("\n成功重置訓練環境！")
    # 確認 obs_train 不是 None
    if hasattr(obs_train, 'observation_space'):
        print("觀測空間維度:", obs_train.observation_space.shape)
    else:
        print("觀測物件已取得，但無 observation_space 屬性。")

except Exception as e:
    print(f"\n操作環境時發生錯誤: {e}")