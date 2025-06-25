### Dependencies 

使用env.yaml來創建環境：
```
conda env create --file env.yaml
``` 

### Data setup

第一次執行程式時需要下載環境，可以使用：
```
python create_env.py
```

### Agent training

Note that wandb is used for monitoring the progress of the experiment.
If you wish to use wandb make sure to specify the `WANDB_API_KEY` in the `.env` file. Alternatively, comment out `WandbLoggerCallback` in the `train.py` file.

#### Setup

調整gpu：  
由於learner會耗比較多，default先把整張gpu都給learner用 （例如colab， num_gpus_per_worker 就設 0）  
如果是在kaggle這種有兩張gpu的環境，可以更改train_hierarchical.py
```python
    # 根據是否有 GPU 來設定資源
    # Learner 是訓練的核心，如果用 GPU，就給它一整張卡
    num_gpus_for_learner = 1 if use_gpu else 0
    # 如果 worker 數量多，可以設定 0.1, 0.2 等小數，讓大家共享 GPU
    # num_gpus_per_worker = 0.1 if use_gpu else 0
    num_gpus_per_worker = 0
```

By default, the 5 best checkpoints in terms of mean episode reward will be saved in the `log_files` directory.

#### Native and hybrid agents

執行程式：已經更新，需要使用的東西不再需要輸入True  
===特別注意===  
如果是在colab這種cpu核心數超少的地方，務必將num_workers設為0  
```
!python train_hierarchical.py --algorithm ppo \
 --algorithm_config_path experiments/hierarchical/full_mlp_share_critic.yaml \
 --use_tune \
 --num_iters 10 \
 --num_samples 2 \
 --checkpoint_freq 5 \
 --with_opponent \
 --num_workers 0
```

#### changes I have done:
待更新