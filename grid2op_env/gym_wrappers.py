import numpy as np
from gymnasium import ObservationWrapper

def _cast_to_float32(obs):
    """
    遞迴地走訪觀測值(字典、列表或陣列)，並將所有浮點數Numpy陣列轉換為float32。
    """
    if isinstance(obs, dict):
        # 處理字典結構的觀測值
        return {k: _cast_to_float32(v) for k, v in obs.items()}
    elif isinstance(obs, (np.ndarray, np.generic)):
        # 如果是Numpy陣列且其資料型態為浮點數，則轉換為float32
        if np.issubdtype(obs.dtype, np.floating):
            return obs.astype(np.float32)
        else:
            return obs
    elif isinstance(obs, (list, tuple)):
        # 處理列表或元組
        return type(obs)(_cast_to_float32(v) for v in obs)
    else:
        # 其他型態保持不變
        return obs

class TransformObservation(ObservationWrapper):
    r"""Transform the observation via an arbitrary function.
    Example::
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    Args:
        env (Env): environment
        f (callable): a function that transforms the observation
    """

    def __init__(self, env, f):
        super().__init__(env)
        assert callable(f)
        self.f = f
        self.grid2op_env = env.org_env

    def observation(self, observation):
        """
        對觀測值進行轉換，並確保最終輸出的資料型態正確。
        """
        # 1. 確保傳入的原始觀測值型態正確
        casted_observation = _cast_to_float32(observation)

        # 2. 執行使用者定義的轉換函數 f
        if self.grid2op_env is not None:
            transformed_obs = self.f(casted_observation, self.grid2op_env)
        else:
            transformed_obs = self.f(casted_observation)

        # 3. 再次轉換以確保 f 函數沒有意外改變型態
        return _cast_to_float32(transformed_obs)