
import gymnasium as gym
import fancy_gym

def general_test():
    env = gym.make('fancy/AirHockey-7dof-hit')
    observation = env.reset(seed=1)
    env.render()

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

def training_test():
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.utils import set_random_seed

    def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):
        def _init():
            env = gym.make(id=env_id, **kwargs)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    train_env = VecMonitor(SubprocVecEnv([make_env(env_id='fancy/AirHockey-7dof-defend-airhockit2023-v0', rank=i) for i in range(12)]))
    train_env = VecNormalize(train_env)
    ppo = PPO("MlpPolicy", train_env, verbose=1)
    ppo.learn(total_timesteps=1e6, progress_bar=True)

if __name__ == "__main__":
    general_test()