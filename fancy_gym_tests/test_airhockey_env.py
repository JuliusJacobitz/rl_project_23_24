
import gymnasium as gym
import fancy_gym
import os

def general_test():
    env = gym.make('fancy/AirHockey-7dof-hit')
    observation = env.reset(seed=1)
    env.render()

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

def training_test(timesteps=1e6):
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


    # use 90% cpus for training
    cpu_count = 1 if os.cpu_count() <= 1 else int(os.cpu_count() * 0.90)

    train_env = VecMonitor(SubprocVecEnv([make_env(env_id='fancy/AirHockey-7dof-hit', rank=i) for i in range(cpu_count)]))
    train_env = VecNormalize(train_env)
    ppo = PPO("MlpPolicy", train_env, verbose=1)
    ppo.learn(total_timesteps=timesteps, progress_bar=True)
    ppo.save("test_ppo_airhockey")



def loading_test():
    from stable_baselines3 import PPO
    from stable_baselines3.common import vec_env
    from stable_baselines3.common.env_util import make_vec_env

    #load trained model from training_test
    model = PPO.load("test_ppo_airhockey")

    env = gym.make('fancy/AirHockey-7dof-hit')

    observation = env.reset(seed=1)
    #convert to stable baselines observation format
    observation = observation[0]
    
    env.render()

    for i in range(1000):
        action = model.predict(observation, deterministic=True)[0].reshape((2, -1))
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()



if __name__ == "__main__":
    # general_test()
    # training_test()
    loading_test()





