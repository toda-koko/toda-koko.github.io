from stable_baselines3 import DQN, PPO
import gymnasium as gym
import three_echelonSupplyChain_register
from stable_baselines3.common.env_checker import check_env


env = gym.make("ThreeEchelonSupplyChain-v0")

check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# 保存
model.save("PPO_3SC")

# 終了処理
env.close()