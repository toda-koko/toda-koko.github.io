from stable_baselines3 import DQN, PPO
import gymnasium as gym
import three_echelonSupplyChain_register
import matplotlib.pyplot as plt

model = PPO.load("PPO_3SC")
env = gym.make("ThreeEchelonSupplyChain-v0")


obs, info = env.reset()
terminated = truncated = False

rewards = []  # 報酬記録用

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)

plt.plot(rewards)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward per Step (1 Episode)")
plt.grid(True)
plt.tight_layout()
plt.show()