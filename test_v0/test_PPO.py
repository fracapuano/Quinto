from stable_baselines3 import PPO
from quartoenv import RandomOpponentEnv

env = RandomOpponentEnv()
model = PPO("MlpPolicy", env=env, verbose=1)

model.learn(total_timesteps=1e6, progress_bar=True)
model.save("PPOv1_1e6.mdl")
