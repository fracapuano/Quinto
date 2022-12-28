from stable_baselines3 import A2C 
from quartoenv import RandomOpponentEnv

env = RandomOpponentEnv()
model = A2C("MlpPolicy", env=env, verbose=1)

model.learn(total_timesteps=1e6, progress_bar=True)
model.save("regularA2C_1e6.mdl")