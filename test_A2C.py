from stable_baselines3 import A2C 
from quartoenv import RandomOpponentEnv
from policies.A2CPolicy import QuartoA2C
from policies import RandomPolicy
env = RandomOpponentEnv()

# model = A2C("MlpPolicy", env=env, verbose=1)
# model.learn(total_timesteps=1e6, progress_bar=True)
# model.save("regularA2C_1e6.mdl")

model = A2C.load("regularA2C_1e6.mdl")

testingenv = QuartoA2C(env=env, model=model)
controltest = RandomPolicy(env=env)

episodes = 1000

testingenv.test_policy(n_episodes=episodes)
controltest.train(n_episodes=episodes)