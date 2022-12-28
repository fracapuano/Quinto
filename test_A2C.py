from stable_baselines3 import A2C 
from quartoenv import RandomOpponentEnv
from policies.A2CPolicy import QuartoA2C
from policies import RandomPolicy

env = RandomOpponentEnv()

model = A2C.load("trainedmodels/A2Cv0_1e6.mdl")

testingenv = QuartoA2C(env=env, model=model)
controltest = RandomPolicy(env=env)

episodes = 2000
testingenv.test_policy(n_episodes=episodes)
controltest.train(n_episodes=episodes)
