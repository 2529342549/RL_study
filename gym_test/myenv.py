import gym

env = gym.make("MyEnv-v0")
env.reset()
env.step(1)

while True:
    env.render()
