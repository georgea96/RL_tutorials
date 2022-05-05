import gym
import time

'''
observation_space for our environment was Box(2,), and the action_space was Discrete(2,)

data structures are derived from the gym.Space base class.  
'''
'''
Box(n,): corresponds to the n-dimensional continuous space. In our case n=2, thus the observational space of our environment is a 2-D space.
  -  bounded by upper and lower limits which describe the legitimate values our observations can take
  
  - Determine bounds: using 
    print("Upper Bound for Env Observation", env.observation_space.high)
    print("Lower Bound for Env Observation", env.observation_space.low)
'''

'''
Discrete(n) box describes a discrete space with [0.....n-1] possible values. 
    
    - In our case n = 3, meaning our actions can take values of either 0, 1, or 2. 
    
    - Discrete does not have a high and low method, since, by the very definition
'''

env = gym.make("BreakoutNoFrameskip-v4",render_mode='human')

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

# Observation space = (210,160,3) with values from 0 to 255
print("Max: ", env.observation_space.high,"Min: ",env.observation_space.low)

obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # env.render()
    time.sleep(0.01)
env.close()

