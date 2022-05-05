'''
Wrapper class in OpenAI Gym provides you with the functionality to modify various parts of an environment to suit your needs.

normalize your pixel input, or maybe you want to clip your rewards
'''

import gym
import time

env = gym.make("BreakoutNoFrameskip-v4",render_mode='human')

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)


obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # env.render()
    time.sleep(0.01)
env.close()

'''
It's a common practice in Deep RL that we construct our observation 
by concatenating the past k frames together. 

We have to modify the Breakout Environment such that both our reset 
and step functions return concatenated observations.

1) Define a class of type gym.Wrapper to override the 'reset' and 'return' functions 
   of the environment so they return k-concat frames
   
"Wrapper class", as the name suggests, is a wrapper on top of an Env class that modifies 
some of its attributes and functions.
'''

# Used for using double ended queues
# Double ended queues prefered over queues due to O(1) instead of O(n)
from collections import deque
from gym import spaces
import numpy as np

class ConcatObs(gym.Wrapper):
    def __init__(self,env,k):
        gym.Wrapper.__init__(self,env)
        self.k=k
        self.frames=deque([],maxlen=k)
        shp = env.observation_space.shape
        self.observation_space=spaces.Box(low=0,high=255,shape=((k,)+shp),dtype=env.observation_space.dtype)

    def reset(self):
        ob= self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self,action):
        ob,reward,done,info=self.env.step(action)
        self.frames.append(ob)

        return self._get_ob(),reward,done,info

    def _get_ob(self):
        return np.array(self.frames)