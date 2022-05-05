
import gym
import time
import random
import numpy as np

# ---------------Setup environment ---------------------------------------------#
env = gym.make("FrozenLake-v1")

action_space_size=env.action_space.n
# Action can be Up, Down, Left, Right => 4
# Lake is a table 4x4 ==> 16 possible states
state_space_size=env.observation_space.n
print(f"states no: {state_space_size}, actions no:{action_space_size}")
q_table=np.zeros((state_space_size,action_space_size))
# print(q_table)

# %%%%%%%%%%%%%%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_episodes = 200000
max_steps_per_episodes = 100

learning_rate=0.08
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate=0.01

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rewards_all_episodes =[]
for episode in range(num_episodes):

    state = env.reset()
    done = False
    reward_current_episode = 0

    # Exploration vs Exploitation trade-off
    while not done:
        exploration_rate_threshold =random.uniform(0,1)
        if exploration_rate_threshold>exploration_rate:
            action=np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        # Transition
        new_state, reward, done, info = env.step(action)

        #Update q-table
        q_table[state,action] = (1-learning_rate)*q_table[state,action] + learning_rate * (reward+discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        reward_current_episode += reward

        if done == True:
            break

    #Update Exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    # env.render()
    # time.sleep(0.01)

    rewards_all_episodes.append(reward_current_episode)

print("==========")
print(rewards_all_episodes)
rewards_per_thousand_episodes=np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("-------------------------Average reward per 1k episodes ------------------------------")
for r in rewards_per_thousand_episodes:
    print(f"{count}: {sum(r/1000)}")
    count+=1000

print("\n \n --------------- Q-table --------------")
print(q_table)