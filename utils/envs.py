import gym
import numpy as np
import random
from copy import deepcopy

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
# def play_episode(env, policy, render = True):
#     states, actions, rewards = [], [], []
#     states.append(env.reset())
#     done = False
#     #if render: env.render()
#     while not done:
#         action = policy(env, states[-1])
#         actions.append(action)
#         obs, reward, done, info = env.step(action)
#         if render: env.render()
#         states.append(obs)
#         rewards.append(reward)
#     return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.FirstStep())
    done = False
    counterToDie =0
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done = env.step(action)
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)

        #add counter to prevent is sping at one position not moving forward
        #prevent it is rorating but not hiting checkpoints
        if reward<0.51:
            counterToDie+=1
            if counterToDie>100:
                done=True
        else:
            counterToDie=0 
    return states, actions, rewards


# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = True):
    states, actions, rewards = [], [], []
    states.append(env.FirstStep())
    done = False
    counterToDie=0
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done = env.step(action)
        if render: env.render()
        states.append(obs)
        rewards.append(reward)

        #add counter to prevent is sping at one position not moving forward
        if reward<0.51:
            counterToDie+=1
            if counterToDie>100:
                done=True
        else:
            counterToDie=0 
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
# def play_episode_rb_car(env, policy, buf):
#     states, actions, rewards = [], [], []
#     states.append(env.reset())
#     done = False
#     while not done:
#         action = policy(env, states[-1])
#         actions.append(action)
#         obs, reward, done = env.step(action)
#         buf.add(states[-1], action, reward, obs, done)
#         states.append(obs)
#         rewards.append(reward)
#     return states, actions, rewards
