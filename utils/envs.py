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
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done = env.step(action)
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards


# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = True):
    states, actions, rewards = [], [], []
    states.append(env.FirstStep())
    done = False
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done = env.step(action)
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
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
