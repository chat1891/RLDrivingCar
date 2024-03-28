import pygame
import numpy as np
import RacingEnvironment
from DQN_test import DQN_test

NUM_EPISODES=10000

#create game env
racingGame = RacingEnvironment.RacingEnvironment()
racingGame.fps=60

#create AI agent
dqn = DQN_test()

#---------
REPLACE_TARGET = 10

def testEnv():
    racingGame = RacingEnvironment.RacingEnvironment()
    while True:  
        observation_, reward, done = racingGame.step(0)
        racingGame.render(1)       

# def trainDQN():
#     env, test_env, buf, Q, Qt, OPT = dqn.create_everything()
#     # epsilon greedy exploration
#     dqn.EPSILON = dqn.STARTING_EPSILON
#     testRs = []
#     last25testRs = []
#     print("Training:")
    
#     for epi in range(NUM_EPISODES):    
#         #rest the game??
#         env.reset()
#         done =False
#         #if no reward collected in 150 ticks, it dies
#         autoDieCounter = 0
#         currentReward=0
#         obs_new, reward, done = env.step(0)
#         obs= np.array(obs_new)
        
#         isRender = False
#         #render every 5 episode
#         if epi%5==0:
#             isRender = True

#         while not done:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     return
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         pygame.quit()
#                         return
            
#             action = dqn.policy(env, obs) #choose an action
#             obs_new, reward, done = env.step(action)
#             obs_new= np.array(obs_new)


#             #if no reward collected, after counter increased to 150, it dies
#             if reward == 0:
#                 autoDieCounter += 1
#                 if autoDieCounter > 150:
#                     done = True
#             else:
#                 autoDieCounter = 0

#             score += reward

#             observation = obs_new

#             racingGame.render(action)    

def testTrain():
    dqn.train()

if __name__ == "__main__":
    testTrain()
