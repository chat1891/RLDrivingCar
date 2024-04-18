import pygame
import numpy as np
import RacingEnvironment
from DQN_test import DQN_test



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

N_EPISODES = 10000
TOTAL_GAMETIME = 1000

ddqn_scores = []
eps_history = []


def testTrain():
    curves = []
    
    curves += dqn.train()
    curLabel = "Reward"
    dqn.plot_arrays(curves, 'b', curLabel)


    
def keyBoardDriveControl():
    running = True
    clock = pygame.time.Clock()
    done = False
    # In your game loop
    #clock.tick(60)  # Caps the frame rate at 60 FPS
    env = RacingEnvironment.RacingEnvironment()
    env.fps=60
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if done:
            env.reset()   
            done = False     
        # Get pressed keys
        keys = pygame.key.get_pressed()

        # Move the car based on key presses
        if keys[pygame.K_LEFT]:
            new_state, reward, done = env.step(3)
        if keys[pygame.K_RIGHT]:
            new_state, reward, done = env.step(4)
        if keys[pygame.K_UP]:
            new_state, reward, done = env.step(1)
        if keys[pygame.K_DOWN]:
            new_state, reward, done= env.step(2)
        if keys[pygame.K_1]:
            new_state, reward, done = env.reset()
        env.render()  
        #clock.tick(60)
        

if __name__ == "__main__":
    #keyBoardDriveControl()
    testTrain()
