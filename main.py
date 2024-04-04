import pygame
import numpy as np
import RacingEnvironment
from DQN_test import DQN_test

from ddqn_keras import DDQNAgent


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
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target= REPLACE_TARGET, batch_size=512, input_dims=11)
N_EPISODES = 10000
TOTAL_GAMETIME = 1000

ddqn_scores = []
eps_history = []

def testddqn():

    for e in range(N_EPISODES):
        
        racingGame.reset() #reset env 

        done = False
        score = 0
        counter = 0
        
        observation_, reward, done = racingGame.step(0)
        observation = np.array(observation_)

        gtime = 0 # set game time back to 0
        
        renderFlag = True # if you want to render every episode set to true

        if e % 10 == 0 and e > 0: # render every 10 episodes
            renderFlag = True

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = racingGame.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                racingGame.render()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
        print('episode: ', e,'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

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
    #testddqn()
