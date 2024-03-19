import pygame
import numpy as np
import RacingEnvironment

NUM_EPISODES=10000

racingGame = RacingEnvironment.RacingEnvironment()
racingGame.fps=60

#---------
REPLACE_TARGET = 10


def RunRacingGame():
    
    for e in range(NUM_EPISODES):    
        observation_, reward, done = racingGame.step(0)
        observation = np.array(observation_)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                
            done = False
            score = 0
            counter = 0

            
            #action = ddqn_agent.choose_action(observation)
            action = 1
            observation_, reward, done = racingGame.step(action)
            observation_ = np.array(observation_)

            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            observation = observation_

            racingGame.drawGameEnv(action)  
            #pygame.display.flip()
              
      

if __name__ == "__main__":
    RunRacingGame()
      