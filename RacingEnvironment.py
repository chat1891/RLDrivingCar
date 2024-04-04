import pygame
import car
import checkPoints

class RacingEnvironment:
    def __init__(self):
        pygame.init()
        #self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        
        self.history = []

        self.fps = 60
        self.width = car.SCREEN_WIDTH #1920#1000
        self.height = car.SCREEN_HEIGHT #1080#600
        

        self.gameScreen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racing Gaming")
        #self.screen.fill((0,0,0))
        self.gameReward = 0
        self.score = 0
        self.raceTrack = pygame.image.load('map6.jpg').convert()
        self.raceTrack  = pygame.transform.scale(self.raceTrack , (self.width, self.height))
        self.raceTrack_rect = self.raceTrack.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
        
        self.carInitX=417
        self.carInitY=502
 
        self.car = car.Car(self.carInitX, self.carInitY,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        
    def step(self, action):

        done = False
        self.car.action(action)
        self.car.update()
        reward = car.ALIVE_REWARD

        idx = 1
        endGoalCheckPoint = self.checkPoints[len(self.checkPoints)-1]
        for checkpoint in self.checkPoints:
            
            if idx >= len(self.checkPoints):
                idx = 1
            if checkpoint.isTriggered:
                if self.car.reachedEndGoal(endGoalCheckPoint):
                    done=True
                    reward += car.FINALGOAL_REWARD
                    break
                if self.car.calScore(checkpoint):
                    checkpoint.isTriggered = False
                    self.checkPoints[idx].isTriggered = True
                    reward += car.CHECKPOINT_REWARD
                    #print("reward:" + str(reward))
                    break

            idx = idx + 1

        #Check collision
        self.car.evalCollision(self.raceTrack)
        if not self.car.isAlive:
            reward += car.DIE_PENALTY
            done = True

        new_state = self.car.rayCast()
        
        #if the car too close to the wall -> 10, give -0.5 penalty
        #(300-15)/300
        #max ray is the ray with smallest distance
        maxRay = max(new_state[:-1])
        minRay =min(new_state[:-1])
        if maxRay > 0.985:
            reward +=car.CLOST_TO_WALL_PENALTY
            #print("too close to wall 15: "+str(maxRay))
        #if the car too close to the wall -> 15, give -0.25 penalty
        #(300-20)/300    
        if maxRay > 0.98:
            reward +=car.CLOST_TO_WALL_PENALTY_2
            #print("too close to wall 20: "+str(maxRay))
        
        #(1000-40)/1000
        #[:-1] eliminate the velocity, only consider min of ray casts
        if maxRay < 0.96:
            reward +=car.FAR_TO_WALL_REWARD
            #print("FAR to wall " + str(maxRay))
            
       
        if done:
            new_state = None

        return new_state, reward, done
    
    def render(self):
        
        #pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        #self.gameScreen.fill((0, 0, 0))
        
        drawRayCast = True
        drawCheckPoints = True
        
        #self.gameScreen.fill((0, 0, 0))
        #self.gameScreen.blit(self.raceTrack, (self.raceTrack_rect))       
        self.gameScreen.blit(self.raceTrack, (self.raceTrack_rect))   
        self.car.draw(self.gameScreen)
        if(drawRayCast):
            self.car.drawRayCasts(self.gameScreen)
            
        if drawCheckPoints:
            for cp in self.checkPoints:
                cp.draw(self.gameScreen)
        self.car.draw4corners(self.gameScreen)
        
        scoreText = self.font.render(f'Score {self.car.score}', True, pygame.Color('green'))
        self.gameScreen.blit(scoreText, dest=(0, 0))
        
        
        self.clock.tick(self.fps)
        
        pygame.display.update()
        #pygame.display.flip()
        
    def render2(self,action):
        drawRayCast = True
        drawCheckPoints = True
        
        #self.gameScreen.fill((0, 0, 0))
        #self.gameScreen.blit(self.raceTrack, (self.raceTrack_rect))       
        self.gameScreen.blit(self.raceTrack, (self.raceTrack_rect))   
        self.car.draw(self.gameScreen)
        if(drawRayCast):
            self.car.drawRayCasts(self.gameScreen)
        if drawCheckPoints:
            for cp in self.checkPoints:
                cp.draw(self.gameScreen)
        
        #self.clock.tick(self.fps)
        pygame.display.update()
        
    def reset(self):
        self.gameScreen.fill((0, 0, 0))
        self.car = car.Car(self.carInitX, self.carInitY,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        self.game_reward = 0
        
        # obs, reward, done = self.step(0)
        # return obs
    
    def FirstStep(self):
        self.gameScreen.fill((0, 0, 0))
        self.car = car.Car(self.carInitX, self.carInitY,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        self.game_reward = 0
        obs, reward, done = self.step(0)
        return obs
            

            
        