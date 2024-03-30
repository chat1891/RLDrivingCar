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
 
        self.car = car.Car(417, 530,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        
    def step(self, action):

        done = False
        self.car.action(action)
        self.car.update()
        reward = car.ALIVE_REWARD

        idx = 1
        for checkpoint in self.checkPoints:
            
            if idx >= len(self.checkPoints):
                idx = 1
            if checkpoint.isTriggered:
                if self.car.calScore(checkpoint):
                    checkpoint.isTriggered = False
                    self.checkPoints[idx].isTriggered = True
                    reward += car.CHECKPOINT_REWARD
                    print("reward:" + str(reward))
                    break

            idx = idx + 1

        #Check collision
        self.car.evalCollision(self.raceTrack)
        if not self.car.isAlive:
            reward += car.DIE_PENALTY
            done = True

        new_state = self.car.rayCast()
        
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
        self.car = car.Car(417, 530,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        self.game_reward = 0
        
        # obs, reward, done = self.step(0)
        # return obs
    
    def FirstStep(self):
        self.gameScreen.fill((0, 0, 0))
        self.car = car.Car(417, 530,self.raceTrack)
        self.checkPoints = checkPoints.getCheckPoints()
        self.game_reward = 0
        obs, reward, done = self.step(0)
        return obs
            

            
        