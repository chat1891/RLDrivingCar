import math
import pygame

SCREEN_WIDTH = 1000#1920
SCREEN_HEIGHT = 600#1080

BOUNDARY_COLOR = (0, 0, 0)

CHECKPOINT_REWARD=1
ALIVE_REWARD=0
DIE_PENALTY=-1
FINALGOAL_REWARD = 10
CLOST_TO_WALL_PENALTY=0.5
CLOST_TO_WALL_PENALTY_2=0.25

#coordinates
class Coord:
    def __init__(self,ix,iy):
        self.x = ix
        self.y = iy

class Line:
    def __init__(self,iPoint1,iPoint2):
        self.point1 = iPoint1
        self.point2 = iPoint2

def getDistance(point1, point2):
    return(((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5)

def rotation(origin,curPoint,turnAngle):
    #rotate based on the origin startPt
    #2D rotation matrix
    rotatedX = origin.x + math.cos(turnAngle) * (curPoint.x - origin.x) - math.sin(turnAngle) * (curPoint.y - origin.y)
    rotatedY = origin.y + math.sin(turnAngle) * (curPoint.x - origin.x) + math.cos(turnAngle) * (curPoint.y - origin.y)
    rotatedPoint = Coord(rotatedX, rotatedY)
    return rotatedPoint

def rotateCorners(point1, point2, point3, point4, angle):

    centerPoint = Coord((point1.x + point3.x)/2, (point1.y + point3.y)/2)

    point1 = rotation(centerPoint,point1,angle)
    point2 = rotation(centerPoint,point2,angle)
    point3 = rotation(centerPoint,point3,angle)
    point4 = rotation(centerPoint,point4,angle)

    return point1, point2, point3, point4

class Ray:
    def __init__(self, ix, iy, idistance, iangle):
        self.x=ix
        self.y = iy
        self.distance =idistance
        self.angle = iangle
       
class Car:
    def __init__(self, ix, iy, raceTrackImage):
        #load car image
        self.loadedImage = pygame.image.load('car5.png').convert()
        self.carWidth = 25
        self.carHeight =15
        #scaled_size = (self.loadedImage.get_width() // 10, self.loadedImage.get_height() // 10)
        self.scaledOrigImage = pygame.transform.scale(self.loadedImage, (self.carWidth,self.carHeight))
        self.scaledImage = self.scaledOrigImage   #need to have a image for rotation reference
          
        #car coordinates set start potistion
        #self.point = Coord(ix,iy)
        self.position = Coord(ix,iy) #center #[543,1000]#[830, 920]
        self.x = ix
        self.y = iy
        
        self.raceTrackMap = raceTrackImage
        
        self.angle = math.radians(0)
        self.drivingAngle = self.angle
        self.velocity = 0
        self.deltaVelocity = 1
        self.maxVelocity = 15
        
        self.velocityX = 0
        self.velocityY = 0
        
        #image rectangle
        self.rectangle = self.scaledImage.get_rect()
        self.rectangle.center = (self.x, self.y)
        
        self.distances = []
        
        #car 4 points
        self.corner1 = Coord(self.position.x - self.carWidth / 2, self.position.y - self.carHeight / 2)
        self.corner2 = Coord(self.position.x + self.carWidth / 2, self.position.y - self.carHeight / 2)
        self.corner3 = Coord(self.position.x + self.carWidth / 2, self.position.y + self.carHeight / 2)
        self.corner4 = Coord(self.position.x - self.carWidth / 2, self.position.y + self.carHeight / 2)

        self.bottomLeft = self.corner1 #point bottom left
        self.topLeft = self.corner2 #point bottom right
        self.bottomRight = self.corner3 #point top right    
        self.topRight = self.corner4 #point top
        #self.get_rectangle_corners()
        self.isAlive = True
        
        self.rayCasts = []
        
        self.score =0
    
    # def get_rectangle_corners(self):
    #     # Top left corner
    #     top_right = (self.rectangle.center[0]+self.rectangle.width/2, self.rectangle.center[1]+self.rectangle.height/2)
        
    #     # Top right corner
    #     top_left = (self.rectangle.center[0]+self.rectangle.width/2, self.rectangle.center[1]-self.rectangle.height/2)
        
    #     # Bottom left corner
    #     bottom_right = (self.rectangle.center[0]-self.rectangle.width/2, self.rectangle.center[1]+self.rectangle.height/2)
        
    #     # Bottom right corner
    #     bottom_left = (self.rectangle.center[0]-self.rectangle.width/2, self.rectangle.center[1]-self.rectangle.height/2)
        
    #     self.bottomRight=Coord(bottom_right[0],bottom_right[1])
    #     self.bottomLeft=Coord(bottom_left[0],bottom_left[1])
        
    #     self.topLeft=Coord(top_left[0],top_left[1])
    #     self.topRight=Coord(top_right[0],top_right[1])
    
    #     return top_left, top_right, bottom_left, bottom_right  
    def acceleration(self,deltaVelo):
        self.velocity=deltaVelo
        # if self.velocity>self.maxVelocity:
        #     self.velocity = self.maxVelocity
        # elif self.velocity < -self.maxVelocity:
        #     self.velocity = -self.maxVelocity
        # else:
        #     self.velocity = self.velocity+deltaVelo
    
    def turn(self, turnDir):
        self.drivingAngle -= math.radians(2.5) * turnDir
        
        
    def action(self,actionChoice):
        if actionChoice == 0:
            pass
        elif actionChoice == 1:
            self.acceleration(self.deltaVelocity)
        elif actionChoice == 2:
            #????? will be changed back to -self.deltaVelocity
            self.acceleration(self.deltaVelocity)            
        elif actionChoice == 3:
            self.acceleration(self.deltaVelocity)
            self.turn(1)
        elif actionChoice == 4:
            self.acceleration(self.deltaVelocity)
            self.turn(-1)
        # elif actionChoice == 5:
        #     self.acceleration(-self.deltaVelocity)
        #     self.turn(1)
        # elif actionChoice == 6:
        #     self.acceleration(-self.deltaVelocity)
        #     self.turn(-1)
        # elif actionChoice == 7:
        #     self.acceleration(self.deltaVelocity)
        #     self.turn(-1)
        # elif actionChoice == 8:
        #     self.acceleration(self.deltaVelocity)
        #     self.turn(1)
        pass

    def update(self):
        
        self.angle = self.drivingAngle
        # Normalize the angle to be within -π to π or 0 to 2π range
        #self.angle = self.angle % (2 * math.pi)
        #new velocity vector
        rotateCoord = rotation(Coord(0,0), Coord(self.velocity,0), self.angle)
        self.velocityX=rotateCoord.x
        self.velocityY = rotateCoord.y

        self.x = self.x + self.velocityX
        self.y = self.y + self.velocityY

        self.position.x = self.x
        self.position.y = self.y
        self.rectangle.center = self.x, self.y

        self.corner1 = Coord(self.corner1.x + self.velocityX, self.corner1.y + self.velocityY)
        self.corner2 = Coord(self.corner2.x + self.velocityX, self.corner2.y + self.velocityY)
        self.corner3 = Coord(self.corner3.x + self.velocityX, self.corner3.y + self.velocityY)
        self.corner4 = Coord(self.corner4.x + self.velocityX, self.corner4.y + self.velocityY)

        self.bottomLeft ,self.topLeft ,self.bottomRight ,self.topRight  = rotateCorners(self.corner1, self.corner2, self.corner3, self.corner4, self.drivingAngle)
        self.scaledImage = pygame.transform.rotate(self.scaledOrigImage, 360-self.drivingAngle * 180 / math.pi)
        x, y = self.rectangle.center  
        self.rectangle = self.scaledImage.get_rect()  
        self.rectangle.center = (x, y)
        
    
    def evalCollision(self, raceMap):
        self.isAlive = True
        # check the 4 corners on rectangle
        # if the corner collide with the border then crash
        if raceMap.get_at((int(self.bottomLeft.x),int(self.bottomLeft.y))) == BOUNDARY_COLOR \
        or raceMap.get_at((int(self.bottomRight.x),int(self.bottomRight.y))) == BOUNDARY_COLOR \
        or raceMap.get_at((int(self.topRight.x),int(self.topRight.y))) == BOUNDARY_COLOR \
        or raceMap.get_at((int(self.topLeft.x),int(self.topLeft.y))) == BOUNDARY_COLOR:
            self.isAlive = False
            
    def calScore(self, checkPoint):
        #Separating Axis Theorem a method used in computational geometry to determine if two convex shapes intersect
        x1 = self.topLeft.x
        y1 = self.topLeft.y
        x2 = self.topRight.x
        y2 = self.topRight.y
        
        x3 = checkPoint.x1
        y3 = checkPoint.y1
        x4 = checkPoint.x2
        y4 = checkPoint.y2
        
        # Calculate denominators
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return False  # Lines are parallel
        
        # Calculate numerators
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        # Check if intersection bwteen 0 and 1
        if 0 <= t <= 1 and 0 <= u <= 1:
            # pt = math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1))
            # d = getDistance(Coord(self.x, self.y), Coord(pt[0], pt[1]))
            # if d < 20:
            self.score+= CHECKPOINT_REWARD
            return True
        else:
            return False
    
    def rayCast(self):
        self.rayCasts=[]
        rayAngles = [15,-15,35,-35,55,-55,-90,90,120,-120]
        for deg in rayAngles:
            self.CalculateRayCast(deg,self.raceTrackMap)
        
        #distances are observation of the car, it is state
        distances = []
        for ray in self.rayCasts:
            curDistance = ray.distance
            #revert the distance to 0 is far, 1 is close
            distances.append((1000-curDistance)/1000)
        
        distances.append(self.velocity / self.maxVelocity)
             
        return distances
        #ray1 = (self.x, self.y, self.drivingAngle)
        #ray2 = Ray(self.x, self.y, self.soll_angle - math.radians(30))
        
    
    def CalculateRayCast(self,dirDegree,raceTrack):
        len =0
        curAngle = self.angle + dirDegree
        ray_x = int(self.position.x + math.cos(math.radians(360 - curAngle)) * len)
        ray_y = int(self.position.y + math.sin(math.radians(360 - curAngle)) * len)
        
        # While not hitting the boarder, set max =300
        # the ray goes further and further
        while not raceTrack.get_at((ray_x, ray_y)) == BOUNDARY_COLOR and len < 350:
            len = len + 1
            ray_x = int(self.position.x + math.cos(math.radians(360 - curAngle)) * len)
            ray_y = int(self.position.y + math.sin(math.radians(360 - curAngle)) * len)
        
        
        distToBorder = int(math.sqrt(math.pow(ray_x - self.position.x, 2) + math.pow(ray_y - self.position.y, 2)))
        curRay = Ray(ray_x,ray_y,distToBorder, curAngle)
        self.rayCasts.append(curRay)
        
    def drawRayCasts(self, screen):
        for ray in self.rayCasts:
            pygame.draw.line(screen, (0, 255, 0), (self.position.x,self.position.y), (ray.x,ray.y), 1)
            pygame.draw.circle(screen, (0, 255, 0), (ray.x,ray.y), 5)
    
    def draw4corners(self, screen):
        
        pygame.draw.circle(screen, (0, 0, 255), (self.topLeft.x,self.topLeft.y), 3)
        pygame.draw.circle(screen, (0, 255, 0), (self.topRight.x,self.topRight.y), 3)
        pygame.draw.circle(screen, (255, 0, 255), (self.bottomLeft.x,self.bottomLeft.y), 3)
        pygame.draw.circle(screen, (0, 255, 255), (self.bottomRight.x,self.bottomRight.y), 3)
        
    def draw(self, window):
        window.blit(self.scaledImage, self.rectangle)
        
    def reset(self):

        self.x = 417
        self.y = 530
        self.position = Coord(self.x,self.y) #center #[543,1000]#[830, 920]
        self.velocityX = 0
        self.velocityY = 0
        self.velocity = 0
        self.angle = math.radians(0)
        self.drivingAngle= self.angle
        self.score = 0

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4
        #self.get_rectangle_corners()
        #car 4 points
        self.corner1 = Coord(self.position.x - self.carWidth / 2, self.position.y - self.carHeight / 2)
        self.corner2 = Coord(self.position.x + self.carWidth / 2, self.position.y - self.carHeight / 2)
        self.corner3 = Coord(self.position.x + self.carWidth / 2, self.position.y + self.carHeight / 2)
        self.corner4 = Coord(self.position.x - self.carWidth / 2, self.position.y + self.carHeight / 2)

        self.bottomLeft = self.corner1 #point bottom left
        self.topLeft = self.corner2 #point bottom right
        self.bottomRight = self.corner3 #point top right    
        self.topRight = self.corner4 #point top
        
        self.isAlive = True

    def reachedEndGoal(self, endGoal):
        if self.calScore(endGoal):
            return True
        return False

        
        
            
        
        