import pygame

class CheckPoint:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.isTriggered = False
    
    def draw(self, win):
        pygame.draw.line(win, (0,255,0), (self.x1, self.y1), (self.x2, self.y2), 2)
        pygame.draw.line(win, (0,0,255), (477, 194), (489, 308), 3)
        if self.isTriggered:
            pygame.draw.line(win, (255,0,0), (self.x1, self.y1), (self.x2, self.y2), 2)

# the file of shame
def getCheckPoints():
    checkPoints = []

    checkPoints1 = CheckPoint(0,200,120,200)
    checkPoints.append(checkPoints1)
    
    checkPoints[len(checkPoints)-1].isTriggered = True
    
    return checkPoints
    