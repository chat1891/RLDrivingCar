import pygame

class CheckPoint:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.isTriggered = False
    
    def draw(self, win):
        pygame.draw.line(win, (0,0,255), (self.x1, self.y1), (self.x2, self.y2), 2)
        #pygame.draw.line(win, (0,0,255), (477, 194), (489, 308), 3)
        if self.isTriggered:
            pygame.draw.line(win, (255,0,0), (self.x1, self.y1), (self.x2, self.y2), 2)


def getCheckPoints():
    checkPoints=[]
    checkPointsCoords = [  
        # [394, 508],
        # [388, 555],
        # [432, 509],
        # [427, 555],
        [440, 509],
        [443, 558],
        [468, 510],
        [467, 554],
        [507, 506],
        [504, 553],
        [548, 500],
        [548, 552],
        [582, 499],
        [580, 551],
        [610, 549],
        [608, 495],
        [646, 490],
        [654, 542],
        [675, 538],
        [668, 486],
        [688, 479],
        [701, 531],
        [717, 528],
        [706, 477],
        [728, 468],
        [748, 518],
        [778, 506],
        [759, 455],
        [787, 441],
        [815, 485],
        [839, 470],
        [807, 428],
        [821, 416],
        [855, 457],
        [866, 448],
        [830, 409],
        [841, 393],
        [881, 432],
        [900, 408],
        [855, 376],
        [864, 364],
        [906, 392],
        [921, 371],
        [878, 343],
        [890, 329],
        [931, 349],
        [937, 342],
        [893, 317],
        [898, 307],
        [941, 338],
        [948, 317],
        [903, 296],
        [906, 283],
        [953, 304],
        [957, 293],
        [907, 274],
        [911, 263],
        [958, 285],
        [965, 271],
        [913, 254],
        [915, 240],
        [970, 251],
        [972, 232],
        [915, 225],
        [915, 211],
        [972, 211],
        [970, 185],
        [912, 181],
        [907, 168],
        [960, 151],
        [953, 135],
        [899, 154],
        [891, 141],
        [941, 110],
        [930, 96],
        [883, 132],
        [866, 121],
        [902, 69],
        [883, 58],
        [845, 110],
        [828, 104],
        [854, 45],
        [834, 42],
        [816, 105],
        [800, 102],
        [805, 38],
        [790, 35],
        [780, 102],
        [764, 103],
        [741, 37],
        [718, 48],
        [753, 110],
        [737, 121],
        [685, 79],
        [672, 94],
        [721, 141],
        [705, 155],
        [653, 117],
        [643, 131],
        [683, 173],
        [669, 187],
        [624, 150],
        [600, 176],
        [638, 215],
        [613, 237],
        [576, 194],
        [563, 206],
        [598, 251],
        [583, 260],
        [550, 212],
        [537, 218],
        [561, 271],
        [542, 278],
        [523, 226],
        [503, 229],
        [506, 284],
        [486, 283],
        [489, 231],
        [470, 232],
        [466, 283],
        [447, 283],
        [451, 227],
        [429, 219],
        [400, 270],
        [383, 260],
        [415, 209],
        [396, 199],
        [358, 238],
        [346, 231],
        [377, 185],
        [364, 174],
        [323, 208],
        [297, 184],
        [331, 149],
        [305, 128],
        [273, 165],
        [251, 155],
        [277, 108],
        [259, 98],
        [232, 147],
        [215, 145],
        [218, 87],
        [194, 87],
        [196, 144],
        [180, 150],
        [154, 91],
        [132, 102],
        [165, 158],
        [151, 169],
        [104, 123],
        [87, 135],
        [132, 188],
        [114, 207],
        [62, 169],
        [50, 191],
        [108, 225],
        [99, 250],
        [31, 237],
        [27, 270],
        [93, 280],
        [92, 322],
        [25, 328],
        [32, 380],
        [96, 375],
        [109, 415],
        [48, 439],
        [65, 476],
        [124, 450],
        [142, 470],
        [89, 525],
        [105, 538],
        [155, 479],
        [173, 487],
        [164, 551],
        [198, 554],
        [203, 493],
        [244, 498],
        [239, 555],
        [275, 554],
        [279, 501], 
        [314, 502], #end point
        [312, 552],
        [347, 505], #after start point, before car
        [344, 556],
        [369, 504],
        [367, 555], 
    ]
    

    i=0
    while i <= len(checkPointsCoords)-2:
        cur = CheckPoint(checkPointsCoords[i][0],checkPointsCoords[i][1],checkPointsCoords[i+1][0],checkPointsCoords[i+1][1])
        cur.isTriggered = False
        #checkPoints1 = CheckPoint(0,200,120,200)
        checkPoints.append(cur) 
        i+=2
    
    checkPoints[0].isTriggered = True
    #checkPoints[len(checkPoints)-1].isTriggered = True
    
    return checkPoints
    