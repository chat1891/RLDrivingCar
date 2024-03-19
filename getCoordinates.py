# # import the required library
# import cv2

# # define a function to display the coordinates of

# # of the points clicked on the image
# def click_event(event, x, y, flags, params):
#    if event == cv2.EVENT_LBUTTONDOWN:
#       print(f'({x},{y})')
      
#       # put coordinates as text on the image
#       cv2.putText(img, f'({x},{y})',(x,y),
#       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
#       # draw point on the image
#       cv2.circle(img, (x,y), 3, (0,255,255), -1)
 
# # read the input image
# img = cv2.imread('map6.jpg')


# # create a window
# cv2.namedWindow('Point Coordinates')
# cv2.resizeWindow('Point Coordinates', 1000, 600)

# # bind the callback function to window
# cv2.setMouseCallback('Point Coordinates', click_event)

# # display the image
# while True:
#    cv2.imshow('Point Coordinates',img)
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27:
#       break
# cv2.destroyAllWindows()


import pygame

pygame.init()

# Load the image
original_image = pygame.image.load('map6.jpg')
original_width, original_height = original_image.get_size()

# Window size
width, height = 1000, 600  # Example size, you can set this to your screen size
screen = pygame.display.set_mode((width, height))

# Scale the image to fit the window
scaled_image = pygame.transform.scale(original_image, (width, height))
#raceTrack_rect = raceTrack.get_rect().move(0, 0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the mouse position
            x, y = event.pos

            # Translate the mouse position to the original image's coordinate system
            # original_x = int(x * (original_width / width))
            # original_y = int(y * (original_height / height))

            print(f'Original image coordinates: ({x}, {y})')
            
            pygame.draw.circle(screen, (0, 255, 0), (x, y), 30)
            font = pygame.font.SysFont(None, 24)
            text = font.render(f'({x},{y})', True, (255, 0, 0))
            screen.blit(text, (x, y))

    screen.blit(scaled_image, (0, 0))
    pygame.display.flip()

pygame.quit()
