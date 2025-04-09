import pygame #pygame library 

#intialize the pygame
pygame.init()

#after that we have to create our screen with 800 px width and 600 px height
screen = pygame.display.set_mode((800,600))

#title and icon
pygame.display.set_caption("Space Invaders")
icon = pygame.image.load('Tutorial Pygame/Space Invaders/spaceship_icon.png')
pygame.display.set_icon(icon)

#player position and image
playerImg = pygame.image.load('Tutorial Pygame/Space Invaders/spaceship_pl_1.png')
playerX = 370
playerY = 480

playerX_change = 0

#"we are drawing on the screen"
def player(x,y):
    screen.blit(playerImg, (x, y))


#variable 
running = True

while running:
    
    #background color
    screen.fill((0,0,0))

    #here there're the events that will happen in the game window
    for event in pygame.event.get():

        #quit event
        if event.type == pygame.QUIT:
            running = False
    
        #if keystroke happens, check it
        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_LEFT:
                playerX_change = -0.3
            if event.key == pygame.K_RIGHT:
                playerX_change = +0.3

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                playerX_change = 0

    playerX += playerX_change

    #implements the borders for the spaceship
    if playerX <= 0:
        playerX = 0

    #the spaceship is 64x64 pixels big
    elif playerX >= 736:
        playerX = 736

    player(playerX, playerY)
    pygame.display.update()    

    pass