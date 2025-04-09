import pygame #pygame library 
import random

#intialize the pygame
pygame.init()

#after that we have to create our screen with 800 px width and 600 px height
screen = pygame.display.set_mode((800,600))

#title and icon
pygame.display.set_caption("Space Invaders")
icon = pygame.image.load('Tutorial Pygame/Space Invaders/spaceship_icon.png')
pygame.display.set_icon(icon)

#background
background = pygame.image.load('Tutorial Pygame/Space Invaders/background_img.jpg')

#player position and image
playerImg = pygame.image.load('Tutorial Pygame/Space Invaders/spaceship_pl_1.png')
playerX = 370
playerY = 480
playerX_change = 0

#enemy position and image
enemyImg = pygame.image.load('Tutorial Pygame/Space Invaders/alien.png')
enemyX = random.randint(0,800)
enemyY = random.randint(50,150)
enemyX_change = 0.3
enemyY_change = 10

#bullet position and image
#ready = you can't see it on the screen
#fire = you can see it on the screen
bulletImg = pygame.image.load('Tutorial Pygame/Space Invaders/bullet.png')
bulletX = 0
bulletY = 480
bulletX_change = 0
bulletY_change = 0.7
bullet_state = "ready"


#"we are drawing on the screen"
def player(x,y):
    screen.blit(playerImg, (x, y))

def enemy(x,y):
    screen.blit(enemyImg, (x, y))

def fire_bullet(x,y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x+16, y+10))

#variable 
running = True

while running:
    
    #background color
    screen.fill((20,24,82))

    #background image
    screen.blit(background, (0,0))

    #here there're the events that will happen in the game window
    for event in pygame.event.get():

        #quit event
        if event.type == pygame.QUIT:
            running = False
    
        #if keystroke happens, check it
        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_LEFT:
                playerX_change = -0.6
            if event.key == pygame.K_RIGHT:
                playerX_change = +0.6
            if event.key == pygame.K_SPACE:
                bulletX = playerX
                fire_bullet(bulletX,bulletY)

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

    #enemy movements
    enemyX += enemyX_change

    if enemyX <= 0:
        enemyX_change = 0.4
        enemyY += enemyY_change

    elif enemyX >= 736:
        enemyX_change = -0.4
        enemyY += enemyY_change

    #bullet movement
    if bullet_state == "fire":
        fire_bullet(bulletX, bulletY)
        bulletY -= bulletY_change

    player(playerX, playerY)
    enemy(enemyX, enemyY)
    pygame.display.update()    

    pass