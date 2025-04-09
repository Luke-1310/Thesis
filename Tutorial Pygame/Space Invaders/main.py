import pygame #pygame library 
import random
import math

#audio
from pygame import mixer

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

#background sound
mixer.music.load('Tutorial Pygame/Space Invaders/background.wav')
mixer.music.play(-1) #-1 to play it in loop

#player position and image
playerImg = pygame.image.load('Tutorial Pygame/Space Invaders/spaceship_pl_1.png')
playerX = 370
playerY = 480
playerX_change = 0

#score

score_value = 0
font = pygame.font.Font('freesansbold.ttf', 32)

textX = 10
textY = 10

#gameover

game_over_font = pygame.font.Font('freesansbold.ttf', 64)

#enemy position and image
enemyImg = []
enemyX = []
enemyY = []
enemyX_change = []
enemyY_change = []
num_of_enemies = 6

for i in range(num_of_enemies):
    enemyImg.append(pygame.image.load('Tutorial Pygame/Space Invaders/alien.png'))
    enemyX.append(random.randint(0,735))
    enemyY.append(random.randint(50,150))
    enemyX_change.append(0.3)
    enemyY_change.append(10)

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
def show_score(x,y):
    score = font.render("Score: " + str(score_value), True, (255,255,255))
    screen.blit(score, (x, y))

def show_gameover():
    gameover = game_over_font.render("GAME OVER", True, (255,0,0))
    screen.blit(gameover, (200, 250))

def player(x,y):
    screen.blit(playerImg, (x, y))

def enemy(x,y, i):
    screen.blit(enemyImg[i], (x, y))

def fire_bullet(x,y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x+16, y+10))

def isCollision(enemyX, enemyY, bulletX, bulletY):
    distance = math.sqrt(math.pow(enemyX-bulletX, 2) + (math.pow(enemyY-bulletY, 2)))
    if distance < 27:
        return True
    else:
        return False

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
                if bullet_state == "ready":
                    bullet_sound = mixer.Sound('Tutorial Pygame/Space Invaders/laser.wav')
                    bullet_sound.play()
                    #get the current X-value of the spaceship
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
    for i in range(num_of_enemies):

        #game over
        if enemyY[i] > 450:
            for j in range(num_of_enemies):
                enemyY[j] = 9999
            game_over_text()
            break

        enemyX[i] += enemyX_change[i]

        if enemyX[i] <= 0:
            enemyX_change[i] = 2
            enemyY[i] += enemyY_change[i]

        elif enemyX[i] >= 736:
            enemyX_change[i] = -2
            enemyY[i] += enemyY_change[i]

        #Collision
        collision = isCollision(enemyX[i], enemyY[i], bulletX, bulletY)
        if collision:
            explosion_sound = mixer.Sound('Tutorial Pygame/Space Invaders/explosion.wav')
            explosion_sound.play()
            bulletY = 480
            bullet_state = "ready"
            score_value += 1
            enemyX[i] = random.randint(0,735)
            enemyY[i] = random.randint(50,150)
        
        enemy(enemyX[i], enemyY[i], i)

    #bullet movement
    if bulletY <= 0:
        bulletY= 480
        bullet_state = "ready"

    if bullet_state == "fire":
        fire_bullet(bulletX, bulletY)
        bulletY -= bulletY_change

   

    player(playerX, playerY)
    show_score(textX, textY)
    pygame.display.update()    

    pass