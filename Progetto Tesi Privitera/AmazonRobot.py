import pygame
import sys

# Inizializzazione
pygame.init()

# Dimensioni della finestra (45 colonne × 25 righe)
CELL_SIZE = 32
COLUMNS = 45
ROWS = 25
WINDOW_WIDTH = COLUMNS * CELL_SIZE  # 1400
WINDOW_HEIGHT = ROWS * CELL_SIZE    # 800

# Crea la finestra
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Simulatore 40x25")

#player position and image
playerImg = pygame.image.load('Progetto Tesi Privitera/imgs/robot.png')
playerX = 370
playerY = 480
playerX_change = 0

# Carica e adatta immagine
image = pygame.image.load("Progetto Tesi Privitera/imgs/Senza titolo-1.png").convert()
image = pygame.transform.scale(image, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Ciclo principale
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(image, (0, 0))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
