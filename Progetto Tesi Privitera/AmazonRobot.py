import pygame

pygame.init()

info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

print(f"Risoluzione totale: {screen_width} x {screen_height}")

# Imposta una finestra leggermente più piccola
window_width = screen_width - 100
window_height = screen_height - 100

screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Finestra con dimensioni sicure")

# Loop semplice
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
