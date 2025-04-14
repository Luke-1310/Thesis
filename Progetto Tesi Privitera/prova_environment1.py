import pygame
import sys

# Inizializzazione di Pygame
pygame.init()

# Crea la finestra (non a schermo intero, così puoi vedere la barra delle applicazioni)
pygame.display.set_caption("Visualizzazione immagine")

# Carica l'immagine della mappa
self.map_image = pygame.image.load("Progetto Tesi Privitera/imgs/NuovaMappa50_30.png")
self.map_image = pygame.transform.scale(self.map_image, (width * cell_size, height * cell_size))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Disegna l'immagine di sfondo senza adattarla
    window.blit(background, (0, 0))

    # Aggiorna lo schermo
    pygame.display.flip()

# Esce da Pygame
pygame.quit()
sys.exit()
