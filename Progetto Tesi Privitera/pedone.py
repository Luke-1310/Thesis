class Pedone:
    def __init__(self, start_pos, end_pos, griglia, speed=1):
        self.x, self.y = start_pos
        self.end_x, self.end_y = end_pos
        self.griglia = griglia  # salva la griglia per usi futuri
        self.speed = speed
        self.arrivato = False

        self.image = pygame.image.load("immagini/pedone.png").convert_alpha()  # se hai un'immagine, altrimenti usa un rettangolo

    def aggiorna(self):
        if self.arrivato:
            return

        dx = self.end_x - self.x
        dy = self.end_y - self.y
        distanza = (dx**2 + dy**2) ** 0.5

        if distanza < self.speed:
            self.x, self.y = self.end_x, self.end_y
            self.arrivato = True
        else:
            self.x += self.speed * dx / distanza
            self.y += self.speed * dy / distanza

    def disegna(self, schermo):
        if hasattr(self, 'image'):
            schermo.blit(self.image, (self.x, self.y))
        else:
            pygame.draw.rect(schermo, (200, 0, 0), pygame.Rect(self.x, self.y, 10, 10))
