import numpy as np
import pygame
import os
from environments.base_environment import BaseEnvironment
from environments.pedone import Pedone

class Map2Environment(BaseEnvironment):
    
    def __init__(self, width, height, cell_size, screen=None):
        
        # Inizializza tutto ciò che serve nella superclasse
        super().__init__(width, height, cell_size, screen)

        # Posizione iniziale e obiettivo dell'agente
        self.start_position=[22, 23]
        self.agent_position = self.start_position
        self.goal_positions = [(8, 5)]  # Posizione di arrivo

        # Carica le risorse specifiche della mappa
        self.load_assets()
        self.create_grid()

        self.map_name = "Foresta"

        #NUOVA ROBA (DA VEDERE)     
        self.pedoni = []
        start = (0, 0)
        goal = (47, 24)
        path = self.find_path(self.map_pedone, start, goal, walkable_value=1)
        print("Percorso pedone:", path)
        self.pedoni.append(Pedone(start, goal, path))

    def load_assets(self):
        # Carica tutte le immagini che ti servono

        # Inizializza Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))

        # Carica l'immagine della macchina
        self.agent_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/car.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size // 2, self.cell_size))

        #Carica l'immagine del pedone
        self.pedone_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/pedone.png")
        self.pedone_image = pygame.transform.scale(self.pedone_image, (self.cell_size // 2, self.cell_size))

        # Carica l'immagine originale
        self.car_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/car2.png")

        # Ottieni le dimensioni originali dell'immagine
        original_width, original_height = self.car_image.get_size()

        # Calcola le nuove dimensioni
        new_width = int(original_width * 0.08)
        new_height = int(original_height * 0.08)

        # Scala l'immagine
        self.car_image = pygame.transform.scale(self.car_image, (new_width, new_height))

        self.car2_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/car2.png")
        self.car2_image = pygame.transform.scale(self.car2_image, (new_width, new_height))

        self.car3_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/car2.png")
        self.car3_image = pygame.transform.scale(self.car3_image, (new_width, new_height))

        # Carica l'immagine della mappa
        self.map_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/foresta_map.png")
        self.map_image = pygame.transform.scale(self.map_image, (self.width * self.cell_size, self.height * self.cell_size))

        # Font
        pygame.font.init()
        self.font = pygame.font.Font('Progetto Tesi Privitera/assets/PixeloidSansBold.ttf', 20)


    def create_grid(self):

        #Crea la mappa e carico i percorsi delle auto nemiche con l'agente
        self.percorso1=[[46, 24],[46, 23],[46, 22],[46, 21],[46, 20],[46, 19],[46, 18],[46, 17],[46, 16],[46, 15],[46, 14],
                        [46, 13],[46, 12],[46, 11],[46, 10],[46, 9],[46, 8],[46, 7],[46, 6],[46, 5],[46, 4],
                        [46, 3],[46, 2],[46, 1],[45, 1],[44, 1],[43, 1],[42, 1],[41, 1],[40, 1],[39, 1],
                        [38, 1],[37, 1],[36, 1],[36, 2],[36, 3],[36, 4],[36, 5],[36, 6],[36, 7],[36, 8],
                        [36, 9],[36, 10],[37, 10],[38, 10],[39, 10],[40, 10],[41, 10],[42, 10],[43, 10],[44, 10],
                        [45, 10],[45, 11],[45, 12],[45, 13],[44, 13],[43, 13],[42, 13],[41, 13],[40, 13],[39, 13],[38, 13],
                        [37, 13],[36, 13],[35, 13],[34, 13],[33, 13],[32, 13],[32, 14],[32, 15],[32, 16],[32, 17],[32, 18],
                        [32, 19],[32, 20],[32, 21],[32, 22],[32, 23],[32, 24]]

        self.percorso2 = [[32, 2],[31, 2],[30, 2],[29, 2],[28, 2],[27, 2],[26, 2],[25, 2],[24, 2],[23, 2],[22, 2],[21, 2],
                            [20, 2],[19, 2],[18, 2],[17, 2],[16, 2],[15, 2],[14, 2],[14, 3],[14, 4],[14, 5],[14, 6],[14, 7],
                            [14, 8],[14, 9],[15, 9],[16, 9],[17, 9],[18, 9],[19, 9],[20, 9],[21, 9],[22, 9],[23, 9],[24, 9],[25, 9],
                            [26, 9],[27, 9],[28, 9],[29, 9],[30, 9],[31, 9],[32, 9],[32, 8],[32, 7],[32, 6],[32, 5],[32, 4],[32, 3]]

        self.percorso3 = [[13, 10],[12, 10],[11, 10],[10, 10],[9, 10],[8, 10],[7, 10],[6, 10],[5, 10],[4, 10],[3, 10],[2, 10],[2, 11],[2, 12],[2, 13],
                            [2, 14],[2, 15],[2, 16],[2, 17],[2, 18],[2, 19],[3, 19],[4, 19],[5, 19],[6, 19],[7, 19],[8, 19],[9, 19],[10,19],[11,19],[12,19],
                            [13,19],[13,18],[13,17],[13,16],[13,15],[13,14],[13,13],[13,12],[13,11]]

        # Mappa dei percorsi
        self.percorsi = {
            1: self.percorso1,
            2: self.percorso2,
            3: self.percorso3
        }
        self.q_values = np.zeros((self.height, self.width, 2, 4))
        self.actions = ['up', 'down', 'right', 'left']
        self.traffic_lights = {
            (14, 18): 'green',
            (13, 18): 'green',
            (12, 19): 'red',
            (12, 20): 'red',
            (15, 19): 'red',
            (15, 20): 'red',
            (35, 1): 'green',
            (35, 2): 'green',
            (38, 1): 'green',
            (38, 2): 'green',
            (36, 3): 'red',
            (37, 3): 'red'
        }
        self.traffic_light_cycle = 0  # Contatore per il ciclo dei semafori
        self.traffic_light_duration = 120  # Durata del ciclo del semaforo in frame
        
        # Zone "sicure" dove fermarsi anche se il semaforo è rosso, sono DENTRO GLI INCROCI
        self.safe_zones = [(13, 19), (14, 19),(14, 20), (13, 20),
                    (36, 1), (37, 1),(36, 2), (37, 2)]
        
        self.incroci = {
            (14, 2): [self.percorso2, self.percorso3],
            (13, 19): [self.percorso3, self.percorso1],
            (32, 13): [self.percorso1, self.percorso2]
        }
        # Percorsi di transizione (in rosso sullo schema, per passare da un pezzo all'altro)
        self.transizioni = {
            (2, 3): [(14, 2),(13, 2),(12, 2),(11, 2),(10, 2),(9, 2),(8, 2),(7, 2),(6, 2),(5, 2),(4, 2),(3, 2),(2, 2),(2, 3),(2, 4),(2, 5),(2, 6),(2, 7),(2, 8),(2,9),(2,10)],
            (3, 1): [(13, 19),(14, 19),(15, 19),(16, 19),(17, 19),(18, 19),(19, 19),(20, 19),(21, 19),(22, 19),(23, 19),(24, 19),(25, 19),(26, 19),(27, 19),(28, 19),(29, 19),(30, 19),(31, 19),(32, 19)],
            (1, 2): [(32, 13),(32, 12),(32, 11),(32, 10)]
        }

        self.cars = [
            {'position': [46, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [32, 2], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [13, 10], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
        ]
        
        #griglia per la mappa
        self.map = [
            #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]

        #griglia per il pedone (PROBABILMENTE CORRETTA MA DA VEDERE) i 2 sono per le striscie pedonali
        self.map_pedone = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        ]
            

        # Matrice dei reward (premi e penalità)
        self.reward_matrix = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        # Assegna +100000 alle celle del parcheggio
        for pos in self.goal_positions:
            self.reward_matrix[pos[1]][pos[0]] = 10000000
        
        # Assegna -10 ai bordi delle strade
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 0:
                    self.reward_matrix[y][x] = -10

        def reset_game(self):
            super().reset_game()
            self.cars = [
                {'position': [46, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
                {'position': [32, 2], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
                {'position': [13, 10], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
            ]
