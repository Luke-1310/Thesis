import numpy as np
import pygame
import os
from environments.base_environment import BaseEnvironment
from environments.pedone import Pedone

#ATTENZIONE IL FORMATO è COLONNA,RIGA

class Map1Environment(BaseEnvironment):
    
    def __init__(self, width, height, cell_size, screen=None):
        
        # Inizializza tutto ciò che serve nella superclasse
        super().__init__(width, height, cell_size, screen)

        # Posizione iniziale e obiettivo dell'agente
        self.start_position=[2, 24]
        self.agent_position = self.start_position
        self.goal_positions = [(41, 5)]  # Posizione di arrivo

        # Carica le risorse specifiche della mappa
        self.load_assets()
        self.create_grid()

        #Mi voglio creare un attributo per capire che mappa sto trattando
        self.map_name = "Città"

        #NUOVA ROBA (DA VEDERE)     
        self.pedoni = []
        start = (0, 0)
        goal = (47, 21)
        path = self.find_path(self.map_pedone, start, goal, walkable_value=1, cost_matrix=self.cost_matrix)
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

        # Carica l'immagine del pedone
        self.pedone_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/pedone.png")
        self.pedone_image = pygame.transform.scale(self.pedone_image, (self.cell_size // 1.5, self.cell_size))

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
        self.map_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/città_map.png")
        self.map_image = pygame.transform.scale(self.map_image, (self.width * self.cell_size, self.height * self.cell_size))

        # Font
        pygame.font.init()
        self.font = pygame.font.Font('Progetto Tesi Privitera/assets/PixeloidSansBold.ttf', 20)


    def create_grid(self):

        #Crea la mappa e carico i percorsi delle auto nemiche con l'agente
        self.percorso1=[[14, 24],[14, 23],[14, 22],[14, 21],[14, 20],[15, 20],[16, 20],[17, 20],[18, 20],[19, 20],[20, 20],
                        [21, 20],[22, 20],[23, 20],[24, 20],[25, 20],[26, 20],[27, 20],[28, 20],[29, 20],[30, 20],
                        [31, 20],[32, 20],[33, 20],[34, 20],[35, 20],[36, 20],[37, 20],[38, 20],[39, 20],[40, 20],
                        [41, 20],[42, 20],[43, 20],[44, 20],[45, 20],[46, 20],[46, 19],[46, 18],[46, 17],[46, 16],
                        [46, 15],[46, 14],[46, 13],[46, 12],[46, 11],[46, 10],[46, 9],[46, 8],[46, 7],[46, 6],
                        [46, 5],[46, 4],[46, 3],[46, 2],[46, 1],[45, 1],[44, 1],[43, 1],[42, 1],[41, 1],[40, 1],
                        [39, 1],[38, 1],[37, 1],[36, 1],[35, 1],[34, 1],[33, 1],[32, 1],[31, 1],[30, 1],[29, 1],
                        [28, 1],[27, 1],[26, 1],[25, 1],[24, 1],[23, 1],[22, 1],[21, 1],[20, 1],[19, 1],[18, 1],
                        [17, 1],[16, 1],[15, 1],[14, 1],[13, 1],[12, 1],[11, 1],[10, 1],[9, 1],[8, 1],[7, 1],
                        [6, 1],[5, 1],[4, 1],[3, 1],[2, 1],[1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],[1, 7],
                        [1, 8],[1, 9],[1, 10],[1, 11],[1, 12],[1, 13],[1, 14],[1, 15],[1, 16],[1, 17],[1, 18],[1, 19],
                        [1, 20],[1, 21],[1, 22],[1, 23],[1, 24]]
        self.percorso2 = [[15, 15],[16, 15],[17, 15],[18, 15],[19, 15],[20, 15],[21, 15],[22, 15],[23, 15],[24, 15],[25, 15],[26, 15],
                           [27, 15],[28, 15],[29, 15],[30, 15],[31, 15],[32, 15],[33, 15],[34, 15],[34, 14],[34, 13],[34, 12],[34, 11],
                           [34, 10],[34, 9],[33, 9],[32, 9],[31, 9],[30, 9],[29, 9],[28, 9],[27, 9],[26, 9],[25, 9],[24, 9],[23, 9],
                           [22, 9],[21, 9],[20, 9],[19, 9],[18, 9],[17, 9],[16, 9],[15, 9],[14, 9],[13, 9],[13, 10],[13, 11],[13, 12],
                           [13, 13],[13, 14],[13, 15],[14, 15]]
        self.percorso3 = [[2, 8],[2, 7],[2, 6],[2, 5],[2, 4],[2, 3],[2, 2],[3, 2],[4, 2],[5, 2],[6, 2],[7, 2],[8, 2],[9, 2],[9, 3],
                           [9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9],[8, 9],[7, 9],[6, 9],[5, 9],[4, 9],[3, 9],[2, 9]]

        # Mappa dei percorsi
        self.percorsi = {
            1: self.percorso1,
            2: self.percorso2,
            3: self.percorso3
        }
        self.q_values = np.zeros((self.height, self.width, 2, 4))
        self.actions = ['up', 'down', 'right', 'left']
        self.traffic_lights = {
            (14, 11): 'green',
            (13, 11): 'green',
            (12, 10): 'red',
            (12, 9): 'red',
            (15, 9): 'red',
            (15, 10): 'red',
            (32, 20): 'green',
            (32, 19): 'green',
            (35, 19): 'green',
            (35, 20): 'green',
            (33, 18): 'red',
            (34, 18): 'red'
        }
        self.traffic_light_cycle = 0  # Contatore per il ciclo dei semafori
        self.traffic_light_duration = 120  # Durata del ciclo del semaforo in frame
        
        # Zone "sicure" dove fermarsi anche se il semaforo è rosso
        self.safe_zones = [(14, 10), (13, 10),(14, 9), (13, 9),
                      (34, 19), (33, 19),(34, 20), (33, 20)]
        #praticamnete qui si passa alla transizione e poi al percorso successivo4
        self.incroci = {
            (34, 10): [self.percorso2, self.percorso1],
            (2, 9): [self.percorso3, self.percorso1]
        }
        # Percorsi di transizione
        # stai passando dal percorso x al y (x,y) mediante questi passaggi 
        self.transizioni = {
            (2, 1): [(34, 10),(35, 10),(36, 10),(37, 10),(38, 10),(39, 10),(40, 10),(41, 10),(42, 10),(43, 10),(44, 10),(45, 10),(46, 10)],
            (3, 1): [(2, 9),(1, 9)]
        }
        self.cars = [
            {'position': [14, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [15, 15], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [2, 8], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
        ]
        
        #griglia per la mappa
        self.map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        #griglia per il pedone (PROBABILMENTE CORRETTA MA DA VEDERE) #i 2 sono per le striscie pedonali
        self.map_pedone = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1],
            [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        self.cost_matrix = []
        for y, row in enumerate(self.map_pedone):
            cost_row = []
            for x, val in enumerate(row):
                if val == 0:
                    cost_row.append(float('inf'))  # Non percorribile
                elif x == 0 or x == self.width-1 or y == 0 or y == self.height-1:
                    cost_row.append(10)  # Costo alto per i bordi
                elif val == 2:
                    cost_row.append(2)  # Costo intermedio per le strisce
                else:
                    cost_row.append(1)  # Costo normale
            self.cost_matrix.append(cost_row) 


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
        #print(Map1Environment.__module__)  LO STAMPA   

        def reset_game(self):
            super().reset_game()
            self.cars = [
                {'position': [14, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0},
                {'position': [15, 15], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0},
                {'position': [2, 8], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0}
            ]

