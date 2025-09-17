import numpy as np
import pygame
from environments.base_environment import BaseEnvironment
from environments.pedone import Pedone

class Map2Environment(BaseEnvironment):

    def __init__(self, width, height, cell_size, screen=None, num_pedoni=0, pedone_error_prob=0.0, route_change_probability=0, num_episodi=2000, realistic_mode=False, seed = None):

        # Inizializza tutto ciò che serve nella superclasse
        super().__init__(width, height, cell_size, screen, num_pedoni, pedone_error_prob, route_change_probability, num_episodi, realistic_mode, seed)

        # Posizione iniziale e obiettivo dell'agente
        self.start_position=[22, 23]
        self.agent_position = self.start_position
        self.goal_positions = [(8, 5)]  # Posizione di arrivo

        # Carica le risorse specifiche della mappa
        self.load_assets()
        self.create_grid()

        self.map_name = "Foresta"

        self.realistic_mode = realistic_mode

        if getattr(self, 'realistic_mode', False):
            #[y, x, auto_visibili, pedoni_visibili, semafori (#0=nessuno, 1=verde, 2=rosso), azioni]
            self.q_values = np.zeros((self.height, self.width, 2, 2, 3, 5))
        else:
            #Q-table estesa: [y, x, auto_visibili, pedoni_visibili, azioni] 0 = non visibile, 1 = visibile
            self.q_values = np.zeros((self.height, self.width, 2, 2, 5))

    def load_assets(self):
        # Carica tutte le immagini che ti servono

        # Inizializza Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))

        # Carica l'immagine della macchina
        self.agent_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/car.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size // 2, self.cell_size))

        #Carica le immagini dei pedoni
        self.pedone_image = pygame.image.load("Progetto Tesi Privitera/assets/imgs/pedone.png")

        scale_factor = 0.9  # Fattore di scala per il pedone
        size = int(self.cell_size * scale_factor)  # Calcola la dimensione in base alla cella
        self.pedone_image = pygame.transform.scale(self.pedone_image, (size, size))

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

        self.percorso2 = [[33, 1],[32, 1],[31, 1],[30, 1],[29, 1],[28, 1],[27, 1],[26, 1],[25, 1],[24, 1],[23, 1],[22, 1],
                            [21, 1],[20, 1],[19, 1],[18, 1],[17, 1],[16, 1],[15, 1],[14, 1],[13, 1],[13, 2],[13, 3],[13, 4],
                            [13, 5],[13, 6],[13, 7],[13, 8],[13, 9],[13, 10],[14, 10],[15, 10],[16, 10],[17, 10],[18, 10],[19, 10],[20, 10],
                            [21, 10],[22, 10],[23, 10],[24, 10],[25, 10],[26, 10],[27, 10],[28, 10],[29, 10],[30, 10],[31, 10],[32, 10],
                            [33, 10],[33,9],[33,8],[33,7],[33,6],[33,5],[33,4],[33,3],[33,2]]

        self.percorso3 = [[13, 9],[12, 9],[11, 9],[10, 9],[9, 9],[8, 9],[7, 9],[6, 9],[5, 9],[4, 9],[3, 9],[2, 9],[2, 9],[1, 9],[1, 10],
                            [1, 11],[1, 12],[1, 13],[1, 14],[1, 15],[1, 16],[1, 17],[1, 18],[1, 19],[1, 20],[2, 20],[3, 20],[4, 20],[5,20],[6,20],[7,20],
                            [8,20],[9,20],[10,20],[11,20],[12,20],[13,20],[13,19],[13,18],[13,17],[13,16],[13,15],[13,14],[13,13],[13,12],[13,11],[13,10]]

        # Mappa dei percorsi
        self.percorsi = {
            1: self.percorso1,
            2: self.percorso2,
            3: self.percorso3
        }

        self.actions = ['up', 'down', 'right', 'left', 'stay']
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
        self.traffic_light_duration = 40  # Durata del ciclo del semaforo in frame
        
        # Zone "sicure" dove fermarsi anche se il semaforo è rosso, sono DENTRO GLI INCROCI
        self.safe_zones = [(13, 19), (14, 19),(14, 20), (13, 20),
                    (36, 1), (37, 1),(36, 2), (37, 2)]
        
        self.incroci = {
            (13, 1): [self.percorso2, self.percorso3],
            (13, 20): [self.percorso3, self.percorso1],
            (33, 13): [self.percorso1, self.percorso2]
        }
        # Percorsi di transizione (in rosso sullo schema, per passare da un pezzo all'altro)
        self.transizioni = {
            (2, 3): [(13, 1),(12, 1),(11, 1),(10, 1),(9, 1),(8, 1),(7, 1),(6, 1),(5, 1),(4, 1),(3, 1),(2, 1),(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 7),(1,8),(1,9)],
            (3, 1): [(13, 20),(14, 20),(15, 20),(16, 20),(17, 20),(18, 20),(19, 20),(20, 20),(21, 20),(22, 20),(23, 20),(24, 20),(25, 20),(26, 20),(27, 20),(28, 20),(29, 20),(30, 20),(31, 20),(32, 20)],
            (1, 2): [(33, 13),(33, 12),(33, 11),(33, 10)]
        }

        self.cars = [
            {'position': [46, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [33, 1], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [13, 9], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
        ]
        
        #griglia per la mappa
        self.map = [
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]

        #griglia per il pedone (PROBABILMENTE CORRETTA MA DA VEDERE) #2 = striscia pedonale #3 = edificio
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
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        ]

        #Inizializza la matrice dei costi
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
        # Assegna 10000 alle celle del parcheggio
        for pos in self.goal_positions:
            self.reward_matrix[pos[1]][pos[0]] = 10000
        
        # Assegna -10 ai bordi delle strade
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 0:
                    self.reward_matrix[y][x] = -10

    def reset_game(self):
        super().reset_game()
        self.cars = [
            {'position': [46, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [33, 1], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [13, 9], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
        ]
