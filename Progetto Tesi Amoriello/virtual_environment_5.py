import numpy as np
import pygame

class VirtualEnvironment:
    def __init__(self, width, height, cell_size):

        # Inizializzazione dei parametri di base dell'ambiente
        self.width = width  # larghezza della griglia
        self.height = height  # altezza della griglia
        self.cell_size = cell_size  # dimensione di ogni cella (in pixel)

        # Posizione iniziale e obiettivo dell'agente
        self.start_position = [2, 24]
        self.agent_position = self.start_position
        self.goal_positions = [(41, 5)]  # destinazione finale (parcheggio)

        self.agent_rotation = 0  # orientamento iniziale dell'agente (0 = su)
        self.FPS = 60  # frame per secondo per la simulazione
        self.clock = pygame.time.Clock()

        # Inizializzazione auto "nemica" 1
        self.car_position  = [14, 24]
        self.car_rotation = 0
        self.prev_agent_position = []
        self.prev_car_position = []
        self.prev_car2_position = []
        self.prev_car3_position = []
        
        self.car_in_vision = False  # flag che indica se un'auto è nella zona visiva dell'agente
        self.car_route_index = 0  # indice posizione attuale del percorso

        # lista delle coordinate del percorso della prima auto
        self.car_route=[[14, 24],[14, 23],[14, 22],[14, 21],[14, 20],[15, 20],[16, 20],[17, 20],[18, 20],[19, 20],[20, 20],
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
        
        # Percorsi e stati iniziali di altre due auto "nemiche"
        self.car2_route = [[15, 15],[16, 15],[17, 15],[18, 15],[19, 15],[20, 15],[21, 15],[22, 15],[23, 15],[24, 15],[25, 15],[26, 15],
                           [27, 15],[28, 15],[29, 15],[30, 15],[31, 15],[32, 15],[33, 15],[34, 15],[34, 14],[34, 13],[34, 12],[34, 11],
                           [34, 10],[34, 9],[33, 9],[32, 9],[31, 9],[30, 9],[29, 9],[28, 9],[27, 9],[26, 9],[25, 9],[24, 9],[23, 9],
                           [22, 9],[21, 9],[20, 9],[19, 9],[18, 9],[17, 9],[16, 9],[15, 9],[14, 9],[13, 9],[13, 10],[13, 11],[13, 12],
                           [13, 13],[13, 14],[13, 15],[14, 15]]
        self.car3_route = [[2, 8],[2, 7],[2, 6],[2, 5],[2, 4],[2, 3],[2, 2],[3, 2],[4, 2],[5, 2],[6, 2],[7, 2],[8, 2],[9, 2],[9, 3],
                           [9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9],[8, 9],[7, 9],[6, 9],[5, 9],[4, 9],[3, 9],[2, 9]]
        self.car2_position = [15, 15]
        self.car2_rotation = 0
        self.car2_route_index = 0
        self.car3_position = [2, 8]
        self.car3_rotation = 0
        self.car3_route_index = 0

        # Q-table: mappa Q con dimensioni (altezza, larghezza, visione_macchine, azioni)
        self.q_values = np.zeros((self.height, self.width, 2, 4))  # Aggiunta della dimensione per car_in_vision
        self.actions = ['up', 'down', 'right', 'left']

        # Semafori e ciclo di aggiornamento
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

        # Mappa stradale: 0 = muro, 1 = strada
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

        # Matrice dei reward (premi e penalità)
        self.reward_matrix = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        # Assegna +10000 alle celle del parcheggio
        for pos in self.goal_positions:
            self.reward_matrix[pos[1]][pos[0]] = 10000
        
        # Assegna -10 ai bordi delle strade
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 0:
                    self.reward_matrix[y][x] = -10

        # Inizializza Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Find The Parking")
        
        # Carica l'immagine della macchina
        self.agent_image = pygame.image.load("Progetto Tesi Amoriello/car.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (cell_size // 2, cell_size))

        # Carica l'immagine originale
        self.car_image = pygame.image.load("Progetto Tesi Amoriello/car2.png")

        # Ottieni le dimensioni originali dell'immagine
        original_width, original_height = self.car_image.get_size()

        # Calcola le nuove dimensioni
        new_width = int(original_width * 0.08)
        new_height = int(original_height * 0.08)

        # Scala l'immagine
        self.car_image = pygame.transform.scale(self.car_image, (new_width, new_height))

        self.car2_image = pygame.image.load("Progetto Tesi Amoriello/car2.png")
        self.car2_image = pygame.transform.scale(self.car2_image, (new_width, new_height))

        self.car3_image = pygame.image.load("Progetto Tesi Amoriello/car2.png")
        self.car3_image = pygame.transform.scale(self.car3_image, (new_width, new_height))
                
        # Carica l'immagine della mappa
        self.map_image = pygame.image.load("Progetto Tesi Amoriello/imgs/city_map.png")
        self.map_image = pygame.transform.scale(self.map_image, (width * cell_size, height * cell_size))

        pygame.font.init()
        self.font = pygame.font.Font('8-Bit-Madness.ttf', 24)
    
    # Verifica se una delle auto si trova nel campo visivo dell'agente
    def is_car_in_vision(self):
        agent_x, agent_y = self.agent_position
        vision_min_x = max(0, agent_x - 2)
        vision_max_x = min(self.width - 1, agent_x + 2)
        vision_min_y = max(0, agent_y - 2)
        vision_max_y = min(self.height - 1, agent_y + 2)
        
        car_positions = [self.car_position, self.car2_position, self.car3_position]
        
        for car_x, car_y in car_positions:
            if vision_min_x <= car_x <= vision_max_x and vision_min_y <= car_y <= vision_max_y:
                return True
        
        return False
    
    # Aggiorna la posizione di una singola auto secondo il suo percorso
    def _update_single_car_position(self, car_position, car_route, car_route_index, car_rotation):
        if car_route_index > 0:
            prev_position = car_route[car_route_index - 1]
        else:
            prev_position = car_route[-1]
        
        new_position = car_route[car_route_index]
        
        if tuple(new_position) in self.traffic_lights and self.traffic_lights[tuple(new_position)] == 'red':
            if not tuple(car_position) in self.safe_zones:
                return car_position, car_rotation, car_route_index
        
        car_position[0], car_position[1] = new_position
        
        if car_position[0] > prev_position[0]:
            car_rotation = -90  # Destra
        elif car_position[0] < prev_position[0]:
            car_rotation = 90   # Sinistra
        elif car_position[1] > prev_position[1]:
            car_rotation = 180  # Giù
        elif car_position[1] < prev_position[1]:
            car_rotation = 0    # Su
        
        car_route_index = (car_route_index + 1) % len(car_route)
        
        return car_position, car_rotation, car_route_index

     # Aggiorna posizione e rotazione di tutte e 3 le auto
    def update_car_position(self):
        self.car_position, self.car_rotation, self.car_route_index = self._update_single_car_position(
            self.car_position, self.car_route, self.car_route_index, self.car_rotation
        )
        self.car2_position, self.car2_rotation, self.car2_route_index = self._update_single_car_position(
            self.car2_position, self.car2_route, self.car2_route_index, self.car2_rotation
        )
        self.car3_position, self.car3_rotation, self.car3_route_index = self._update_single_car_position(
            self.car3_position, self.car3_route, self.car3_route_index, self.car3_rotation
        )
            
    # Restituisce la prossima azione secondo epsilon-greedy
    def get_next_action(self, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[self.agent_position[1], self.agent_position[0], int(self.car_in_vision)])
        else:
            return np.random.randint(4)
    
    # Controlla se una mossa è valida (non esce dai bordi e non va sui muri)
    def is_valid_move(self, new_position):
        if 0 <= new_position[0] < self.width and 0 <= new_position[1] < self.height:
            return self.map[new_position[1]][new_position[0]] == 1
        return False
    
    # Calcola la nuova posizione dell'agente in base all'azione scelta
    def get_next_location(self, action_index):
        new_position = self.agent_position[:]
        if self.actions[action_index] == "up":
            new_position[1] = max(0, self.agent_position[1] - 1)
            self.agent_rotation = 0
        elif self.actions[action_index] == "down":
            new_position[1] = min(self.height - 1, self.agent_position[1] + 1)
            self.agent_rotation = 180
        elif self.actions[action_index] == "right":
            new_position[0] = min(self.width - 1, self.agent_position[0] + 1)
            self.agent_rotation = -90
        elif self.actions[action_index] == "left":
            new_position[0] = max(0, self.agent_position[0] - 1)
            self.agent_rotation = 90

        is_valid = self.is_valid_move(new_position)
        if is_valid:
            if tuple(new_position) in self.traffic_lights and self.traffic_lights[tuple(new_position)] == 'red':
                if not tuple(self.agent_position) in self.safe_zones:
                    return False
            self.prev_agent_position = self.agent_position[:]
            self.agent_position = new_position
        return is_valid
        
    # Verifica se l'agente ha raggiunto l'obiettivo
    def check_goal(self):
        if tuple(self.agent_position) in self.goal_positions:
            return True
        return False
    
    # Verifica se l'agente ha fatto collisione
    def check_loss(self):
        if (self.agent_position in [self.car_position, self.car2_position, self.car3_position] or
            (self.agent_position == self.prev_car_position and self.car_position == self.prev_agent_position) or
            (self.agent_position == self.prev_car2_position and self.car2_position == self.prev_agent_position) or
            (self.agent_position == self.prev_car3_position and self.car3_position == self.prev_agent_position)):
            return True
        return False
    
    # Cambia lo stato dei semafori ogni X frame
    def update_traffic_lights(self):
        self.traffic_light_cycle += 1
        if self.traffic_light_cycle >= self.traffic_light_duration:
            self.traffic_light_cycle = 0
            for position in self.traffic_lights:
                if self.traffic_lights[position] == 'red':
                    self.traffic_lights[position] = 'green'
                else:
                    self.traffic_lights[position] = 'red'

    # Mostra la simulazione aggiornata (grafica + rotazioni + testi)
    def display(self, episode=None, path=None):
        self.screen.blit(self.map_image, (0, 0))
        #self.clock.tick(self.FPS)
        
        # Visualizza i semafori
        for position, state in self.traffic_lights.items():
            color = (255, 0, 0) if state == 'red' else (0, 255, 0)
            pygame.draw.circle(self.screen, color, (position[0] * self.cell_size + self.cell_size // 2, position[1] * self.cell_size + self.cell_size // 2), 10)
        #ruota la macchina agente
        rotated_agent_image = pygame.transform.rotate(self.agent_image, self.agent_rotation)
        rotated_rect = rotated_agent_image.get_rect()
        rotated_rect.center = (self.agent_position[0] * self.cell_size + self.cell_size // 2,
                               self.agent_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_agent_image, rotated_rect)
    
        self.update_car_position()
        self.update_traffic_lights()

        self._display_car(self.car_image, self.car_position, self.car_rotation)
        self._display_car(self.car2_image, self.car2_position, self.car2_rotation)
        self._display_car(self.car3_image, self.car3_position, self.car3_rotation)

        if episode is not None:  # Renderizza il contatore degli episodi
            episode_text = self.font.render(f'Episodio: {episode}', True, (255, 255, 255))
            text_rect = episode_text.get_rect()
            text_rect.topright = (self.width * self.cell_size - 10, 10)  # 10 pixel di margine da destra e dall'alto
            self.screen.blit(episode_text, text_rect)
        if path:
            for pos in path:
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                (pos[0] * self.cell_size + self.cell_size // 2, 
                                    pos[1] * self.cell_size + self.cell_size // 2), 5)
        pygame.display.flip()

    # Ruota e disegna una macchina nella posizione corretta
    def _display_car(self, car_image, car_position, car_rotation):
        rotated_car_image = pygame.transform.rotate(car_image, car_rotation)
        rotated_rect_car = rotated_car_image.get_rect()
        rotated_rect_car.center = (car_position[0] * self.cell_size + self.cell_size // 2, 
                                car_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_car_image, rotated_rect_car)
    
    # Riporta l'ambiente allo stato iniziale per un nuovo episodio
    def reset_game(self):
        self.agent_position = [2, 24]
        self.agent_rotation = 0
        self.car_position=[14, 24]
        self.car2_position=[15, 15]
        self.car3_position=[2, 8]
        self.car_rotation=0
        self.car2_rotation=0
        self.car3_rotation=0
        self.car_route_index=0
        self.car2_route_index = 0
        self.car3_route_index = 0
        self.traffic_light_cycle = 0
