import numpy as np
import pygame
import os
import random
import heapq

class BaseEnvironment:
    
    def __init__(self, width, height, cell_size, screen = None):

        # Inizializzazione dei parametri di base dell'ambiente
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.external_screen = screen is not None
        self.screen = screen or pygame.display.set_mode((width * cell_size, height * cell_size))

        self.agent_rotation = 0  # orientamento iniziale dell'agente (0 = su)
        self.FPS=5 # frame per secondo per la simulazione
        self.clock= pygame.time.Clock()
        self.prev_agent_position = []
        self.prev_car_position = []
        self.car_in_vision = False # flag che indica se un'auto è nella zona visiva dell'agente

    def load_assets(self):
        #Carica immagini e risorse
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    def create_grid(self):
        #Crea la griglia dell'ambiente
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    # Verifica se una delle auto si trova nel campo visivo dell'agente 
    def is_car_in_vision(self):
        agent_x, agent_y = self.agent_position
        vision_min_x = max(0, agent_x - 2)
        vision_max_x = min(self.width - 1, agent_x + 2)
        vision_min_y = max(0, agent_y - 2)
        vision_max_y = min(self.height - 1, agent_y + 2)
        
        for car in self.cars:
            car_x, car_y = car['position']
            if vision_min_x <= car_x <= vision_max_x and vision_min_y <= car_y <= vision_max_y:
                return True
        return False

    # Aggiorna la posizione di una singola auto secondo il suo percorso
    def update_car_position(self):
        self.prev_car_position = [car['position'][:] for car in self.cars]
        self.prev_agent_position = self.agent_position[:]
        
        for car in self.cars:
            prev_pos = car['position'][:]  # Memorizza la posizione precedente
            
            if car['in_transition']:
                if car['transition_index'] < len(car['transition_route']):
                    next_position = car['transition_route'][car['transition_index']]
                    car['transition_index'] += 1
                else:
                    car['in_transition'] = False
                    car['transition_index'] = 0
                    new_route = self.percorsi[car['route']]
                    car['route_index'] = min(range(len(new_route)), key=lambda i: ((new_route[i][0] - car['position'][0])**2 + (new_route[i][1] - car['position'][1])**2)**0.5)
                    next_position = new_route[car['route_index']]
            else:
                current_route = self.percorsi[car['route']]
                next_index = (car['route_index'] + 1) % len(current_route)
                next_position = current_route[next_index]
                
                if tuple(next_position) in self.traffic_lights and self.traffic_lights[tuple(next_position)] == 'red':
                    if tuple(car['position']) not in self.safe_zones:
                        continue
                
                if any(other_car['position'] == next_position for other_car in self.cars if other_car != car):
                    continue
                
                car['route_index'] = next_index

            # Aggiorna la posizione in tutti i casi
            car['position'] = next_position
            
            # Calcola e aggiorna la rotazione
            dx = car['position'][0] - prev_pos[0]
            dy = car['position'][1] - prev_pos[1]
            if dx > 0:
                car['rotation'] = -90
            elif dx < 0:
                car['rotation'] = 90
            elif dy > 0:
                car['rotation'] = 180
            elif dy < 0:
                car['rotation'] = 0
            
            self.check_and_change_route(car)

    #verifica se un'auto si trova su un incrocio e, se sì, con una certa probabilità, farle cambiare percorso seguendo una transizione definita
    def check_and_change_route(self, car):
        current_position = tuple(car['position'])
        #Controlla se la posizione attuale è un incrocio (presente in self.incroci) e se un numero casuale tra 0 e 1 è inferiore a 0.5 → questo crea una probabilità del 50% che il cambio percorso avvenga.
        if current_position in self.incroci and random.random() < 0.5:
            possible_routes = [route for route in self.incroci[current_position] if route != self.percorsi[car['route']]]#Recupera tutte le rotte alternative disponibili all’incrocio, escludendo quella attuale dell’auto
            #se c'è un percorso disponibile, lo cambia
            if possible_routes:
                new_route = random.choice(possible_routes)
                new_route_index = list(self.percorsi.values()).index(new_route)
                transition_key = (car['route'], new_route_index + 1)
                if transition_key in self.transizioni:
                    car['transition_route'] = self.transizioni[transition_key]
                    car['in_transition'] = True
                    car['transition_index'] = 0
                    car['route'] = new_route_index + 1

    def _calculate_rotation(self, car):
        return car.get('rotation', 0)  # Restituisce la rotazione della macchina, o 0 se non è definita

    def get_next_action(self, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            cars_visible = int(self.is_car_in_vision())
            return np.argmax(self.q_values[self.agent_position[1], self.agent_position[0], :, cars_visible])
        
    def is_valid_move(self, new_position):
        if 0 <= new_position[0] < self.width and 0 <= new_position[1] < self.height:
            return self.map[new_position[1]][new_position[0]] == 1
        return False
    
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

    def check_goal(self):
        if tuple(self.agent_position) in self.goal_positions:
            return True
        return False
    
    def check_loss(self):
        for car in self.cars:
            # Controlla se l'agente e la macchina sono nella stessa posizione
            if self.agent_position == car['position']:
                return True
            # Controlla se la posizione precedente della macchina è uguale alla posizione precedente dell'agente
            car_index = self.cars.index(car)
            if (self.agent_position == self.prev_car_position[car_index] and car['position'] == self.prev_agent_position):
                return True
        return False

    def update_traffic_lights(self):
        self.traffic_light_cycle += 1
        if self.traffic_light_cycle >= self.traffic_light_duration:
            self.traffic_light_cycle = 0
            for position in self.traffic_lights:
                if self.traffic_lights[position] == 'red':
                    self.traffic_lights[position] = 'green'
                else:
                    self.traffic_lights[position] = 'red'

    def display(self, episode=None, path=None):
        self.screen.blit(self.map_image, (0, 0))
        #self.clock.tick(self.FPS)
        
        for position, state in self.traffic_lights.items():
            color = (255, 0, 0) if state == 'red' else (0, 255, 0)
            pygame.draw.circle(self.screen, color, (position[0] * self.cell_size + self.cell_size // 2, position[1] * self.cell_size + self.cell_size // 2), 10)
        
        rotated_agent_image = pygame.transform.rotate(self.agent_image, self.agent_rotation)
        rotated_rect = rotated_agent_image.get_rect()
        rotated_rect.center = (self.agent_position[0] * self.cell_size + self.cell_size // 2, self.agent_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_agent_image, rotated_rect)
        
        self.update_car_position()
        
        #NUOVO CODICE PER I PEDONI
        
        # Aggiorna la posizione dei pedoni
        self.update_pedoni(self.pedoni)

        # Visualizza i pedoni
        for pedone in self.pedoni:
            x, y = pedone.position
            self.screen.blit(self.pedone_image, (x * self.cell_size, y * self.cell_size))

        #FINE
        for car in self.cars:
            rotation = self._calculate_rotation(car)
            self._display_car(self.car_image, car['position'], rotation)
        
        if episode is not None:
            episode_text = self.font.render(f'Episodio: {episode}', True, (51, 51, 51))
            text_rect = episode_text.get_rect()
            text_rect.topright = (self.width * self.cell_size - 10, 10)
            self.screen.blit(episode_text, text_rect)
        
        if path:
            for pos in path:
                pygame.draw.circle(self.screen, (255, 0, 0), (pos[0] * self.cell_size + self.cell_size // 2, pos[1] * self.cell_size + self.cell_size // 2), 5)
        
        pygame.display.flip()

    def _display_car(self, car_image, car_position, car_rotation):
        rotated_car_image = pygame.transform.rotate(car_image, car_rotation)
        rotated_rect_car = rotated_car_image.get_rect()
        rotated_rect_car.center = (car_position[0] * self.cell_size + self.cell_size // 2, car_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_car_image, rotated_rect_car)

    def reset_game(self):
            self.agent_position = self.start_position[:]
            self.agent_rotation = 0
            
            self.traffic_light_cycle = 0
            self.car_in_vision = False
            self.prev_car_position = [car['position'] for car in self.cars]
            self.prev_agent_position = self.agent_position[:]

            #Reset dei pedoni
            for pedone in self.pedoni:
                if pedone.path:
                    pedone.position = list(pedone.path[0])
                pedone.arrived = False

#ROBA NUOVA DA QUI IN GIU

    def heuristic(self, a, b):

        # Distanza di Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, grid, start, goal, walkable_value=1):
        """
        Trova il percorso più breve da start a goal su una griglia.
        grid: matrice 2D (lista di liste) dove walkable_value indica le celle percorribili.
        start, goal: tuple (x, y)
        walkable_value: valore che indica una cella percorribile (0 per pedoni, 1 per agenti)
        """
        width, height = self.width, self.height
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]  # Su, giù, sinistra, destra

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            
            _, cost, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path
            
            if current in closed_set:
                continue
            
            closed_set.add(current)

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == walkable_value:
                    next_pos = (nx, ny)
                    
                    if next_pos not in closed_set:
                        heapq.heappush(open_set, (cost + 1 + self.heuristic(next_pos, goal), cost + 1, next_pos, path + [next_pos]))
        
        return None  # Nessun percorso trovato

    def move_pedone_along_path(self, pedone, path):
        """
        Muove il pedone di un passo lungo il percorso.
        pedone: dict con almeno la chiave 'position'
        path: lista di tuple (x, y) che rappresenta il percorso da seguire
        """
        if not path or pedone['position'] == path[-1]:
            return  # Già arrivato o nessun percorso
        current_index = path.index(tuple(pedone['position']))
        if current_index + 1 < len(path):
            pedone['position'] = list(path[current_index + 1])

    def update_pedoni(self, pedoni):
        """
        Aggiorna la posizione di tutti i pedoni lungo i loro percorsi.
        pedoni: lista di oggetti Pedone
        """
        for pedone in pedoni:
            pedone.step()