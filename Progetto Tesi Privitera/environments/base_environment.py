import numpy as np
import pygame
import heapq
from environments.pedone import Pedone

class BaseEnvironment:

    def __init__(self, width, height, cell_size, screen = None, num_pedoni = 0, pedone_error_prob=0.0, route_change_probability=0, num_episodi=2000, realistic_mode=False, seed=None):

        #Inizializzazione dei parametri di base dell'ambiente
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.external_screen = screen is not None
        self.screen = screen or pygame.display.set_mode((width * cell_size, height * cell_size))

        self.agent_rotation = 0  #orientamento iniziale dell'agente (0 = su)
        self.FPS=5 #numero di frame per secondo per la simulazione
        self.clock= pygame.time.Clock()
        self.prev_agent_position = []
        self.prev_car_position = []
        self.car_in_vision = False #flag che indica se un'auto è nella zona visiva dell'agente

        self.num_pedoni = num_pedoni #numero di pedoni nell'ambiente
        self.pedone_error_prob = pedone_error_prob #probabilità che un pedone sbagli il percorso
        self.route_change_probability = route_change_probability  #Valore tra 0.0 e 1.0 -> 0.0 = non sbagliano mai, 1.0 = sbagliano sempre 
        self.num_episodi = num_episodi  #Numero di episodi per l'addestramento

        #Modalità realistica (con regole della strada)
        self.realistic_mode = realistic_mode #Parametro per la modalità realistica

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    #Carica immagini e risorse
    def load_assets(self):
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    #Crea la griglia dell'ambiente
    def create_grid(self):
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    #Verifica se una delle auto si trova nel campo visivo dell'agente
    #MAX → prende il valore più grande → serve per il bound inferiore
    #MIN → prende il valore più piccolo → serve per il bound superiore
    def is_car_in_vision(self):
        agent_x, agent_y = self.agent_position
        vision_min_x = max(0, agent_x - 2)
        vision_max_x = min(self.width - 1, agent_x + 2)
        vision_min_y = max(0, agent_y - 2)
        vision_max_y = min(self.height - 1, agent_y + 2)
        
        for car in self.cars:
            car_x, car_y = car['position']
            if vision_min_x <= car_x <= vision_max_x and vision_min_y <= car_y <= vision_max_y: #si crea un quadrato 5x5 intorno all'agente
                return True
        return False

    #Controlla se ci sono pedoni
    def are_pedestrians_in_vision(self):
        
        # Vede solo pedoni davanti (allineati) e su carreggiata o strisce
        if not hasattr(self, 'pedoni') or not self.pedoni:
            return False

        ax, ay = self.agent_position
        rot = self.agent_rotation
        ahead = 2  #profondità di vista in celle

        for pedone in self.pedoni:
            px, py = pedone.position

            #Devono essere DAVANTI e ALLINEATI alla corsia (niente laterali)
            in_front = False

            if rot == 0:        #su
                in_front = (px == ax and py < ay and ay - py <= ahead)
            
            elif rot == 180:    #giù
                in_front = (px == ax and py > ay and py - ay <= ahead)
            
            elif rot == -90:    #destra
                in_front = (py == ay and px > ax and px - ax <= ahead)
            
            elif rot == 90:     #sinistra
                in_front = (py == ay and px < ax and ax - px <= ahead)

            if not in_front:
                continue

            #Considera solo pedoni su strada o strisce (ignora marciapiede)
            on_road = (self.map[py][px] == 1)
            on_cross = (hasattr(self, 'map_pedone') and self.map_pedone[py][px] == 2)

            if on_road or on_cross:
                return True

        return False
    
    #Ottieni stato completo della visione (auto + pedoni)
    def get_vision_state(self):
        
        cars_visible = int(self.is_car_in_vision()) 
        pedestrians_visible = int(self.are_pedestrians_in_vision())

        #Se si imposta la modalità realistica si devono considerare anche altre regole
        if getattr(self, 'realistic_mode', False):

            #Semafori: #0=nessuno, 1=verde, 2=rosso
            traffic_light = 0
            agent_pos_tuple = tuple(self.agent_position)
            
            if agent_pos_tuple in self.traffic_lights:
                traffic_light = 2 if self.traffic_lights[agent_pos_tuple] == 'red' else 1
            
            return cars_visible, pedestrians_visible, traffic_light
        else:
            return cars_visible, pedestrians_visible  
    
    #Aggiorna la posizione di una singola auto secondo il suo percorso
    def update_car_position(self):
        self.prev_car_position = [car['position'][:] for car in self.cars]
        self.prev_agent_position = self.agent_position[:]
        
        for car in self.cars:
            prev_pos = car['position'][:]  #Memorizza la posizione precedente
            
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

    #Verifica se un'auto si trova su un incrocio e, se sì, con una certa probabilità, farle cambiare percorso seguendo una transizione definita
    def check_and_change_route(self, car):
        current_position = tuple(car['position']) #converte la posizione dell'auto in una tupla per poterla confrontare con le chiavi del dizionario degli incroci
        
        #Controlla se la posizione attuale è un incrocio (presente in self.incroci) e se un numero casuale tra 0 e 1 è inferiore al parametro di probabilità di cambio percorso (ex. 0.4 < 0.6 -> NON CAMBIA PERCORSO HO QUINDI IL 40% DI POSSIBILITÀ DI CAMBIARE PERCORSO)
        if current_position in self.incroci and self.rng.random() < self.route_change_probability:
            possible_routes = [route for route in self.incroci[current_position] if route != self.percorsi[car['route']]]#Recupera tutte le rotte alternative disponibili all’incrocio, escludendo quella attuale dell’auto
            
            #Se c'è un percorso disponibile, lo cambia
            if possible_routes:

                #Anche per le auto è necessario il seed per avere omogeneità nei test
                idx = int(self.rng.integers(0, len(possible_routes)))

                new_route = possible_routes[idx]
                new_route_index = list(self.percorsi.values()).index(new_route)
                transition_key = (car['route'], new_route_index + 1)
                
                if transition_key in self.transizioni:
                    car['transition_route'] = self.transizioni[transition_key]
                    car['in_transition'] = True
                    car['transition_index'] = 0
                    car['route'] = new_route_index + 1

    def _calculate_rotation(self, car):
        return car.get('rotation', 0)  #Restituisce la rotazione della macchina, o 0 se non è definita

    def get_next_action(self, epsilon, traffic_light=None):
        
        if np.random.random() < epsilon:   #ESPLORAZIONE
            return np.random.randint(5)
        
        else:   #SFRUTTAMENTO
            if getattr(self, 'realistic_mode', False):
                cars_visible, pedestrians_visible, traffic_light = self.get_vision_state()
                current_q = self.q_values[self.agent_position[1], self.agent_position[0],
                                        cars_visible, pedestrians_visible, traffic_light, :]
            else:
                cars_visible, pedestrians_visible = self.get_vision_state()
                current_q = self.q_values[self.agent_position[1], self.agent_position[0],
                                        cars_visible, pedestrians_visible, :]
            
            return np.argmax(current_q)


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
        
        elif self.actions[action_index] == "stay":
            new_position = self.agent_position[:]

        is_valid = self.is_valid_move(new_position)
        
        if is_valid:
            if not getattr(self, 'realistic_mode', False):  # Modalità SEMPLIFICATA
                
                # Blocca TUTTI i semafori (rossi E verdi)
                if hasattr(self, 'traffic_lights') and tuple(new_position) in self.traffic_lights:
                    return False  # BLOCCO FISICO - non può andare sui semafori
            
            # In modalità realistica: può andare sui semafori
            # (le regole rosso/verde saranno gestite nel training tramite ricompense)
                
            self.prev_agent_position = self.agent_position[:]
            self.agent_position = new_position
        return is_valid   

    def check_goal(self):
        if tuple(self.agent_position) in self.goal_positions:
            return True
        return False
    
    def check_loss(self):

        #Controlla se l'agente è in collisione con un'auto
        for car in self.cars:
            
            #Controlla se l'agente e la macchina sono nella stessa posizione
            if self.agent_position == car['position']:
                return True
            
            #Controlla se la posizione precedente della macchina è uguale alla posizione precedente dell'agente
            car_index = self.cars.index(car)
            
            #Tale controllo è necessario perché se due auto si incrociano, la collisione non viene rilevata visto che sono in due celle diverse, ma in realtà sono passate una sopra l'altra
            if (self.agent_position == self.prev_car_position[car_index] and car['position'] == self.prev_agent_position):
                return True
        
        #Collisione con pedoni
        for pedone in self.pedoni:
            if self.agent_position == pedone.position:
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
        
        #Semafori
        for position, state in self.traffic_lights.items():
            color = (255, 0, 0) if state == 'red' else (0, 255, 0)
            pygame.draw.circle(self.screen, color, (position[0] * self.cell_size + self.cell_size // 2, position[1] * self.cell_size + self.cell_size // 2), 10)

        #Agente
        rotated_agent_image = pygame.transform.rotate(self.agent_image, self.agent_rotation)
        rotated_rect = rotated_agent_image.get_rect()
        rotated_rect.center = (self.agent_position[0] * self.cell_size + self.cell_size // 2, self.agent_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_agent_image, rotated_rect)

        #Visualizza i pedoni(rendering)
        for pedone in self.pedoni:
            x, y = pedone.position
            self.screen.blit(self.pedone_image, (x * self.cell_size, y * self.cell_size))

        #Visualizza le auto(rendering)
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
        
        pygame.display.flip() #Aggiorna lo schermo mostrando tutto quello che ho disegnato nel buffer di pygame

    def _display_car(self, car_image, car_position, car_rotation):
        rotated_car_image = pygame.transform.rotate(car_image, car_rotation)
        rotated_rect_car = rotated_car_image.get_rect()
        rotated_rect_car.center = (car_position[0] * self.cell_size + self.cell_size // 2, car_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_car_image, rotated_rect_car)
    
    def pedone_path_callback(self, start, can_make_errors=True):

        #Ogni pedone ha una probabilità di errore da 0 a 1 scelto dal menu
        try:
            make_error = can_make_errors and (float(self.rng.random()) < self.pedone_error_prob) #Se il pedone può fare errori e se la probabilità casuale è inferiore alla probabilità di errore del pedone

            while True:
                goal = (int(self.rng.integers(0, self.width)), int(self.rng.integers(0, self.height)))

                #Se deve sbagliare cerca una cella che deve essere non percorribile
                if make_error:
                    if self.map_pedone[goal[1]][goal[0]] == 0 and goal != start:
                        break
                
                #Se non deve sbagliare allora una cella che deve essere percorribile
                else:
                    if self.map_pedone[goal[1]][goal[0]] == 1 and goal != start:
                        break
            
            #Modifica temporaneamente la mappa per trovare un percorso anche se include celle non percorribili
            if make_error:
                
                #Trova la cella percorribile più vicina alla destinazione
                nearest_valid = self._find_nearest_valid_cell(goal)
                
                #Partendo dalla posizione attuale del pedone, trova un percorso verso la cella percorribile più vicina (SEGUE LE REGOLE NORMALI) 
                valid_path = self.find_path(self.map_pedone, start, nearest_valid, walkable_value=(1, 2), cost_matrix=self.cost_matrix)
                
                if valid_path:

                    #Dall'ultima cella valida, va direttamente verso l'errore
                    error_segment = self._create_error_segment(valid_path[-1], goal)
                    if error_segment:
                        full_path = valid_path + error_segment[1:]  # Unisci i percorsi
                        return goal, full_path
            
            #Percorso normale (senza errori)
            path = self.find_path(self.map_pedone, start, goal, walkable_value=(1, 2), cost_matrix=self.cost_matrix)
            
            if path:
                return goal, path
            else:
                #Se non trova un percorso, ritorna al punto di partenza
                return start, [start]
                
        except Exception as e:
            print(f"Errore in pedone_path_callback: {e}")
            return start, [start]  #Fallback sicuro

    def reset_game(self):
            self.agent_position = self.start_position[:]
            self.agent_rotation = 0
            
            self.traffic_light_cycle = 0
            self.car_in_vision = False
            self.prev_car_position = [car['position'] for car in self.cars]
            self.prev_agent_position = self.agent_position[:]

            #Rigenera i pedoni con nuove posizioni e percorsi (qui inizializzo i pedoni)
            self.pedoni = []
            
            for i in range(self.num_pedoni):
                
                while True:
                    
                    #start = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                    #Uso l'RNG col seed anziché quello standard
                    start = (int(self.rng.integers(0, self.width)), int(self.rng.integers(0, self.height)))
                    if self.map_pedone[start[1]][start[0]] == 1: #controllo se la cella è percorribile
                        break
               
                while True:
                    
                    #goal = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                    goal = (int(self.rng.integers(0, self.width)), int(self.rng.integers(0, self.height)))
                    if self.map_pedone[goal[1]][goal[0]] == 1 and goal != start:
                        break
                
                path = self.find_path(self.map_pedone, start, goal, walkable_value=(1, 2), cost_matrix=self.cost_matrix)
                
                if path:
                    #Tendenza all'errore del singolo pedone deterministica dato il seed
                    error_prob = float(self.rng.random()) * self.pedone_error_prob
                    self.pedoni.append(Pedone(start, goal, path, wait_steps=5, path_callback=self.pedone_path_callback, error_prob=error_prob))
    
    #Funzione per calcolare la distanza tra due punti, utilizza la distanza di Manhattan, che è la somma delle differenze assolute delle coordinate x e y
    #Quindi quelle che andiamo a valutare solo le celle adiacenti (su, giù, sinistra, destra) e non quelle diagonali
    def heuristic(self, a, b):

        return abs(a[0] - b[0]) + abs(a[1] - b[1])#restituisce la somma delle differenze assolute delle coordinate x e y

    #Utilizza l'algoritmo A* per trovare il percorso più breve tra due punti in una griglia
    def find_path(self, grid, start, goal, walkable_value=(1,2), cost_matrix=None):
        
        width, height = self.width, self.height 
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]  #Su, giù, sinistra, destra

        open_set = [] #Lista di priorità per i nodi da esplorare
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, [start])) 
        closed_set = set() #Insieme dei nodi già esplorati 

        #Finché ci sono nodi da esplorare
        while open_set:
            _, cost, current, path = heapq.heappop(open_set) #Estrae il nodo con il costo più basso dalla lista di priorità 

            if current == goal:
                return path

            if current in closed_set: #Se il nodo è già stato esplorato, salta
                continue

            closed_set.add(current) #Aggiunge il nodo all'insieme dei nodi esplorati

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy

                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] in (1,2): #Controlla se il nodo è all'interno della griglia e se è percorribile
                    next_pos = (nx, ny) 

                    if next_pos not in closed_set: #Controlla se il nodo non è già stato esplorato
                        step_cost = cost_matrix[ny][nx] if cost_matrix else 1

                        #Aggiunge il nodo alla lista di priorità con il costo totale (costo attuale + costo del passo + distanza al goal)
                        heapq.heappush(open_set, (cost + step_cost + self.heuristic(next_pos, goal), cost + step_cost, next_pos, path + [next_pos])) 

        return None  #Nessun percorso trovato

    #Muovo il pedone lungo il percorso calcolato
    def move_pedone_along_path(self, pedone, path):
        
        if not path or pedone['position'] == path[-1]: #Controlla se il pedone è già arrivato alla fine del percorso
            return  
        
        current_index = path.index(tuple(pedone['position']))
        
        if current_index + 1 < len(path): 
            pedone['position'] = list(path[current_index + 1]) #Sposta il pedone alla prossima cella del percorso

    #Aggiorna la posizione di tutti i pedoni lungo i loro percorsi e passo i semafori
    def update_pedoni(self, pedoni):

        for pedone in pedoni:
            pedone.step(self.map_pedone, self.traffic_lights)

    def check_collision_type(self):
        
        for car in self.cars:
            if self.agent_position == car['position']:
                return "car"
            car_index = self.cars.index(car)
            if (self.agent_position == self.prev_car_position[car_index] and car['position'] == self.prev_agent_position):
                return "car"
        
        for pedone in self.pedoni:
            if self.agent_position == pedone.position:
                return "pedone"
        
        return None

    #Funzione per trovare la cella percorribile più vicina a una cella non percorribile
    def _find_nearest_valid_cell(self, target):
        
        min_dist = float('inf') #Inizializza la distanza minima a infinito
        nearest = None
        
        for y in range(self.height):
            for x in range(self.width):
                if self.map_pedone[y][x] == 1:  # Se è percorribile
                    dist = abs(x - target[0]) + abs(y - target[1])  # Manhattan distance
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)
        
        return nearest

    #Crea un segmento di percorso che porta dall'ultima cella valida al target di errore
    def _create_error_segment(self, valid_end, error_target):
        
        if valid_end == error_target:
            return [valid_end]
        
        path = [valid_end]
        current = list(valid_end)
        
        #Calcola quanti passi servono per raggiungere l'errore
        steps_needed = max(abs(current[0] - error_target[0]), abs(current[1] - error_target[1]))
        if steps_needed == 0:
            return path
        
        dx = (error_target[0] - current[0]) / steps_needed
        dy = (error_target[1] - current[1]) / steps_needed
        
        #Crea percorso diretto verso l'errore
        for i in range(1, steps_needed + 1):
            next_x = int(current[0] + dx * i)
            next_y = int(current[1] + dy * i)
            path.append((next_x, next_y))
        
        return path
    
    #Reinizializza la Q-table in base alla modalità attuale
    def reinitialize_q_values(self):
        
        if getattr(self, 'realistic_mode', False):
            # [y, x, auto_visibili, pedoni_visibili, semaforo(0/1/2), azione]
            self.q_values = np.zeros((self.height, self.width, 2, 2, 3, 5))
        
        else:
            # [y, x, auto_visibili, pedoni_visibili, azione]
            self.q_values = np.zeros((self.height, self.width, 2, 2, 5))
    
    #Restituisce True/False se la cella a destra dell'agente è/non è un bordo strada  
    def is_on_right_edge(self, position=None, rotation=None):
        
        if position is None:
            position = self.agent_position

        if rotation is None:
            rotation = self.agent_rotation

        x, y = position

        #Rotazione: 0=su, 180=giù, -90=destra, 90=sinistra
        if rotation == 0:        #su → destra è x+1
            rx, ry = x+1, y
        
        elif rotation == 180:    #giù → destra è x-1
            rx, ry = x-1, y
        
        elif rotation == -90:    #destra → destra è y+1
            rx, ry = x, y+1
        
        elif rotation == 90:     #sinistra → destra è y-1
            rx, ry = x, y-1
        
        else:
            rx, ry = x, y

    #Fuori mappa o non strada = bordo strada
        if not (0 <= rx < self.width and 0 <= ry < self.height):
            
            return True
        
        return self.map[ry][rx] == 0

    #Penalità o ricompensa per essere sul bordo destro della strada
    def right_edge_penalty(self):

        #Controllo modalità realistica
        if not getattr(self, 'realistic_mode', False):
            return 0
        
        if self.is_on_right_edge():
            return 0.45  #Nessuna penalità se è sul bordo destro
        else:
            return -0.9  #Penalità se non è sul bordo destro
    
   