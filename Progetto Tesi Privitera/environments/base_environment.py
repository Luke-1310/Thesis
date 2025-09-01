import numpy as np
import pygame
import os
import random
import heapq
from environments.pedone import Pedone

class BaseEnvironment:
    
    def __init__(self, width, height, cell_size, screen = None, num_pedoni = 0, pedone_error_prob=0.0, route_change_probability=0, num_episodi=2000, realistic_mode=False):

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
        #self.traffic_rules_mode = realistic_mode 
    
    #Carica immagini e risorse
    def load_assets(self):
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    #Crea la griglia dell'ambiente
    def create_grid(self):
        raise NotImplementedError("Questo metodo non è stato implementato correttamente.")

    #Verifica se una delle auto si trova nel campo visivo dell'agente 
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

    #Controlla se ci sono pedoni nel campo visivo dell'agente 2x2
    def are_pedestrians_in_vision(self):
        
        agent_x, agent_y = self.agent_position
        vision_min_x = max(0, agent_x - 2)
        vision_max_x = min(self.width - 1, agent_x + 2)
        vision_min_y = max(0, agent_y - 2)
        vision_max_y = min(self.height - 1, agent_y + 2)
        
        #Controlla se pedoni esistono
        if not hasattr(self, 'pedoni') or not self.pedoni:
            return False
        
        for pedone in self.pedoni:
            
            ped_x, ped_y = pedone.position
            
            if vision_min_x <= ped_x <= vision_max_x and vision_min_y <= ped_y <= vision_max_y:
                return True
        
        return False

    #Ottieni stato completo della visione (auto + pedoni)
    def get_vision_state(self):
        
        cars_visible = int(self.is_car_in_vision()) 
        pedestrians_visible = int(self.are_pedestrians_in_vision())

        #Se si imposta la modalità realistica si devono considerare anche altre regole
        if getattr(self, 'realistic_mode', False):

            # Modalità realistica: includi anche lo stato del semaforo
            traffic_light = 0  # Default
            agent_pos_tuple = tuple(self.agent_position)
            
            if agent_pos_tuple in self.traffic_lights:
                traffic_light = 1 if self.traffic_lights[agent_pos_tuple] == 'red' else 2
            
            return cars_visible, pedestrians_visible, traffic_light
        else:
            return cars_visible, pedestrians_visible  
        # if self.realistic_mode:
            
        #     traffic_light_state = 0
        #     current_position = tuple(self.agent_position)

        #     #Controlla semaforo nelle vicinanze (non solo posizione esatta)
        #     for light_pos, state in self.traffic_lights.items():
        #         distance = abs(current_position[0] - light_pos[0]) + abs(current_position[1] - light_pos[1])
                
        #         if distance <= 1:  #Semaforo nelle vicinanze immediate
        #             traffic_light_state = 1 if state == 'green' else 2  #0=nessuno, 1=verde, 2=rosso
        #         break

        #     return cars_visible, pedestrians_visible, traffic_light_state
        # else:
        #     return cars_visible, pedestrians_visible
    
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
        if current_position in self.incroci and random.random() < self.route_change_probability:
            possible_routes = [route for route in self.incroci[current_position] if route != self.percorsi[car['route']]]#Recupera tutte le rotte alternative disponibili all’incrocio, escludendo quella attuale dell’auto
            
            #Se c'è un percorso disponibile, lo cambia
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
        return car.get('rotation', 0)  #Restituisce la rotazione della macchina, o 0 se non è definita

    def get_next_action(self, epsilon):
        
        if np.random.random() < epsilon:
            #Ho lo SFRUTTTAMENTO quando random < epsilon
            cars_visible, pedestrians_visible = self.get_vision_state()
            current_q = self.q_values[self.agent_position[1], self.agent_position[0], cars_visible, pedestrians_visible]
            #PER I SEMAFORI
            #cars_visible, pedestrians_visible, traffic_light = self.get_vision_state()
            #current_q = self.q_values[self.agent_position[1], self.agent_position[0], cars_visible, pedestrians_visible, traffic_light]
            return np.argmax(current_q)
        
        else:
            #Ho l'ESPLORAZIONE quando random >= epsilon
            return np.random.randint(4)
        
        #Prima era al contrario, favorendo l'esplorazione quando random < epsilon e lo sfruttamento quando random >= epsilon;
        #comportando un non raggiungimento dell'obiettivo, visto che l'agente non sfruttava le conoscenze acquisite

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

        # da rimuovere perché dovrebbero essere già presenti nel training        
        # self.update_car_position()
        # self.update_pedoni(self.pedoni)

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
            make_error = can_make_errors and random.random() < self.pedone_error_prob #Se il pedone può fare errori e se la probabilità casuale è inferiore alla probabilità di errore del pedone

            while True:
                goal = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                
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

            # Rigenera i pedoni con nuove posizioni e percorsi (qui inizializzo i pedoni)
            self.pedoni = []
            
            for i in range(self.num_pedoni):
                
                while True:
                    
                    start = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                    if self.map_pedone[start[1]][start[0]] == 1: #controllo se la cella è percorribile
                        break
               
                while True:
                    
                    goal = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                    if self.map_pedone[goal[1]][goal[0]] == 1 and goal != start:
                        break
                
                path = self.find_path(self.map_pedone, start, goal, walkable_value=(1, 2), cost_matrix=self.cost_matrix)
                
                if path:
                    # Assegna un valore di errore casuale tra 0 e self.pedone_error_prob -> Ogni pedone ha una tendenza all'errore diversa
                    error_prob = random.random() * self.pedone_error_prob
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

#--------------------------------------------------------------------------- FUNZIONI PER MODALITA' TRAFFIC RULES TRAINING ---------------------------------------------------------------------------

def get_next_action_traffic_rules(self, epsilon):
    """Azione specifica per traffic rules - AI pura senza logica automatica"""
    
    if np.random.random() < epsilon:
        # Sfruttamento basato solo su Q-values
        cars_visible = int(self.is_car_in_vision())
        return np.argmax(self.q_values[self.agent_position[1], self.agent_position[0], cars_visible])
    else:
        # Esplorazione pura - l'AI deve imparare TUTTO
        return np.random.randint(4)

def get_next_location_traffic_rules(self, action_index):
    """Movimento specifico per traffic rules - NESSUNA logica automatica semafori"""
    
    new_position = self.agent_position[:]
    
    # Calcola nuova posizione
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

    # ✅ SOLO controlli base - NO logica semafori automatica
    is_valid = self.is_valid_move_traffic_rules(new_position)
    
    if is_valid:
        self.prev_agent_position = self.agent_position[:]
        self.agent_position = new_position
    
    return is_valid

def is_valid_move_traffic_rules(self, new_position):
    """Controlli validità SOLO base per traffic rules - l'AI deve imparare i semafori"""
    
    # Controlli confini
    if not (0 <= new_position[0] < self.width and 0 <= new_position[1] < self.height):
        return False
    
    # Controlli ostacoli (muri, edifici)
    if self.map[new_position[1]][new_position[0]] != 1:
        return False
    
    # ✅ IMPORTANTE: NON controlla semafori automaticamente!
    # L'AI deve imparare da sola quando fermarsi!
    
    return True

def check_traffic_light_violation_at_position(self, position):
    """Controlla se c'è violazione semaforo in una posizione specifica"""
    
    # Verifica se la posizione ha un semaforo rosso
    if tuple(position) in self.traffic_lights:
        if self.traffic_lights[tuple(position)] == 'red':
            return True
    
    return False

def set_traffic_rules_mode(self, enabled=True):
    """Attiva/disattiva modalità traffic rules training"""
    self.traffic_rules_mode = enabled
    print(f"🚦 Modalità Traffic Rules: {'ATTIVA' if enabled else 'DISATTIVA'}")

def reset_for_traffic_rules(self, start_position, goal_position):
    """Reset specifico per traffic rules training"""
    
    # Posiziona agente alla posizione di start del percorso
    self.agent_position = list(start_position)
    self.agent_rotation = 0
    
    # Salva goal temporaneo per questo percorso
    self.current_traffic_goal = goal_position
    
    # Attiva modalità traffic rules
    self.set_traffic_rules_mode(True)
    
    # Reset normale dell'ambiente (auto, semafori, pedoni)
    self.traffic_light_cycle = 0
    self.car_in_vision = False
    self.prev_car_position = [car['position'] for car in self.cars]
    self.prev_agent_position = self.agent_position[:]
    
    # Rigenera pedoni (opzionale per traffic rules)
    # self.reset_pedoni_for_traffic_rules()

def check_traffic_goal_reached(self):
    """Verifica se goal del percorso traffic rules è raggiunto"""
    
    if hasattr(self, 'current_traffic_goal'):
        distance = abs(self.agent_position[0] - self.current_traffic_goal[0]) + abs(self.agent_position[1] - self.current_traffic_goal[1])
        return distance <= 2  # Margine di 2 celle
    
    return False

def get_traffic_light_at_position(self, position):
    """Ottiene stato semaforo in una posizione"""
    
    pos_tuple = tuple(position)
    if pos_tuple in self.traffic_lights:
        return self.traffic_lights[pos_tuple]
    return None

def get_nearby_traffic_lights(self, radius=2):
    """Ottiene semafori nelle vicinanze dell'agente"""
    
    agent_pos = self.agent_position
    nearby_lights = []
    
    for light_pos, state in self.traffic_lights.items():
        distance = abs(agent_pos[0] - light_pos[0]) + abs(agent_pos[1] - light_pos[1])
        if distance <= radius:
            nearby_lights.append({
                'position': light_pos,
                'state': state,
                'distance': distance
            })
    
    return nearby_lights

def display_traffic_rules_info(self, episode=None):
    """Display con informazioni aggiuntive per traffic rules"""
    
    # Display normale
    self.display(episode)
    
    # Overlay aggiuntivo per traffic rules
    if hasattr(self, 'current_traffic_goal'):
        goal = self.current_traffic_goal
        goal_rect = pygame.Rect(goal[0] * self.cell_size, goal[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 255, 0), goal_rect, 3)  # Goal giallo
    
    # Mostra violazioni recenti se implementato
    if hasattr(self, 'recent_violations'):
        for violation in self.recent_violations[-5:]:  # Ultime 5 violazioni
            # Disegna indicatori violazioni
            pass  