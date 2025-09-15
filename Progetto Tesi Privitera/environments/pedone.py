class Pedone:
    def __init__(self, start, goal, path=None, wait_steps = 5, path_callback=None, error_prob=0.0):
        
        self.path = path or [] #Se non viene fornito un percorso, inizia da una lista vuota
        self.position = list(self.path[0]) if self.path else list(start) 
        self.goal = list(goal) 
        self.arrived = False

        self.wait_steps = wait_steps #Numero di frame da aspettare prima di muoversi
        self.wait_counter = 0 

        self.pre_cross_wait = 0 #Contatore per far attendere il pedone prima dell'incrocio
        self.pre_cross_max = 3 #Aspetta 3 frame prima di attraversare l'incrocio per simulare il pedone che guarda a destra e sinistra

        self.path_callback = path_callback  #Funzione per generare un nuovo path dopo che il pedone è arrivato

        self.error_prob = error_prob  #Probabilità di sbagliare (0.0 = mai, 1.0 = sempre)
        self.waiting_for_light = False
        self.traffic_light_position = None

    def step(self, map_pedone, traffic_lights=None):

        #Questa porzione di codice permette di eseguire un passo ogni 5 frame
        self.frame_counter = getattr(self, 'frame_counter', 0) + 1 
        
        if self.frame_counter < 2: 
            return
        
        self.frame_counter = 0

        if not self.path:
            #Se non c'è un percorso ma c'è una callback, chiede un nuovo percorso
            if self.path_callback:
                new_goal, new_path = self.path_callback(tuple(self.position))
                if new_path:
                    self.goal = list(new_goal)
                    self.path = new_path
                    self.arrived = False
                    self.wait_counter = 0
                    self.pre_cross_wait = 0
                    self.waiting_for_light = False #Serve per far riprovare il pedone ad attraversare
            return

        x, y = self.position
        
        if self.position == self.goal:

            #Quando arriva, sceglie una nuova destinazione e un nuovo path
            if self.path_callback:
                try:
                    result = self.path_callback(tuple(self.position))        
                    #Verifica che il risultato sia una tupla valida
                    if result and isinstance(result, tuple) and len(result) == 2:
                        new_goal, new_path = result
                        if new_path:
                            self.goal = list(new_goal)
                            self.path = new_path
                            self.arrived = False
                            self.wait_counter = 0
                            self.pre_cross_wait = 0
                            self.waiting_for_light = False  # Reset lo stato di attesa
                            return
                    #Se arriviamo qui, qualcosa è andato storto
                    self.arrived = True
                except:
                    #Gestisci eventuali eccezioni
                    self.arrived = True
            else:
                self.arrived = True
            return  #Ritorna dopo aver gestito l'arrivo
            
        #Se il pedone stava aspettando un semaforo, controlla se ora può attraversare
        if self.waiting_for_light and traffic_lights and self.traffic_light_position:
            if traffic_lights.get(self.traffic_light_position) == 'red':
                
                #Il semaforo è diventato rosso per le auto, il pedone può attraversare
                self.waiting_for_light = False
                self.traffic_light_position = None

        #Controllo se il prossimo passo è sulle strisce pedonali
        if len(self.path) > 1:
            next_x, next_y = self.path[1]
            
            #Se sta per entrare sulle strisce
            if map_pedone[next_y][next_x] == 2 and map_pedone[y][x] != 2:
                
                # Controlla il semaforo se disponibile
                if traffic_lights:
                    # Trova il semaforo PIÙ VICINO alle strisce che sta per attraversare
                    nearest_light = None
                    min_distance = float('inf')
                    
                    for pos, state in traffic_lights.items():
                        # Calcola la distanza Manhattan dalle strisce
                        distance = abs(pos[0] - next_x) + abs(pos[1] - next_y)
                        
                        # Se è abbastanza vicino e più vicino del precedente
                        if distance <= 2 and distance < min_distance:
                            min_distance = distance
                            nearest_light = (pos, state)
                    
                    #Se ha trovato un semaforo vicino
                    if nearest_light:
                        pos, state = nearest_light
                        
                        #Se il semaforo è verde per le auto (rosso per i pedoni)
                        if state == 'green':
                            self.waiting_for_light = True
                            self.traffic_light_position = pos
                            return
                        else:
                            #Il semaforo è rosso per le auto, può attraversare
                            self.waiting_for_light = False
                            self.traffic_light_position = None
                
                #Se nessun semaforo rilevante è stato trovato o il semaforo è rosso per le auto
                if self.pre_cross_wait < self.pre_cross_max:
                    self.pre_cross_wait += 1
                    return
                else:
                    self.pre_cross_wait = 0

        #Attende sulle strisce pedonali
        if map_pedone[y][x] == 2:
            self.wait_counter += 1
            if self.wait_counter < self.wait_steps:
                return
            self.wait_counter = 0
        else:
            #Attende sul marciapiede
            self.wait_counter += 1
            if self.wait_counter < self.wait_steps - 3:  # Più veloce sul marciapiede
                return
            self.wait_counter = 0

        #Avanza lungo il percorso
        if len(self.path) > 1:
            self.position = list(self.path[1])
            self.path.pop(0)