class Pedone:
    def __init__(self, start, goal, path=None, wait_steps = 5, path_callback=None):
        
        self.path = path or [] #Se non viene fornito un percorso, inizia da una lista vuota
        self.position = list(self.path[0]) if self.path else list(start) 
        self.goal = list(goal) 
        self.arrived = False

        self.wait_steps = wait_steps #numero di frame da aspettare prima di muoversi
        self.wait_counter = 0 

        self.pre_cross_wait = 0 #Contatore per far attendere il pedone prima dell'incrocio
        self.pre_cross_max = 3 #Aspetta 3 frame prima di attraversare l'incrocio

        self.path_callback = path_callback  # funzione per generare un nuovo path dopo che il pedone è arrivato

    def step(self, map_pedone):

        #Questa porzione di codice permette di eseguire un passo ogni 5 frame
        self.frame_counter = getattr(self, 'frame_counter', 0) + 1 
        
        if self.frame_counter < 5: 
            return
        
        self.frame_counter = 0

        if not self.path:
            # Se non c'è un percorso ma c'è una callback, chiedi un nuovo percorso
            if self.path_callback:
                new_goal, new_path = self.path_callback(tuple(self.position))
                if new_path:
                    self.goal = list(new_goal)
                    self.path = new_path
                    self.arrived = False
                    self.wait_counter = 0
                    self.pre_cross_wait = 0
            return

        x, y = self.position
        
        if self.position == self.goal:
            
            # Quando arriva, scegli una nuova destinazione e path
            
            if self.path_callback:
                new_goal, new_path = self.path_callback(tuple(self.position)) 
                
                if new_path:
                    self.goal = list(new_goal)
                    self.path = new_path  # Nuovo percorso
                    self.arrived = False
                    self.wait_counter = 0
                    self.pre_cross_wait = 0
                    return  # Importante: ritorna dopo aver impostato un nuovo path
                
                else:
                    # Nessuna nuova destinazione disponibile
                    self.arrived = True
            else:
                # Se non c'è una callback, il pedone è arrivato e non può più muoversi
                self.arrived = True
            
            return  # Ritorna dopo aver gestito l'arrivo
            
        
        if len(self.path) > 1:
            next_x, next_y = self.path[1]
            
            if map_pedone[next_y][next_x] == 2 and map_pedone[y][x] != 2:
                
                if self.pre_cross_wait < self.pre_cross_max:
                    self.pre_cross_wait += 1
                    return
                else:
                    self.pre_cross_wait = 0  # resetta il contatore dopo l'attesa

        # Attesa sulle strisce pedonali (più lunga)
        if map_pedone[y][x] == 2:
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps + 5:
                return
            
            self.wait_counter = 0

        # Attesa su marciapiede (normale)
        else:
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps:
                return
            self.wait_counter = 0

        # Avanza di una cella lungo il percorso
        if len(self.path) > 1:
            self.position = list(self.path[1])
            self.path.pop(0)
            if self.position == self.goal:
                self.arrived = True

