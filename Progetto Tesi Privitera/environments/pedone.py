class Pedone:
    def __init__(self, start, goal, path=None, wait_steps = 5):
        
        self.path = path or [] #Se non viene fornito un percorso, inizia da una lista vuota
        self.position = list(self.path[0]) if self.path else list(start) 
        self.goal = list(goal) 
        self.arrived = False

        self.wait_steps = wait_steps #numero di frame da aspettare prima di muoversi
        self.wait_counter = 0 

    def step(self, map_pedone):
        
        if not self.path or self.arrived:
            return

        x, y = self.position
        
        #Se la cella è una striscia pedonale (2), aspetta più a lungo 
        if map_pedone[y][x] == 2:
            
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps + 5:
                return
            
            self.wait_counter = 0

        #Se la cella è un marciapiede (1), aspetta un frame e poi si muove
        else:
            
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps:
                return
            
            self.wait_counter = 0 # resetta il contatore

        #Controlla se il pedone è arrivato alla sua destinazione     
        try:
            current_index = self.path.index(tuple(self.position))

            #Controlla se il pedone è arrivato alla sua destinazione
            if current_index + 1 < len(self.path):
                self.position = list(self.path[current_index + 1]) #sposta il pedone alla prossima cella del percorso

            #Se il pedone è arrivato alla fine del percorso, imposta arrived su True
            if self.position == list(self.path[-1]):
                self.arrived = True
        
        except ValueError:
            self.position = list(self.path[0])