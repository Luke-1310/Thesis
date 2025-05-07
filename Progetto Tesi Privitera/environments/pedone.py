class Pedone:
    def __init__(self, start, goal, path=None, wait_steps = 5):

        self.path = path or []
        self.position = list(self.path[0]) if self.path else list(start)
        self.goal = list(goal)
        self.arrived = False

        #rallentamento del pedone
        self.wait_steps = wait_steps
        self.wait_counter = 0

    def step(self, map_pedone):
        
        if not self.path or self.arrived:
            return

        x, y = self.position
        
        # Se la cella è una striscia pedonale (2), aspetta più a lungo 
        if map_pedone[y][x] == 2:
            
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps + 5:  # ad esempio, aspetta 5 frame in più
                return
            
            self.wait_counter = 0
        
        else:
            
            self.wait_counter += 1
            
            if self.wait_counter < self.wait_steps:
                return
            
            self.wait_counter = 0 # resetta il contatore
            
        try:
            current_index = self.path.index(tuple(self.position))

            if current_index + 1 < len(self.path):
                self.position = list(self.path[current_index + 1])
            
            if self.position == list(self.path[-1]):
                self.arrived = True
        
        except ValueError:
            self.position = list(self.path[0])