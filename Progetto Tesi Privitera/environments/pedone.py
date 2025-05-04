class Pedone:
    def __init__(self, start, goal, path=None, wait_steps = 5):
        self.path = path or []
        self.position = list(self.path[0]) if self.path else list(start)
        self.goal = list(goal)
        self.arrived = False

        #rallentamento del pedone
        self.wait_steps = wait_steps
        self.wait_counter = 0

    def step(self):
        
        if not self.path or self.arrived:
            return
        
        self.wait_counter += 1

        # se il pedone non ha ancora raggiunto la meta, aspetta un certo numero di passi
        if self.wait_counter < self.wait_steps:
            
            return  # aspetta ancora
        
        self.wait_counter = 0  # resetta il contatore
        
        try:
            current_index = self.path.index(tuple(self.position))
            if current_index + 1 < len(self.path):
                self.position = list(self.path[current_index + 1])
            if self.position == list(self.path[-1]):
                self.arrived = True
        except ValueError:
            self.position = list(self.path[0])