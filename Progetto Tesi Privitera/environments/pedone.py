class Pedone:
    def __init__(self, start, goal, path=None):
        self.path = path or []
        self.position = list(self.path[0]) if self.path else list(start)
        self.goal = list(goal)
        self.arrived = False

    def step(self):

        print("Posizione attuale:", self.position)

        if not self.path or self.arrived:
            return
        try:
            current_index = self.path.index(tuple(self.position))
            if current_index + 1 < len(self.path):
                self.position = list(self.path[current_index + 1])
            if self.position == list(self.path[-1]):
                self.arrived = True
        except ValueError:
            self.position = list(self.path[0])