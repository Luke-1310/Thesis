import numpy as np
import pygame
import random

class VirtualEnvironment:

    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        display_scale = 1
        
        #da cambiare
        self.start_position=[1, 1]
        self.agent_position = self.start_position

        #da cambiare
        self.goal_positions = [(35, 1)]  # Posizione di arrivo
        self.agent_rotation = 0  # Rotazione iniziale della macchina (0 gradi)
        self.FPS=5
        self.clock= pygame.time.Clock()
        self.prev_agent_position = []
        self.prev_car_position = []
        self.car_in_vision = False 

        #da cambiare
        self.percorso1 = [[46, 2],[45, 2],[44, 2],[43, 2]]
        
        self.percorso2 = [[40, 36],[40, 35],[40, 34],[40, 33]]

        self.percorso3 = [[2, 36],[2, 35],[2, 34],[2, 33],[2, 32],[3, 32]]

        # Mappa dei percorsi
        self.percorsi = {
            1: self.percorso1,
            2: self.percorso2,
            3: self.percorso3
        }
        self.q_values = np.zeros((self.height, self.width, 2, 4))
        self.actions = ['up', 'down', 'right', 'left']
        #da cambiare
        self.traffic_lights = {
            (20, 1): 'green',
            (20, 2): 'green',
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
        #da cambiare
        self.safe_zones = [(14, 10), (13, 10),(14, 9), (13, 9),
                      (34, 19), (33, 19),(34, 20), (33, 20)]
        self.incroci = {
            (34, 10): [self.percorso2, self.percorso1],
            (2, 9): [self.percorso3, self.percorso1]
        }
        #da cambiare
        # Percorsi di transizione
        self.transizioni = {
            (2, 1): [(34, 10),(35, 10),(36, 10),(37, 10),(38, 10),(39, 10),(40, 10),(41, 10),(42, 10),(43, 10),(44, 10),(45, 10),(46, 10)],
            (3, 1): [(2, 9),(1, 9)]
        }
        self.cars = [
            {'position': [46, 2], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [40, 36], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []},
            {'position': [2, 36], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': []}
        ]
        # self.map = [ #righe 38
        #     #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # ]

        self.map = [ #righe 30
            #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        self.reward_matrix = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        # Assegna +100000 alle celle del parcheggio
        for pos in self.goal_positions:
            self.reward_matrix[pos[1]][pos[0]] = 10000000
        
        # Assegna -10 ai bordi delle strade
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 0:
                    self.reward_matrix[y][x] = -10

        # Inizializza Pygame
        pygame.init()

        scaled_width = int(width * cell_size * display_scale)
        scaled_height = int(height * cell_size * display_scale)
        display_cell_size = int(cell_size * display_scale)

        self.screen = pygame.display.set_mode((scaled_width, scaled_height))
        pygame.display.set_caption("Find The Parking v.2")

        # Carica l'immagine della macchina (agente)
        self.agent_image = pygame.image.load("Progetto Tesi Privitera/imgs/car.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (display_cell_size // 2, display_cell_size))

        # Carica l'immagine originale e scala
        self.car_image = pygame.image.load("Progetto Tesi Privitera/imgs/car2.png")
        original_width, original_height = self.car_image.get_size()
        new_width = int(original_width * 0.08)
        new_height = int(original_height * 0.08)
        self.car_image = pygame.transform.scale(self.car_image, (new_width, new_height))

        self.car2_image = pygame.image.load("Progetto Tesi Privitera/imgs/car2.png")
        self.car2_image = pygame.transform.scale(self.car2_image, (new_width, new_height))

        self.car3_image = pygame.image.load("Progetto Tesi Privitera/imgs/car2.png")
        self.car3_image = pygame.transform.scale(self.car3_image, (new_width, new_height))

        # Carica e scala immagine della mappa
        self.map_image = pygame.image.load("Progetto Tesi Privitera/imgs/NuovaMappa50_30.png").convert()
        self.map_image = pygame.transform.scale(self.map_image, (scaled_width, scaled_height))

        pygame.font.init()
        self.font = pygame.font.Font('Progetto Tesi Privitera/8-Bit-Madness.ttf', 24)

        print("Finestra Pygame:", scaled_width, "x", scaled_height)

        
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


    def check_and_change_route(self, car):
        current_position = tuple(car['position'])
        if current_position in self.incroci and random.random() < 0.5:
            possible_routes = [route for route in self.incroci[current_position] if route != self.percorsi[car['route']]]
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
        # Disegna una griglia celeste sopra la mappa per aiutare l'allineamento
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (100, 200, 255), rect, 1)  # Colore e spessore bordo

        #self.clock.tick(self.FPS)
        
        for position, state in self.traffic_lights.items():
            color = (255, 0, 0) if state == 'red' else (0, 255, 0)
            pygame.draw.circle(self.screen, color, (position[0] * self.cell_size + self.cell_size // 2, position[1] * self.cell_size + self.cell_size // 2), 10)
        
        rotated_agent_image = pygame.transform.rotate(self.agent_image, self.agent_rotation)
        rotated_rect = rotated_agent_image.get_rect()
        rotated_rect.center = (self.agent_position[0] * self.cell_size + self.cell_size // 2, self.agent_position[1] * self.cell_size + self.cell_size // 2)
        self.screen.blit(rotated_agent_image, rotated_rect)
        
        self.update_car_position()
        
        for car in self.cars:
            rotation = self._calculate_rotation(car)
            self._display_car(self.car_image, car['position'], rotation)
        
        if episode is not None:
            episode_text = self.font.render(f'Episodio: {episode}', True, (255, 255, 255))
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

    #da cambiare
    def reset_game(self):
        self.agent_position = self.start_position[:]
        self.agent_rotation = 0
        self.cars = [
            {'position': [14, 24], 'route': 1, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0},
            {'position': [15, 15], 'route': 2, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0},
            {'position': [2, 8], 'route': 3, 'route_index': 0, 'in_transition': False, 'transition_index': 0, 'transition_route': [], 'rotation': 0}
        ]
        self.traffic_light_cycle = 0
        self.car_in_vision = False
        self.prev_car_position = [car['position'] for car in self.cars]
        self.prev_agent_position = self.agent_position[:]


