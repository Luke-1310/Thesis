import numpy as np
import pygame
import os
from environments.map1_environment import Map1Environment
from environments.map2_environment import Map2Environment

os.environ['SDL_VIDEO_CENTERED'] = '1' #Necessario perché, senza ulteriori precisazioni, la finestra viene creata in basso a destra

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def print_q_table(q_table):
     print("Q-Table:")
     print(q_table)

def train_agent(env, font):
    epsilon = 1
    discount_factor = 0.9
    learning_rate = 0.1
    num_episodes= 5#2000

    for episode in range(num_episodes):
        env.reset_game()
        total_reward = 0
        steps = 0

        while not (env.check_loss() or env.check_goal()):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.event.pump()

            env.update_traffic_lights()  # Aggiorna lo stato dei semafori

            action_index = env.get_next_action(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.car_in_vision)
            env.is_car_in_vision()  # Aggiorna lo stato di car_in_vision
            is_valid = env.get_next_location(action_index)

            if is_valid:
                reward = env.reward_matrix[env.agent_position[1]][env.agent_position[0]]
            elif not env.check_loss():
                reward = -10
            else:
                reward = -1000

            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            env.display(episode)
            pygame.time.wait(1)  # Breve pausa per gestire gli eventi


            total_reward += reward
            steps += 1

#            if steps > 1000:  # Previeni episodi troppo lunghi
#                break

        screen = env.screen
        screen.fill((255, 255, 255))  # Pulisce lo schermo

        draw_text(screen, f"Episodio: {episode}", 20, 20, font)
        draw_text(screen, f"Steps: {steps}", 20, 60, font)
        draw_text(screen, f"Total Reward: {total_reward}", 20, 100, font)

        pygame.display.flip()

        epsilon = max(0.01, epsilon * 0.9995)  # Delay più lento

    if show_yes_no_dialog(env.screen, font, "Vuoi visualizzare i risultati?"):
        evaluate_agent(env, font)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare la Q-table?"):
        np.save('q_table.npy', env.q_values)
        print("Q-table salvata con successo.")
        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, "Q-table salvata con successo.", 20, 20, font, (0, 150, 0))
        pygame.display.flip()
        pygame.time.wait(1500)

def show_results(env, font):
    try:
        q_table = np.load('q_table.npy')
        if q_table.shape != env.q_values.shape:
            print(f"Errore: La forma della Q-table caricata ({q_table.shape}) non corrisponde a quella attesa ({env.q_values.shape})")
            return
        env.q_values = q_table
        print("Q-table caricata con successo.")
        evaluate_agent(env, font)
    except FileNotFoundError:
        print("File Q-table non trovato. Assicurati di aver salvato una Q-table prima.")

def evaluate_agent(env, font):
    
    print("Inizio valutazione dell'agente")
    env.reset_game()
    path = []
    running = True
    while running and not (env.check_loss() or env.check_goal()):
        print(f"Posizione attuale: {env.agent_position}")
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                running = False
        
        action_index = np.argmax(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])
        env.get_next_location(action_index)
        path.append(env.agent_position[:])
        env.display(path=path)
        pygame.time.wait(500)  # Attende 500 ms tra ogni movimento
    
    if env.check_goal():
        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, "Obiettivo raggiunto!", 20, 20, font, (0, 150, 0))
        pygame.display.flip()
        pygame.time.wait(2000)

    else:
        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, "L'agente ha perso.", 20, 20, font, (200, 0, 0))
        pygame.display.flip()
        pygame.time.wait(2000)


#1 Implementazione di una interfaccia grafica per il menù
def show_menu(screen, font):
    
    #Bottoni
    buttons = [
        {"text": "1. Allenare l'agente", "action": "train"},
        {"text": "2. Mostrare risultati", "action": "show"},
        {"text": "3. Scegli la mappa", "action": "select_map"},
        {"text": "4. Uscire", "action": "exit"}
    ]

    button_rects = []

    # Riempie l'intera finestra di bianco
    screen.fill((255, 255, 255))

    # Titolo centrato
    title = font.render("Menu Principale", True, (0, 0, 0))
    screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 50))

    y = 150

    for button in buttons:
        rect = pygame.Rect(screen.get_width() // 2 - 150, y, 300, 50)
        pygame.draw.rect(screen, (0, 128, 255), rect)  # Rettangolo blu
        text_surface = font.render(button["text"], True, (255, 255, 255))  # Solo testo bianco
        screen.blit(text_surface, (rect.x + 20, rect.y + 10))
        button_rects.append((rect, button["action"]))
        y += 80

    pygame.display.flip()
    return button_rects

#2 Funzione per poter stampare del testo sul schermo
def draw_text(screen, text, x, y, font, color=(0, 0, 0)):

    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

#3 Funzione per poter chiedere all'utente, dal punto di vista grafico, se vuole o no vedere i risultati
def show_yes_no_dialog(screen, font, question):
    screen.fill((255, 255, 255))

    #(screen, text, x, y, font, color=(0, 0, 0)) devo calcolarmi la lunghezza del testo per poi centrarlo a dovere, non basta fare (screen.get_width()) // 2)-50
    #chiamo la funzione draw_text_centered
    draw_text_centered(screen, question, 100, font)

    button_width = 150
    button_height = 50
    spacing = 40  # spazio tra i due bottoni

    # Calcola la posizione centrale dei due bottoni insieme
    total_width = button_width * 2 + spacing
    start_x = (screen.get_width() - total_width) // 2
    y = 200

    yes_rect = pygame.Rect(start_x, y, button_width, button_height)
    no_rect = pygame.Rect(start_x + button_width + spacing, y, button_width, button_height)

    pygame.draw.rect(screen, (0, 200, 0), yes_rect)
    pygame.draw.rect(screen, (200, 0, 0), no_rect)

    yes_text = font.render("Sì", True, (255, 255, 255))
    no_text = font.render("No", True, (255, 255, 255))

    # Centra il testo all'interno dei bottoni
    screen.blit(yes_text, (yes_rect.centerx - yes_text.get_width() // 2, yes_rect.centery - yes_text.get_height() // 2))
    screen.blit(no_text, (no_rect.centerx - no_text.get_width() // 2, no_rect.centery - no_text.get_height() // 2))

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.collidepoint(pygame.mouse.get_pos()):
                    return True
                elif no_rect.collidepoint(pygame.mouse.get_pos()):
                    return False

#4 funzione che mi centra il testo, utile per stampare dei messaggi come "Vuoi salvare la Q-table"
def draw_text_centered(screen, text, y, font, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    x = (screen.get_width() - text_surface.get_width()) // 2
    screen.blit(text_surface, (x, y))

available_maps = {
    "1": ("Città", Map1Environment),
    "2": ("Foresta", Map2Environment),
}

#5 funzione che mi permette di scegliere la mappa
def select_map(screen, font):
    selecting = True
    selected_map_class = None

    # Lista dei bottoni da visualizzare
    buttons = []
    for key, (map_name, _) in available_maps.items():
        buttons.append({"text": f"{map_name}", "action": key})
    buttons.append({"text": "Torna al menu", "action": "back"})

    while selecting:
        screen.fill((255, 255, 255))

        # Titolo centrato
        draw_text_centered(screen, "Seleziona una mappa:", 50, font)

        y = 150
        button_rects = []

        for button in buttons:
            rect = pygame.Rect(screen.get_width() // 2 - 150, y, 300, 50)
            pygame.draw.rect(screen, (0, 128, 255), rect)
            text_surface = font.render(button["text"], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 20, rect.y + 10))
            button_rects.append((rect, button["action"]))
            y += 80

        pygame.display.flip()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                for rect, action in button_rects:
                    
                    if rect.collidepoint(pos):
                        
                        if action == "back":
                            return None  # Nessuna mappa selezionata, torna al menu
                        
                        elif action in available_maps:
                            map_name = available_maps[action][0]
                            selected_map_class = available_maps[action][1]
                            
                            # Feedback all'utente
                            screen.fill((255, 255, 255))
                            draw_text_centered(screen, f"Hai selezionato: {map_name}", screen.get_height() // 2 - 25, font)
                            pygame.display.flip()
                            pygame.time.delay(1000)  # Pausa di un secondo
                            selecting = False

    return selected_map_class#(screen) così facendo sto restituendo la classe, con (screen) la istanzio


def main():
    
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Centra la finestra
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.update()
    pygame.display.set_caption("Simulatore Agente")

    pygame.event.pump()# Forza aggiornamento della finestra
    font = pygame.font.SysFont(None, 36)

    env = Map1Environment(48, 25, 32, screen)
    running = True
    action = None

    while running:
        screen = pygame.display.set_mode((1536, 800))
        button_rects = show_menu(screen, font)
        
        waiting_for_input = True

        while waiting_for_input:
            
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_input = False
                    pygame.quit()
                    return
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    for rect, act in button_rects:
                        
                        if rect.collidepoint(mouse_pos):
                            action = act
                            waiting_for_input = False
                            break

        if action == "train":
            train_agent(env, font)
        
        elif action == "show":
            show_results(env,font)

        elif action == "select_map":

            selected_environment_class = select_map(screen, font)
            
            if selected_environment_class:
                env = selected_environment_class(48, 25, 32, screen)
                #print("Ambiente selezionato correttamente!")  # questo ora verrà eseguito

        elif action == "exit":
            running = False

    pygame.quit()

if __name__ == "__main__":
    
    try:
        main()
    
    except Exception as e:
        print(f"Errore imprevisto: {e}")

