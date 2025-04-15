import numpy as np
import pygame
from virtual_environment_6 import VirtualEnvironment

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def print_q_table(q_table):
    print("Q-Table:")
    print(q_table)

def train_agent(env):
    epsilon = 1
    discount_factor = 0.9
    learning_rate = 0.1
    num_episodes= 10#2000

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

        print(f"Episode: {episode}, Steps: {steps}, Total Reward: {total_reward}")
        epsilon = max(0.01, epsilon * 0.9995)  # Delay più lento

    # Chiedi all'utente se vuole visualizzare i risultati
    show_choice = input("Vuoi visualizzare i risultati? (s/n): ")
    if show_choice.lower() == 's':
        evaluate_agent(env)
    
    # Dopo l'allenamento, chiedi all'utente se vuole salvare la Q-table
    save_choice = input("Vuoi salvare la Q-table? (s/n): ")
    if save_choice.lower() == 's':
        np.save('q_table.npy', env.q_values)
        print("Q-table salvata con successo.")

def show_results(env):
    try:
        q_table = np.load('q_table.npy')
        if q_table.shape != env.q_values.shape:
            print(f"Errore: La forma della Q-table caricata ({q_table.shape}) non corrisponde a quella attesa ({env.q_values.shape})")
            return
        env.q_values = q_table
        print("Q-table caricata con successo.")
        evaluate_agent(env)
    except FileNotFoundError:
        print("File Q-table non trovato. Assicurati di aver salvato una Q-table prima.")

def evaluate_agent(env):
    
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
        print("Obiettivo raggiunto!")
    
    else:
        print("L'agente ha perso.")

#1 Implementazione di una interfaccia grafica per il menù
def show_menu(screen, font):
    
    #Bottoni
    buttons = [
        {"text": "1. Allenare l'agente", "action": "train"},
        {"text": "2. Mostrare risultati", "action": "show"},
        {"text": "3. Uscire", "action": "exit"}
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


def main():
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.update()
    pygame.display.set_caption("Simulatore Agente")

    pygame.event.pump()# Forza aggiornamento della finestra
    font = pygame.font.SysFont(None, 36)

    env = VirtualEnvironment(48, 25, 32, screen)
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
            train_agent(env)
        
        elif action == "show":
            show_results(env)
        
        elif action == "exit":
            running = False

    pygame.quit()


if __name__ == "__main__":
    
    try:
        main()
    
    except Exception as e:
        print(f"Errore imprevisto: {e}")

