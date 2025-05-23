import numpy as np
import pygame
import os
import sys
from environments.map1_environment import Map1Environment
from environments.map2_environment import Map2Environment

import matplotlib.pyplot as plt       # Per disegnare i grafici
import pandas as pd                   # Per gestire e analizzare dati in modo ordinato

os.environ['SDL_VIDEO_CENTERED'] = '1' #Necessario perché, senza ulteriori precisazioni, la finestra viene creata in basso a destra

np.set_printoptions(precision=3, suppress=True, linewidth=200)

# La Q-table è una tabella che usa un agente di Q-learning (un tipo di reinforcement learning) per imparare quale azione fare in ogni stato.
#     Ogni riga rappresenta uno stato (ad esempio: "sono all'incrocio, il semaforo è rosso").
#     Ogni colonna rappresenta una azione possibile (tipo "vai dritto", "gira a destra", "aspetta").
#     Ogni valore dentro la tabella (Q-value) dice quanto è buona quell'azione in quello stato.

def print_q_table(q_table):
     print("Q-Table:")
     print(q_table)

def train_agent(env, font):
    epsilon = 1
    discount_factor = 0.9
    learning_rate = 0.1
    num_episodes = 250  # Come nel file 5
    episode_data = []  # lista che contiene (episodio, step, reward)
    collision_list = []  # Lista per tenere traccia delle collisioni cumulative
    collision_count = 0

    for episode in range(num_episodes):
        env.reset_game()
        total_reward = 0
        steps = 0

        # STRUTTURA DEL FILE 5: while not (env.check_loss() or env.check_goal())
        while not (env.check_loss() or env.check_goal()):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.event.pump()

            env.update_traffic_lights()  # Aggiorna lo stato dei semafori
            env.update_pedoni(env.pedoni)  # Aggiorna lo stato dei pedoni (AGGIUNTO)

            action_index = env.get_next_action(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.car_in_vision)
            env.is_car_in_vision()  # Aggiorna lo stato di car_in_vision
            is_valid = env.get_next_location(action_index)

            # LOGICA DEL FILE 5: Gestione reward nel ciclo
            if is_valid:
                reward = env.reward_matrix[env.agent_position[1]][env.agent_position[0]]
            elif not env.check_loss():
                reward = -10
            else:
                reward = -100  # Questo reward non verrà mai usato nel ciclo

            # Q-learning update
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            
            env.display(episode)
            pygame.time.wait(1)  # Breve pausa per gestire gli eventi

            total_reward += reward
            steps += 1

            if steps > 1000:  # Previeni episodi troppo lunghi (come nel file 5)
                break

        # LOGICA DEL FILE 5: Gestione collisioni DOPO il ciclo
        if env.check_loss():
            collision_count += 1

        # Aggiungi il conteggio cumulativo alla lista
        collision_list.append(collision_count)
        
        screen = env.screen
        screen.fill((255, 255, 255))  # Pulisce lo schermo

        print(f"Episodio: {episode}")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        print(f"Collisioni totali: {collision_count}")
        print(f"---------------------")
        
        pygame.display.flip()
        epsilon = max(0.01, epsilon * 0.9995)  # Decay dell'epsilon (come nel file 5)

        episode_data.append((episode, steps, total_reward))

    if show_yes_no_dialog(env.screen, font, "Vuoi visualizzare i risultati?"):
        evaluate_agent(env, font)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare la Q-table?"):
        filename = f'q_table_{env.map_name}.npy'
        np.save(filename, env.q_values)

        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, f"Q-table {env.map_name} salvata con successo.", 20, 20, font, (0, 150, 0))
        pygame.display.flip()
        pygame.time.wait(1500)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare i grafici del training?"):
        show_training_charts(env.screen, font, episode_data, collision_list)

    return episode_data

def show_results(env, font):
    try:
        filename = f'q_table_{env.map_name}.npy'  # Nome file dipende dalla mappa
        q_table = np.load(filename)

        screen = env.screen
        screen.fill((255, 255, 255))

        if q_table.shape != env.q_values.shape:
            message = "Non è stato possibile caricare la Q-table"
            color = (255, 0, 0)  # Rosso
            wait_time = 2000
        else:
            env.q_values = q_table
            message = f"Q-table {env.map_name} caricata con successo."
            color = (0, 150, 0)  # Verde
            wait_time = 1500

        draw_text(screen, message, 0, screen.get_height() // 2 - 20, font, color, center=True)
        pygame.display.flip()
        pygame.time.wait(wait_time)

        if q_table.shape == env.q_values.shape:
            evaluate_agent(env, font)

    except FileNotFoundError:
        screen = env.screen
        screen.fill((255, 255, 255))

        message = f"Q-table {env.map_name} non trovata."
        draw_text(screen, message, 0, screen.get_height() // 2 - 20, font, (255, 0, 0), center=True)
        pygame.display.flip()
        pygame.time.wait(1500)

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
        pygame.time.wait(500)# Attende 500 ms tra ogni movimento
    
    if env.check_goal():
        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, "Obiettivo raggiunto!", 0, screen.get_height() // 2 - 20, font, (0, 150, 0), center=True)
        pygame.display.flip()
        pygame.time.wait(2000)

    else:
        screen = env.screen
        screen.fill((255, 255, 255))
        draw_text(screen, "L'agente ha perso.", 0, screen.get_height() // 2 - 20, font, (200, 0, 0), center=True)
        pygame.display.flip()
        pygame.time.wait(2000)


#Implementazione di una interfaccia grafica per il menù
def show_menu(screen, font):
    
    #Bottoni
    buttons = [
        {"text": "1. Allenare l'agente", "action": "train"},
        {"text": "2. Mostrare risultati", "action": "show"},
        {"text": "3. Scegli la mappa", "action": "select_map"},
        {"text": "4. Impostazioni", "action": "settings"},
        {"text": "5. Uscire", "action": "exit"}
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

#Funzione per poter stampare del testo sul schermo
def draw_text(screen, text, x, y, font, color=(0, 0, 0), center=False):

    text_surface = font.render(text, True, color)
    #se voglio qualcosa al centro devo gestirlo con un if
    if center:
        x = (screen.get_width() - text_surface.get_width()) // 2

    screen.blit(text_surface, (x, y))

#Funzione per poter chiedere all'utente, dal punto di vista grafico, se vuole o no vedere i risultati
def show_yes_no_dialog(screen, font, question):
    screen.fill((255, 255, 255))

    #(screen, text, x, y, font, color=(0, 0, 0)) devo calcolarmi la lunghezza del testo per poi centrarlo a dovere, non basta fare (screen.get_width()) // 2)-50
    draw_text(screen, question,0, 100, font, center=True)

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

available_maps = {
    "1": ("Città", Map1Environment),
    "2": ("Foresta", Map2Environment),
}

#Funzione che mi permette di scegliere la mappa
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
        draw_text(screen, "Seleziona una mappa:", 0, 50, font, center=True)

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
                            draw_text(screen, f"Hai selezionato: {map_name}", 0, 50, font, color=(0, 0, 0), center=True)

                            # Costruisco il nome del file immagine
                            file_name = map_name.lower().replace(" ", "_") + "_map.png"
                            preview_path = f"Progetto Tesi Privitera/assets/imgs/{file_name}"

                            try:
                                preview_img = pygame.image.load(preview_path)
                                preview_img = pygame.transform.smoothscale(preview_img, (1000, 520)) #smoothscale più lento di scale ma più qualitativo
                                img_x = (screen.get_width() - preview_img.get_width()) // 2
                                img_y = 150
                                screen.blit(preview_img, (img_x, img_y))
                            except Exception as e:
                                print(f"Errore caricamento immagine: {e}")

                            pygame.display.flip()
                            pygame.time.delay(2000)
                            selecting = False

    return selected_map_class #ritorna la classe della mappa selezionata

#Funzione per stampare il resoconto del training del agente
def show_training_results(screen, font, episode_data):
    import sys  # Assicurati che sys sia importato nel file

    scroll_y = 0
    scroll_speed = 20
    running = True
    clock = pygame.time.Clock()

    # Bottone stile menu
    buttons = [{"text": "Torna al menu", "action": "menu"}]
    button_rects = []
    y = screen.get_height() - 100  # Posizione verticale del bottone

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:  # Click sinistro
                    
                    for rect, action in button_rects:
                        
                        if rect.collidepoint(event.pos) and action == "menu":
                            running = False  # Torna al menu
                
                elif event.button == 4:  # Scroll up
                    scroll_y = min(scroll_y + scroll_speed, 0)
                
                elif event.button == 5:  # Scroll down
                    scroll_y -= scroll_speed

        screen.fill((255, 255, 255))

        # Intestazioni
        header = font.render(f"{'Episodio':<10}{'Steps':<10}{'Reward'}", True, (0, 0, 0))
        screen.blit(header, (20, 20 + scroll_y))

        pygame.draw.line(screen, (0, 0, 0), (20, 50 + scroll_y), (screen.get_width() - 20, 50 + scroll_y), 2)

        # Dati
        for idx, (episode, steps, reward) in enumerate(episode_data):
            text = font.render(f"{episode:<10}{steps:<10}{reward}", True, (0, 0, 0))
            screen.blit(text, (20, 60 + idx * 30 + scroll_y))

        # Bottone "Torna al menu"
        button_rects.clear()
        for button in buttons:
            rect = pygame.Rect(screen.get_width() // 2 - 150, y, 250, 50)
            pygame.draw.rect(screen, (0, 128, 255), rect)
            text_surface = font.render(button["text"], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 20, rect.y + 10))
            button_rects.append((rect, button["action"]))

        pygame.display.flip()
        clock.tick(60)

def show_settings(screen, font, current_error_prob):
    setting = True
    error_prob = current_error_prob
    
    while setting:
        screen.fill((255, 255, 255))
        draw_text(screen, "Impostazioni Pedoni", 0, 50, font, center=True)
        
        # Mostra il valore attuale
        draw_text(screen, f"Probabilità di errore: {error_prob:.2f}", 0, 150, font, center=True)
        
        # Pulsanti per aumentare/diminuire
        less_rect = pygame.Rect(screen.get_width() // 2 - 200, 200, 100, 50)
        more_rect = pygame.Rect(screen.get_width() // 2 + 100, 200, 100, 50)
        ok_rect = pygame.Rect(screen.get_width() // 2 - 100, 300, 200, 50)
        
        pygame.draw.rect(screen, (200, 0, 0), less_rect)  # Rosso
        pygame.draw.rect(screen, (0, 200, 0), more_rect)  # Verde
        pygame.draw.rect(screen, (0, 128, 255), ok_rect)  # Blu
        
        draw_text(screen, "-", less_rect.centerx - 10, less_rect.centery - 10, font, center=False)
        draw_text(screen, "+", more_rect.centerx - 10, more_rect.centery - 10, font, center=False)
        draw_text(screen, "Conferma", ok_rect.centerx - 50, ok_rect.centery - 10, font, center=False)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return error_prob
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if less_rect.collidepoint(pos):
                    error_prob = max(0.0, error_prob - 0.10)
                
                if more_rect.collidepoint(pos):
                    error_prob = min(1.0, error_prob + 0.10)
                
                if ok_rect.collidepoint(pos):
                    return error_prob
    
    return error_prob

def show_training_charts(screen, font, episode_data, cumulative_collisions):
    # Salva la modalità di visualizzazione corrente
    current_mode = pygame.display.get_surface().get_size()
    
    # Usa il backend Agg di matplotlib che non interferisce con la visualizzazione
    import matplotlib
    matplotlib.use('Agg')
    
    # Estrai i dati
    episodes = [data[0] for data in episode_data]
    steps = [data[1] for data in episode_data]
    rewards = [data[2] for data in episode_data]
    
    # Crea due grafici (uno sopra l'altro) con dimensioni 10x12
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Converti la lista dei rewards in una Series di pandas
    rewards_series = pd.Series(rewards)

    # Calcola la media mobile con una finestra di 50 per le ricompense
    rewards_rolling = rewards_series.rolling(window=50, min_periods=1).mean()

    # Grafico delle ricompense con media mobile
    ax1.plot(range(1, len(rewards) + 1), rewards, alpha=0.3, label='Ricompensa')
    ax1.plot(range(1, len(rewards) + 1), rewards_rolling, label='Media mobile')
    ax1.set_title('Ricompensa Totale per Episodio')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Ricompensa Totale')
    ax1.legend()
    ax1.grid(True)

    # Grafico delle collisioni cumulative
    ax2.plot(range(1, len(cumulative_collisions) + 1), cumulative_collisions)
    ax2.set_title('Collisioni Cumulative')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Numero di Collisioni')
    ax2.grid(True)

    # Aggiusta lo spazio tra i sottografici
    plt.tight_layout(pad=3.0)
    
    # Salva l'immagine direttamente come PNG
    result_path = f"training_results_new_250.png"
    plt.savefig(result_path)
    
    # Chiudi la figura per liberare memoria
    plt.close(fig)
    plt.close('all')  # Chiudi tutte le figure per sicurezza
    
    # Mostra il messaggio di conferma
    screen.fill((255, 255, 255))
    draw_text(screen, f"Grafico salvato come: {result_path}", 0, 100, font, (0, 150, 0), center=True)
    pygame.display.flip()
    pygame.time.wait(2000)  # Mostra il messaggio per 2 secondi
    
    return

def main():

    episode_data = []
    
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Centra la finestra
    pygame.init()
    
    screen = pygame.display.set_mode((1536, 800))
    pygame.display.update()

    pygame.display.set_caption("Find The Parking v.2")

    pygame.event.pump()# Forza aggiornamento della finestra
    font =  pygame.font.Font("Progetto Tesi Privitera/assets/PixeloidSansBold.ttf", 20)

    env = Map1Environment(48, 25, 32, screen)
    running = True
    action = None

    while running:
        
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
            episode_data = train_agent(env, font)
            show_training_results(env.screen, font, episode_data)

        elif action == "show":
            show_results(env,font)

        elif action == "select_map":

            selected_environment_class = select_map(screen, font)
            
            if selected_environment_class:
                env = selected_environment_class(48, 25, 32, screen)

        elif action == "exit":
            running = False

        elif action == "settings":
            #error_prob = 0.3 ci dice che i pedoni hanno il 30% di probabilità di sbagliare
            env.pedone_error_prob = show_settings(screen, font, env.pedone_error_prob)

    pygame.quit()

if __name__ == "__main__":
    
    try:
        main()
    
    except Exception as e:
        print(f"Errore imprevisto: {e}")

