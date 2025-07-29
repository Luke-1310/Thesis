import numpy as np #Impostata con alias np, essa è una libreria che serve per gestire array multidimensionali e funzioni matematiche
import pygame
import os
from datetime import datetime
import sys
from environments.map1_environment import Map1Environment
from environments.map2_environment import Map2Environment

import matplotlib.pyplot as plt       #Per disegnare i grafici
import pandas as pd                   #Per gestire e analizzare dati in modo ordinato

os.environ['SDL_VIDEO_CENTERED'] = '1' #Necessario perché, senza ulteriori precisazioni, la finestra viene creata in basso a destra

np.set_printoptions(precision=3, suppress=True, linewidth=200)

#CANCELLABILE (?)
def print_q_table(q_table):
     print("Q-Table:")
     print(q_table)

def train_agent(env, font):
    epsilon = 1
    discount_factor = 0.9
    learning_rate = 0.1
    num_episodes = getattr(env, 'num_episodes', 2000)  #Numero di episodi da eseguire, predefinito a 2000 se non specificato
    episode_data = []  #Lista che contiene (episodio, step, reward)
    collision_list = []  #Lista per tenere traccia delle collisioni cumulative
    collision_count = 0

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

            env.update_traffic_lights()  #Aggiorna lo stato dei semafori
            env.update_pedoni(env.pedoni)  #Aggiorna lo stato dei pedoni

            action_index = env.get_next_action(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.car_in_vision)
            env.is_car_in_vision()  #Aggiorna lo stato di car_in_vision
            is_valid = env.get_next_location(action_index)

            if is_valid:
                reward = env.reward_matrix[env.agent_position[1]][env.agent_position[0]]
            elif not env.check_loss():
                reward = -10
            else:
                reward = -100  #Questo reward viene usato nell'aggiornamento della Q-table in caso di perdita

            #Q-learning update
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            
            env.display(episode)
            pygame.time.wait(1)  #Breve pausa per gestire gli eventi

            total_reward += reward
            steps += 1

            if steps > 1000:  #Previene loop infiniti
                break

        if env.check_loss():
            collision_count += 1

        collision_list.append(collision_count)
        
        screen = env.screen
        screen.fill((255, 255, 255))  #Pulisce lo schermo per il prossimo episodio

        print(f"Episodio: {episode}")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        print(f"Collisioni totali: {collision_count}")
        print(f"---------------------")
        
        pygame.display.flip()
        epsilon = max(0.01, epsilon * 0.9995)  #Riduce epsilon per l'esplorazione

        episode_data.append((episode, steps, total_reward))

    if show_yes_no_dialog(env.screen, font, "Vuoi visualizzare i risultati?"):
        evaluate_agent(env, font)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare la Q-table?"):

        path_q_table = "Progetto Tesi Privitera/q_tables"

        filename = f'q_table_{env.map_name}.npy'  
        full_path_q_table = f"{path_q_table}/{filename}" 

        np.save(full_path_q_table, env.q_values)

        screen = env.screen
        screen.fill((255, 255, 255))

        draw_text(screen, f"Q-table {env.map_name} salvata con successo.", 0, screen.get_height() // 2 - 40, font, (0, 150, 0), center=True)
        draw_text(screen, f"Percorso: {full_path_q_table}", 0, screen.get_height() // 2, font, (0, 100, 0), center=True)
        
        pygame.display.flip()
        pygame.time.wait(1500)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare i grafici del training?"):
        show_training_charts(env.screen, font, episode_data, collision_list, env)

    return episode_data

def show_results(env, font):

    try:
        filename = f'q_table_{env.map_name}.npy'  #Il nome file dipende dalla mappa selezionata
        path_q_table = f"Progetto Tesi Privitera/q_tables/{filename}"
        q_table = np.load(path_q_table)

        screen = env.screen
        screen.fill((255, 255, 255))

        if q_table.shape != env.q_values.shape:
            message = "Non è stato possibile caricare la Q-table"
            color = (255, 0, 0)  #Rosso
            wait_time = 2000
        else:
            env.q_values = q_table
            message = f"Q-table {env.map_name} caricata con successo."
            color = (0, 150, 0)  #Verde
            wait_time = 1500 #Attende 1.5 secondi per mostrare il messaggio

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
        
        env.update_traffic_lights()      
        env.update_pedoni(env.pedoni)     
        env.is_car_in_vision()           
        
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


#Implementazione di una interfaccia grafica per il menu
def show_menu(screen, font):
    
    buttons = [
        {"text": "1. Allenare l'agente", "action": "train"},
        {"text": "2. Mostrare risultati", "action": "show"},
        {"text": "3. Scegli la mappa", "action": "select_map"},
        {"text": "4. Impostazioni", "action": "settings"},
        {"text": "5. Uscire", "action": "exit"}
    ]

    button_rects = []

    #Riempie l'intera finestra di bianco
    screen.fill((255, 255, 255))

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

    #Se il testo deve essere centrato, calcola la posizione x
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

    #Lista dei bottoni da visualizzare
    buttons = []
    for key, (map_name, _) in available_maps.items():
        buttons.append({"text": f"{map_name}", "action": key})
    buttons.append({"text": "Torna al menu", "action": "back"})

    while selecting:
        screen.fill((255, 255, 255))

        #Titolo centrato
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
                            return None  #Nessuna mappa selezionata, torna al menu
                        
                        elif action in available_maps:
                            map_name = available_maps[action][0]
                            selected_map_class = available_maps[action][1]
                            
                            #Feedback all'utente
                            screen.fill((255, 255, 255))
                            draw_text(screen, f"Hai selezionato: {map_name}", 0, 50, font, color=(0, 0, 0), center=True)

                            #Costruisco il nome del file immagine
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

    return selected_map_class #Ritorna la classe della mappa selezionata

#Funzione per stampare il resoconto del training del agente
def show_training_results(screen, font, episode_data):

    scroll_y = 0
    scroll_speed = 20
    running = True
    clock = pygame.time.Clock()

    #Bottone stile menu
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

        #Intestazioni
        header = font.render(f"{'Episodio':<10}{'Steps':<10}{'Reward'}", True, (0, 0, 0))
        screen.blit(header, (20, 20 + scroll_y))

        pygame.draw.line(screen, (0, 0, 0), (20, 50 + scroll_y), (screen.get_width() - 20, 50 + scroll_y), 2)

        #Dati
        for idx, (episode, steps, reward) in enumerate(episode_data):
            text = font.render(f"{episode:<10}{steps:<10}{reward}", True, (0, 0, 0))
            screen.blit(text, (20, 60 + idx * 30 + scroll_y))

        #Bottone "Torna al menu"
        button_rects.clear()
        for button in buttons:
            rect = pygame.Rect(screen.get_width() // 2 - 150, y, 250, 50)
            pygame.draw.rect(screen, (0, 128, 255), rect)
            text_surface = font.render(button["text"], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 20, rect.y + 10))
            button_rects.append((rect, button["action"]))

        pygame.display.flip()
        clock.tick(60)

def show_settings(screen, font, env):
    
    setting = True

    # Valori attuali dall'ambiente
    num_pedoni = env.num_pedoni
    error_prob_pedoni = env.pedone_error_prob
    prob_change_auto = env.route_change_probability
    num_episodi = getattr(env, 'num_episodes', 2000) #Valore predefinito se non esiste
    
    while setting:
        
        screen.fill((255, 255, 255))
        
        draw_text(screen, "Impostazioni Ambiente", 0, 30, font, (0, 0, 0), center=True)
        
        #Sezione per i pedoni
        y_start = 100
        draw_text(screen, f"Numero Pedoni: {num_pedoni}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni pedoni
        pedoni_less_rect = pygame.Rect(screen.get_width() // 2 - 250, y_start + 40, 80, 40)
        pedoni_more_rect = pygame.Rect(screen.get_width() // 2 + 170, y_start + 40, 80, 40)
        
        pygame.draw.rect(screen, (200, 50, 50), pedoni_less_rect)    # Rosso
        pygame.draw.rect(screen, (50, 200, 50), pedoni_more_rect)    # Verde
        
        draw_text(screen, "-", pedoni_less_rect.centerx - 8, pedoni_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", pedoni_more_rect.centerx - 8, pedoni_more_rect.centery - 10, font, (255, 255, 255))
        
        #Sezione per l'errore dei pedoni
        y_start += 120
        draw_text(screen, f"Prob. Errore Pedoni: {error_prob_pedoni:.0%}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni errore pedoni
        err_ped_less_rect = pygame.Rect(screen.get_width() // 2 - 250, y_start + 40, 80, 40)
        err_ped_more_rect = pygame.Rect(screen.get_width() // 2 + 170, y_start + 40, 80, 40)
        
        pygame.draw.rect(screen, (200, 50, 50), err_ped_less_rect)   # Rosso
        pygame.draw.rect(screen, (50, 200, 50), err_ped_more_rect)   # Verde
        
        draw_text(screen, "-", err_ped_less_rect.centerx - 8, err_ped_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", err_ped_more_rect.centerx - 8, err_ped_more_rect.centery - 10, font, (255, 255, 255))
        
        #Sezione per il cambio percorso delle auto
        y_start += 120
        draw_text(screen, f"Prob. Cambio Percorso Auto: {prob_change_auto:.0%}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni cambio percorso auto
        auto_less_rect = pygame.Rect(screen.get_width() // 2 - 250, y_start + 40, 80, 40)
        auto_more_rect = pygame.Rect(screen.get_width() // 2 + 170, y_start + 40, 80, 40)
        
        pygame.draw.rect(screen, (200, 50, 50), auto_less_rect)      # Rosso
        pygame.draw.rect(screen, (50, 200, 50), auto_more_rect)      # Verde
        
        draw_text(screen, "-", auto_less_rect.centerx - 8, auto_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", auto_more_rect.centerx - 8, auto_more_rect.centery - 10, font, (255, 255, 255))

        #Sezione per il numero degli episodi
        y_start += 90
        draw_text(screen, f"Numero Episodi: {num_episodi}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni numero episodi
        episodi_less_rect = pygame.Rect(screen.get_width() // 2 - 250, y_start + 30, 60, 35)
        episodi_more_rect = pygame.Rect(screen.get_width() // 2 + 190, y_start + 30, 60, 35)
        
        pygame.draw.rect(screen, (200, 50, 50), episodi_less_rect)
        pygame.draw.rect(screen, (50, 200, 50), episodi_more_rect)
        
        draw_text(screen, "-", episodi_less_rect.centerx - 6, episodi_less_rect.centery - 8, font, (255, 255, 255))
        draw_text(screen, "+", episodi_more_rect.centerx - 6, episodi_more_rect.centery - 8, font, (255, 255, 255))
        
        #Sezione finale
        y_final = y_start + 120
        
        #Bottone Conferma
        confirm_rect = pygame.Rect(screen.get_width() // 2 - 220, y_final, 180, 50)
        pygame.draw.rect(screen, (0, 150, 0), confirm_rect)          # Verde
        draw_text(screen, "Conferma", confirm_rect.centerx - 45, confirm_rect.centery - 10, font, (255, 255, 255))
        
        #Bottone Annulla
        cancel_rect = pygame.Rect(screen.get_width() // 2 + 40, y_final, 180, 50)
        pygame.draw.rect(screen, (150, 0, 0), cancel_rect)           # Rosso
        draw_text(screen, "Annulla", cancel_rect.centerx - 40, cancel_rect.centery - 10, font, (255, 255, 255))
        
        pygame.display.flip()
        
        #Gestione degli eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return  #Esce senza salvare
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                #Controllo numero dei pedoni (0-10)
                if pedoni_less_rect.collidepoint(pos):
                    num_pedoni = max(0, num_pedoni - 1)
                
                if pedoni_more_rect.collidepoint(pos):
                    num_pedoni = min(10, num_pedoni + 1)
                
                #Controllo errore dei pedoni (0%-100%, step del 10%)
                if err_ped_less_rect.collidepoint(pos):
                    error_prob_pedoni = max(0.0, error_prob_pedoni - 0.10)
                
                if err_ped_more_rect.collidepoint(pos):
                    error_prob_pedoni = min(1.0, error_prob_pedoni + 0.10)
                
                #Controllo cambio percorso auto (0%-100%, step del 10%)
                if auto_less_rect.collidepoint(pos):
                    prob_change_auto = max(0.0, prob_change_auto - 0.10)
                
                if auto_more_rect.collidepoint(pos):
                    prob_change_auto = min(1.0, prob_change_auto + 0.10)

                #Controllo numero episodi (1-3000, step di 50)
                if episodi_less_rect.collidepoint(pos):
                    num_episodi = max(1, num_episodi - 50)
                
                if episodi_more_rect.collidepoint(pos):
                    num_episodi = min(3000, num_episodi + 50)
                
                #Bottone per la conferma
                if confirm_rect.collidepoint(pos):
                    # Applica le modifiche all'ambiente
                    env.num_pedoni = num_pedoni
                    env.pedone_error_prob = error_prob_pedoni
                    env.route_change_probability = prob_change_auto
                    env.num_episodes = num_episodi
                    
                    #Messaggio di conferma delle modifiche
                    screen.fill((255, 255, 255))
                    draw_text(screen, "Impostazioni salvate con successo!", 0, screen.get_height() // 2 - 20, font, (0, 150, 0), center=True)
                    pygame.display.flip()
                    pygame.time.wait(1500)
                    
                    return
                
                #Bottone per annullare le modifiche
                if cancel_rect.collidepoint(pos):
                    # Messaggio di annullamento
                    screen.fill((255, 255, 255))
                    draw_text(screen, "Modifiche annullate.", 0, screen.get_height() // 2 - 20, font, (200, 0, 0), center=True)
                    pygame.display.flip()
                    pygame.time.wait(1000)
                    
                    return

def show_training_charts(screen, font, episode_data, cumulative_collisions, env):
    
    #Salva la modalità di visualizzazione corrente
    current_mode = pygame.display.get_surface().get_size()
    
    #Usa il backend Agg di matplotlib che non interferisce con la visualizzazione
    import matplotlib
    matplotlib.use('Agg')
    
    #Estrai i dati
    # episodes = [data[0] for data in episode_data]
    # steps = [data[1] for data in episode_data]
    rewards = [data[2] for data in episode_data]

    # Raccolta dati da aggiungere al grafico
    map_name = env.map_name
    num_pedoni = len(env.pedoni)
    prob_change_percorso = env.route_change_probability  # ← CORRETTO
    error_prob_pedoni = env.pedone_error_prob
    
    # Calcola statistiche
    num_episodes = len(episode_data)
    total_collisions = cumulative_collisions[-1] if cumulative_collisions else 0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    max_reward = max(rewards) if rewards else 0
    
    # Conta goal raggiunti (reward molto alto)
    goals_reached = sum(1 for _, _, reward in episode_data if reward > 1000)
    success_rate = (goals_reached / num_episodes * 100) if num_episodes > 0 else 0

    # Crea due grafici (uno sopra l'altro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Converti la lista dei rewards in una Series di pandas
    rewards_series = pd.Series(rewards)
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

    #Informazioni per i grafici
    #Prima riga di informazioni
    info_line1 = f"Mappa: {map_name} | Pedoni: {num_pedoni} | Prob. errore pedoni: {error_prob_pedoni:.1%} | Prob. cambio percorso auto: {prob_change_percorso:.1%}"
    
    #Seconda riga di informazioni
    info_line2 = f"Episodi: {num_episodes} | Goal raggiunti: {goals_reached} ({success_rate:.1f}%) | Collisioni: {total_collisions} | Reward medio: {avg_reward:.1f} | Reward max: {max_reward:.1f}"

    #Aggiungi le informazioni sotto i grafici
    fig.text(0.5, 0.06, info_line1, fontsize=10, ha='center', fontweight='bold')
    fig.text(0.5, 0.02, info_line2, fontsize=10, ha='center', fontweight='bold')

    #Aggiusta il layout per lasciare spazio sotto
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Lascia spazio sotto per le informazioni

    results_folder = "Progetto Tesi Privitera/training_charts"  #Nome cartella per i risultati
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    #Nome file più descrittivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_{map_name}_{num_episodes}ep_{timestamp}.png"
    
    result_path = f"{results_folder}/{filename}"  # ← Versione semplice
    
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    
    #Chiudi la figura per liberare memoria
    plt.close(fig)
    plt.close('all')
    
    #Mostra il messaggio di conferma
    screen.fill((255, 255, 255))
    draw_text(screen, f"Grafico salvato come: {result_path}", 0, 100, font, (0, 150, 0), center=True)
    pygame.display.flip()
    pygame.time.wait(2000)
    
    return

def main():

    episode_data = []
    
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Centra la finestra
    pygame.init()
    
    screen = pygame.display.set_mode((1536, 800))
    pygame.display.update()
    pygame.display.set_caption("Find The Parking v.2")
    pygame.event.pump()#Forza aggiornamento della finestra
    font =  pygame.font.Font("Progetto Tesi Privitera/assets/PixeloidSansBold.ttf", 20)

    env = Map1Environment(48, 25, 32, screen,                    
        num_pedoni=0,           
        pedone_error_prob=0.0,              
        route_change_probability=0.2,  
        num_episodi = 2000      
    )
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
                
                current_num_pedoni = env.num_pedoni if 'env' in locals() else 2
                current_error_prob = env.pedone_error_prob if 'env' in locals() else 0.0
                current_route_prob = env.route_change_probability if 'env' in locals() else 0.2
                current_num_episodi = getattr(env, 'num_episodi', 2000) if 'env' in locals() else 2000 
                
                env = selected_environment_class(
                    48, 25, 32, screen,
                    #Le successive quattro righe mantengono le impostazioni correnti
                    num_pedoni=current_num_pedoni,           
                    pedone_error_prob=current_error_prob,    
                    route_change_probability=current_route_prob,  
                    num_episodi=current_num_episodi         
                )

        elif action == "exit":
            running = False

        elif action == "settings":
            show_settings(screen, font, env)

    pygame.quit()

if __name__ == "__main__":
    
    try:
        main()
    
    except Exception as e:
        print(f"Errore imprevisto: {e}")

