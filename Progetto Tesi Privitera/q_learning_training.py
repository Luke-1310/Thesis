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

def train_agent(env, font):

    epsilon = 1 #Esplorazione iniziale
    discount_factor = 0.9 #Fattore di sconto, ovvero quanto ci si fida del reward futuro
    learning_rate = 0.1 #Tasso di apprendimento
    rho = 0.9999 #Fattore di decadimento per epsilon
    num_episodes = getattr(env, 'num_episodes', 10000)
    episode_data = []  #Lista che contiene (episodio, step, reward)
    collision_list = []  #Lista per tenere traccia delle collisioni cumulative
    collision_count = 0
    goal_reached_count = 0
    goal_already_visited_count = 0

    for episode in range(num_episodes):
        env.reset_game()
        steps = 0
        total_steps_reward = 0.0
        total_right_penalty = 0.0
        total_reward = 0.0
        
        #Contatori eventi semaforo
        red_light_crossings = 0
        green_light_crossings = 0
        waiting_at_red_light = 0

        while not (env.check_loss() or env.check_goal()):
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            pygame.event.pump() #.pump serve per aggiornare lo stato degli eventi di pygame

            if hasattr(env, "update_traffic_lights"):
                env.update_traffic_lights()  #Aggiorna lo stato dei semafori

            if hasattr(env, "update_car_position"):
                env.update_car_position() #Aggiorna le auto nemiche
            
            if hasattr(env, "pedoni"):
                env.update_pedoni(env.pedoni) #Aggiorna i pedoni
            
            reward = 0.0

            if env.realistic_mode:
            
                #Stato di visione PRIMA della scelta azione con semafori
                old_cars_visible, old_pedestrians_visible,  old_traffic_light = env.get_vision_state()
                action_index = env.get_next_action(epsilon, old_traffic_light)
                old_position = env.agent_position[:]

                is_valid = env.get_next_location(action_index)

                current_position = tuple(env.agent_position)
                old_position_tuple = tuple(old_position)

                if is_valid:    

                    reward += env.reward_matrix[env.agent_position[1]][env.agent_position[0]]

                    if env.reward_matrix[env.agent_position[1]][env.agent_position[0]] == -1:
                        total_steps_reward += -1

                    if hasattr(env, 'intermediate_goals') and current_position in env.intermediate_goals:
                        
                        if current_position in env.visited_goals:
                            reward -= env.reward_matrix[env.agent_position[1]][env.agent_position[0]]
                            goal_already_visited_count += 1
                        else:
                            env.visited_goals.add(current_position)  
                            goal_reached_count += 1

                    right_penalty = env.right_edge_penalty()
                    reward += right_penalty
                    total_right_penalty += right_penalty

                    #Logica semafori
                    if current_position in env.traffic_lights:
                        
                        if old_position_tuple in env.safe_zones:   #Se era in una safe zone, non penalizza
                            pass
                        else:
                            is_entering_intersection = (old_position_tuple not in env.traffic_lights and 
                                                      current_position in env.traffic_lights)   #Verifica se è un PRIMO ingresso nell'incrocio e se passa col rosso lo penalizza
                            
                            if is_entering_intersection and env.traffic_lights[current_position] == 'red':
                                reward += -1000.0
                                red_light_crossings += 1  

                            elif is_entering_intersection and env.traffic_lights[current_position] == 'green':
                                reward += 80.0  
                                green_light_crossings += 1  
                    else:
                        #Premia l'agente per fermarsi prima di un semaforo rosso
                        if old_position_tuple == current_position and action_index == 4:  #Se ha scelto di stare fermo (azione 'stay')
                            
                            if old_position_tuple in env.traffic_light_approach_zones:
                                traffic_light_pos = env.traffic_light_approach_zones[old_position_tuple]
                                
                                if traffic_light_pos in env.traffic_lights and env.traffic_lights[traffic_light_pos] == 'red':
                                    if action_index == 4:  #Se ha scelto di stare fermo
                                        reward += 50.0  
                                        waiting_at_red_light += 1  
                        
                elif not env.check_loss():
                    reward = -10
                else:
                    reward = -100  #Questo reward viene usato nell'aggiornamento della Q-table in caso di perdita

                #Q-learning update
                old_q_value = env.q_values[old_position[1], old_position[0], old_cars_visible, old_pedestrians_visible, old_traffic_light, action_index]
                new_cars_visible, new_pedestrians_visible, new_traffic_light = env.get_vision_state()

                temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], new_cars_visible, new_pedestrians_visible, new_traffic_light])) - old_q_value
                
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                env.q_values[old_position[1], old_position[0], old_cars_visible, old_pedestrians_visible, old_traffic_light, action_index] = new_q_value
            
            else:

                #Stato di visione PRIMA della scelta azione senza semafori
                old_cars_visible, old_pedestrians_visible = env.get_vision_state()
                action_index = env.get_next_action(epsilon)
                old_position = env.agent_position[:]

                is_valid = env.get_next_location(action_index)

                current_position = tuple(env.agent_position)
                old_position_tuple = tuple(old_position)

                if is_valid:

                    reward += env.reward_matrix[env.agent_position[1]][env.agent_position[0]]

                    if env.reward_matrix[env.agent_position[1]][env.agent_position[0]] == -1:
                        total_steps_reward += -1

                    if hasattr(env, 'intermediate_goals') and current_position in env.intermediate_goals: 
                        reward -= env.reward_matrix[env.agent_position[1]][env.agent_position[0]]

                elif not env.check_loss():
                    reward = -10
                else:
                    reward = -100  #Questo reward viene usato nell'aggiornamento della Q-table in caso di perdita

                #Q-learning update
                old_q_value = env.q_values[old_position[1], old_position[0], old_cars_visible, old_pedestrians_visible, action_index]
                new_cars_visible, new_pedestrians_visible = env.get_vision_state()
                
                temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], new_cars_visible, new_pedestrians_visible])) - old_q_value

                new_q_value = old_q_value + (learning_rate * temporal_difference)
                env.q_values[old_position[1], old_position[0], old_cars_visible, old_pedestrians_visible, action_index] = new_q_value

            env.display(episode)
            pygame.time.wait(1)  #Breve pausa per gestire gli eventi

            total_reward += reward
            total_reward = round(total_reward, 2) #approssima a 2 decimali
            steps += 1

            if steps > 1500:  #Previene loop infiniti
                print("Episodio terminato per superamento step massimi.")
                break
        
        #Conta le collisioni solo se l'agente non ha raggiunto l'obiettivo
        fail = steps > 1500 or env.check_loss()

        if fail:  # L'agente non ha raggiunto il traguardo
            collision_count += 1
            esito_episodio = "Collisione"
        else:
            esito_episodio = "Successo"
        
        screen = env.screen
        screen.fill((255, 255, 255))  #Pulisce lo schermo per il prossimo episodio

        print(f"Episodio: {episode}")
        print(f"Steps: {steps}")
        print(f"Reward step totali: {total_steps_reward:.2f}")

        if env.realistic_mode:
            print(f"Right Edge Penalty totale: {total_right_penalty:.2f}")
            print(f"Obiettivi intermedi raggiunti: {goal_reached_count}/{len(env.intermediate_goals)}")
            print(f"Obiettivi intermedi già visitati: {goal_already_visited_count}")
            
            #Stampa riassuntiva degli eventi semaforo
            if red_light_crossings > 0:
                print(f"Semaforo rosso attraversato: {red_light_crossings} volte")
            if green_light_crossings > 0:
                print(f"Semaforo verde attraversato: {green_light_crossings} volte")
            if waiting_at_red_light > 0:
                print(f"In attesa al semaforo rosso: {waiting_at_red_light} volte")

        print(f"Total Reward: {total_reward:.2f}")
        print(f"Collisioni totali: {collision_count}")
        print(f"Esito episodio: {esito_episodio}")
        print(f"---------------------")
        
        pygame.display.flip()
        epsilon = max(0.01, epsilon * rho)  #Decadimento di epsilon, mai sotto 0.01

        episode_data.append((episode, steps, total_reward))
        collision_list.append(collision_count)

        goal_reached_count = 0
        goal_already_visited_count = 0

    if show_yes_no_dialog(env.screen, font, "Vuoi visualizzare i risultati?"):
        evaluate_agent(env, font)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare la Q-table?"):

        path_q_table = "Progetto Tesi Privitera/q_tables"

        #creo il timestamp per rendere univoca la q-table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        #aggiungo un suffisso se la modalità è realistica
        if getattr(env, 'realistic_mode', False):
    
            filename = f'q_table_{env.map_name}_{timestamp}_realistic.npy'
        
        else:    
            filename = f'q_table_{env.map_name}_{timestamp}.npy'


        full_path_q_table = f"{path_q_table}/{filename}" 

        np.save(full_path_q_table, env.q_values)

        screen = env.screen
        screen.fill((255, 255, 255))

        draw_text(screen, f"Q-table salvata con successo!", 0, screen.get_height() // 2 - 40, font, (0, 150, 0), center=True)
        draw_text(screen, f"Nome: {filename}", 0, screen.get_height() // 2, font, (0, 100, 0), center=True)
        
        pygame.display.flip()
        pygame.time.wait(1500)

    if show_yes_no_dialog(env.screen, font, "Vuoi salvare i grafici del training?"):
        show_training_charts(env.screen, font, episode_data, collision_list, env)

    return episode_data

def show_results(env, font):

    #Carichiamo le Q-table disponibili
    qtables_info = []
    qtables_dir = "Progetto Tesi Privitera/q_tables"

    try:
        files = os.listdir(qtables_dir)

        #è necessario un filtro in base alla modalità

        if getattr(env, 'realistic_mode', False):

            #Modalità realistica: cerca Q-table con suffisso "_realistic"
            qtable_files = [f for f in files if f.startswith(f"q_table_{env.map_name}_") and f.endswith('_realistic.npy')]

        else:
            #Modalità normale: cerca Q-table senza suffisso "_realistic"
            qtable_files = [f for f in files if f.startswith(f"q_table_{env.map_name}_") and f.endswith('.npy') and '_realistic' not in f]


        for qtable_file in qtable_files:
            
            #Estraggo timestamp dal nome file per poi mostrarlo a schermo
            parts = qtable_file.replace('.npy', '').split('_')
            
            if len(parts) >= 4:
                date_part = parts[3]    #Il terzo elemento è la data
                time_part = parts[4]     #Il quarto elemento è l'ora
                
                #Formatta data/ora leggibile
                try:
                    formatted_date = f"{date_part[6:8]}/{date_part[4:6]}/{date_part[:4]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    mode_suffix = " (Realistica)" if "_realistic" in qtable_file else " (Normale)"
                    display_name = f"{formatted_date} alle {formatted_time}{mode_suffix}"

                except:
                    display_name = qtable_file.replace('.npy', '')
            
            else:
                display_name = qtable_file.replace('.npy', '')
            
            qtables_info.append({
                'filename': qtable_file,
                'filepath': os.path.join(qtables_dir, qtable_file),
                'display_name': display_name
            })
        
        qtables_info.sort(key=lambda x: x['filename'], reverse=True) #Ordina le table dalla più vecchia alla più recente
               
    except Exception as e:
        print(f"Errore caricamento Q-table: {e}")  

    #Se nessuna Q-table è stata trovata
    if not qtables_info:
        screen = env.screen
        screen.fill((255, 255, 255))
        mode_text = "realistica" if getattr(env, 'realistic_mode', False) else "semplificata"
        draw_text(screen, f"Nessuna Q-table {mode_text} trovata per {env.map_name}", 0, screen.get_height() // 2, font, (255, 0, 0), center=True)
        draw_text(screen, "Esegui prima un training.", 0, screen.get_height() // 2 + 40, font, (100, 100, 100), center=True)
        pygame.display.flip()
        pygame.time.wait(2000)
        return
    
    #Menu selezione
    selecting = True
    selected_index = 0
    
    while selecting:
        screen = env.screen
        screen.fill((245, 245, 245))
        
        # Titolo
        draw_text(screen, f"Seleziona Q-table per {env.map_name}", 0, 50, font, (0, 0, 0), center=True)
        draw_text(screen, f"Trovate {len(qtables_info)} Q-table", 0, 80, font, (100, 100, 100), center=True)
        
        # Lista Q-table (max 8 visibili)
        y_start = 140
        max_visible = 8
        visible_items = qtables_info[:max_visible]
        
        for i, qtable_info in enumerate(visible_items):
            y_pos = y_start + i * 60
            
            # Background per item selezionato
            item_rect = pygame.Rect(200, y_pos - 5, screen.get_width() - 400, 50)
            if i == selected_index:
                pygame.draw.rect(screen, (200, 230, 255), item_rect, border_radius=5)
                pygame.draw.rect(screen, (0, 120, 255), item_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(screen, (255, 255, 255), item_rect, border_radius=5)
                pygame.draw.rect(screen, (200, 200, 200), item_rect, 1, border_radius=5)
            
            # Info Q-table: solo data/ora pulita
            clean_name = qtable_info['display_name']
            draw_text(screen, clean_name, item_rect.centerx, y_pos + 15, font, (0, 0, 0), center=True)
        
        # Istruzioni
        draw_text(screen, "↑↓: Naviga | INVIO: Seleziona | ESC: Torna al menu", 
                 0, screen.get_height() - 40, font, (0, 0, 0), center=True)
        
        pygame.display.flip()
        
        # Gestione eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                
                elif event.key == pygame.K_RETURN:
                    # Carica Q-table selezionata
                    selected_qtable = qtables_info[selected_index]
                    
                    try:
                        # Carica Q-table
                        q_table = np.load(selected_qtable['filepath'])
                        
                        # Verifica compatibilità
                        if q_table.shape != env.q_values.shape:
                            screen = env.screen
                            screen.fill((255, 255, 255))
                            draw_text(screen, f"Q-table incompatibile!", 0, screen.get_height() // 2 - 20, font, (255, 0, 0), center=True)
                            draw_text(screen, f"Forma: {q_table.shape} vs {env.q_values.shape}", 0, screen.get_height() // 2 + 20, font, (255, 0, 0), center=True)
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            continue  # Torna al menu selezione
                        
                        # Carica Q-table
                        env.q_values = q_table
                        
                        # Messaggio caricamento
                        screen = env.screen
                        screen.fill((255, 255, 255))
                        draw_text(screen, f"Q-table caricata!", 0, screen.get_height() // 2 - 20, font, (0, 150, 0), center=True)
                        draw_text(screen, f"{selected_qtable['display_name']}", 0, screen.get_height() // 2 + 20, font, (0, 100, 0), center=True)
                        pygame.display.flip()
                        pygame.time.wait(1500)
                        
                        # Avvia evaluation e esci
                        evaluate_agent(env, font)
                        return
                        
                    except Exception as e:
                        screen = env.screen
                        screen.fill((255, 255, 255))
                        draw_text(screen, f"Errore caricamento: {str(e)}", 0, screen.get_height() // 2, font, (255, 0, 0), center=True)
                        pygame.display.flip()
                        pygame.time.wait(3000)
                        continue  # Torna al menu selezione
                
                elif event.key == pygame.K_UP:
                    selected_index = max(0, selected_index - 1)
                
                elif event.key == pygame.K_DOWN:
                    selected_index = min(len(visible_items) - 1, selected_index + 1)

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
        
        #Aggiorna le auto nemiche
        if hasattr(env, "update_car_position"):
            env.update_car_position()
        
        #env.update_pedoni(env.pedoni)

        if hasattr(env, "pedoni"):
            env.update_pedoni(env.pedoni)

        if env.realistic_mode:
            cars_visible, pedestrians_visible, traffic_light = env.get_vision_state()
            action_index = np.argmax(
                env.q_values[env.agent_position[1], env.agent_position[0], cars_visible, pedestrians_visible, traffic_light]
            )

        else:
            cars_visible, pedestrians_visible = env.get_vision_state()
            action_index = np.argmax(
                env.q_values[env.agent_position[1], env.agent_position[0], cars_visible, pedestrians_visible]
            )

        env.get_next_location(action_index)
        path.append(env.agent_position[:])
        env.display(path=path)

        pygame.time.wait(500)
    
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

def show_menu(screen, font):
    
    buttons = [
        {"text": "1. Avvio simulazione", "action": "train"},
        {"text": "2. Visualizza risultati", "action": "show"},
        {"text": "3. Seleziona mappa", "action": "select_map"},
        {"text": "4. Opzioni", "action": "settings"},
        {"text": "5. Esci", "action": "exit"}
    ]

    button_rects = []

    #Riempie l'intera finestra di bianco
    screen.fill((255, 255, 255))

    title = font.render("Menu Principale", True, (0, 0, 0))
    screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 50))

    y = 150

    for button in buttons:
        rect = pygame.Rect(screen.get_width() // 2 - 200, y, 380, 50)
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

#Funzione per poter chiedere all'utente di fare una scelta sì/no
def show_yes_no_dialog(screen, font, question):
    screen.fill((255, 255, 255))

    #(screen, text, x, y, font, color=(0, 0, 0)) devo calcolarmi la lunghezza del testo per poi centrarlo a dovere, non basta fare (screen.get_width()) // 2)-50
    draw_text(screen, question,0, 100, font, center=True)

    button_width = 150
    button_height = 50
    spacing = 40  # spazio tra i due bottoni

    #Calcola la posizione centrale dei due bottoni insieme
    total_width = button_width * 2 + spacing
    start_x = (screen.get_width() - total_width) // 2
    y = 200

    yes_rect = pygame.Rect(start_x, y, button_width, button_height)
    no_rect = pygame.Rect(start_x + button_width + spacing, y, button_width, button_height)

    pygame.draw.rect(screen, (0, 200, 0), yes_rect)
    pygame.draw.rect(screen, (200, 0, 0), no_rect)

    yes_text = font.render("Sì", True, (255, 255, 255))
    no_text = font.render("No", True, (255, 255, 255))

    #Centra il testo all'interno dei bottoni
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
            rect = pygame.Rect(screen.get_width() // 2 - 200, y, 400, 50)
            pygame.draw.rect(screen, (0, 128, 255), rect)
            text_surface = font.render(button["text"], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 20, rect.y + 10))
            button_rects.append((rect, button["action"]))

        pygame.display.flip()
        clock.tick(60)

def show_settings(screen, font, env):
    
    setting = True

    #Valori attuali dall'ambiente
    num_pedoni = env.num_pedoni
    error_prob_pedoni = env.pedone_error_prob
    prob_change_auto = env.route_change_probability
    num_episodi = getattr(env, 'num_episodes', 10000) #Valore predefinito se non esiste
    
    editing_episodi = False
    episodi_input = str(num_episodi)

    while setting:
        
        screen.fill((255, 255, 255))
        draw_text(screen, "Impostazioni Ambiente", 0, 30, font, (0, 0, 0), center=True)
        
        #Parametri per creare un layout uniforme
        button_width = 60
        button_height = 40
        center_x = screen.get_width() // 2
        button_spacing = 300  #Distanza tra bottoni - e +

        #Sezione per i pedoni
        y_start = 100
        draw_text(screen, f"Numero Pedoni: {num_pedoni}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni pedoni
        pedoni_less_rect = pygame.Rect(center_x - button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        pedoni_more_rect = pygame.Rect(center_x + button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        
        pygame.draw.rect(screen, (200, 50, 50), pedoni_less_rect)    #Rosso
        pygame.draw.rect(screen, (50, 200, 50), pedoni_more_rect)    #Verde
        
        draw_text(screen, "-", pedoni_less_rect.centerx - 8, pedoni_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", pedoni_more_rect.centerx - 8, pedoni_more_rect.centery - 10, font, (255, 255, 255))
        
        #Sezione per l'errore dei pedoni
        y_start += 120
        draw_text(screen, f"Prob. Errore Pedoni: {error_prob_pedoni:.0%}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni errore pedoni
        err_ped_less_rect = pygame.Rect(center_x - button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        err_ped_more_rect = pygame.Rect(center_x + button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        
        pygame.draw.rect(screen, (200, 50, 50), err_ped_less_rect)   
        pygame.draw.rect(screen, (50, 200, 50), err_ped_more_rect)   
        
        draw_text(screen, "-", err_ped_less_rect.centerx - 8, err_ped_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", err_ped_more_rect.centerx - 8, err_ped_more_rect.centery - 10, font, (255, 255, 255))
        
        #Sezione per il cambio percorso delle auto
        y_start += 120
        draw_text(screen, f"Prob. Cambio Percorso Auto: {prob_change_auto:.0%}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni cambio percorso auto
        auto_less_rect = pygame.Rect(center_x - button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        auto_more_rect = pygame.Rect(center_x + button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)

        pygame.draw.rect(screen, (200, 50, 50), auto_less_rect)     
        pygame.draw.rect(screen, (50, 200, 50), auto_more_rect)      
        
        draw_text(screen, "-", auto_less_rect.centerx - 8, auto_less_rect.centery - 10, font, (255, 255, 255))
        draw_text(screen, "+", auto_more_rect.centerx - 8, auto_more_rect.centery - 10, font, (255, 255, 255))

        #Sezione per il numero degli episodi
        y_start += 90
        draw_text(screen, f"Numero Episodi: {num_episodi}", 0, y_start, font, (0, 0, 0), center=True)
        
        #Bottoni numero episodi
        episodi_less_rect = pygame.Rect(center_x - button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        episodi_more_rect = pygame.Rect(center_x + button_spacing//2 - button_width//2, y_start + 40, button_width, button_height)
        
        input_width = 160
        input_height = 50
        episodi_input_rect = pygame.Rect(center_x - input_width//2, y_start + 40, input_width, input_height)

        pygame.draw.rect(screen, (200, 50, 50), episodi_less_rect)
        pygame.draw.rect(screen, (50, 200, 50), episodi_more_rect)
        
        draw_text(screen, "-", episodi_less_rect.centerx - 6, episodi_less_rect.centery - 8, font, (255, 255, 255))
        draw_text(screen, "+", episodi_more_rect.centerx - 6, episodi_more_rect.centery - 8, font, (255, 255, 255))
        
        # Rendering del box di input con cursore lampeggiante
        border_color = (0, 150, 0) if editing_episodi else (120, 120, 120)
        bg_color = (255, 255, 255) if editing_episodi else (245, 245, 245)
        
        pygame.draw.rect(screen, bg_color, episodi_input_rect)
        pygame.draw.rect(screen, border_color, episodi_input_rect, 3)

        #Testo da visualizzare con cursore
        shown_value = episodi_input if editing_episodi else str(num_episodi)
        
        #Cursore lampeggiante quando in editing
        if editing_episodi:
            
            #Cursore lampeggia ogni 500ms
            cursor_visible = (pygame.time.get_ticks() // 500) % 2
            if cursor_visible:
                shown_value += "|"
        
        #Colore testo diverso quando editing
        text_color = (0, 0, 150) if editing_episodi else (0, 0, 0)
        value_surface = font.render(shown_value, True, text_color)
        
        #Placeholder se vuoto
        if editing_episodi and not episodi_input:
            placeholder_surface = font.render("Digita qui", True, (150, 150, 150))
            screen.blit(placeholder_surface, (episodi_input_rect.centerx - placeholder_surface.get_width() // 2,episodi_input_rect.centery - placeholder_surface.get_height() // 2))
        else:
            screen.blit(value_surface, (episodi_input_rect.centerx - value_surface.get_width() // 2, episodi_input_rect.centery - value_surface.get_height() // 2))
        
        #Istruzioni sotto il campo
        if editing_episodi:
            help_text = "INVIO: Conferma | ESC: Annulla | BACKSPACE: Cancella"
            help_surface = pygame.font.Font(None, 16).render(help_text, True, (100, 100, 100))
            screen.blit(help_surface, (episodi_input_rect.centerx - help_surface.get_width() // 2,
                                     episodi_input_rect.bottom + 5))

        #SEZIONE MODALITÀ REALISTICA 
        y_start += 120  #Spazio dopo la sezione episodi
        
        #Stato attuale della modalità
        mode_text = "ATTIVATA" if getattr(env, 'realistic_mode', False) else "DISATTIVATA"
        
        draw_text(screen, f"Modalità Realistica: {mode_text}", 0, y_start, font, (0,0,0), center=True)
        
        #Bottone toggle
        toggle_width = 150  
        toggle_height = 45
        toggle_rect = pygame.Rect(center_x - toggle_width//2, y_start + 40, toggle_width, toggle_height)
        
        button_color = (150, 0, 0) if getattr(env, 'realistic_mode', False) else (0, 150, 0)
        button_text = "DISATTIVA" if getattr(env, 'realistic_mode', False) else "ATTIVA"
        
        pygame.draw.rect(screen, button_color, toggle_rect)
    
        
        button_text_surface = font.render(button_text, True, (255, 255, 255))
        text_x = toggle_rect.centerx - button_text_surface.get_width() // 2
        text_y = toggle_rect.centery - button_text_surface.get_height() // 2
        screen.blit(button_text_surface, (text_x, text_y))

        #Si definisce y_final dopo tutte le sezioni
        y_final = y_start + 120 
        
        #Bottone Conferma
        confirm_rect = pygame.Rect(screen.get_width() // 2 - 220, y_final, 180, 50)
        pygame.draw.rect(screen, (0, 150, 0), confirm_rect)          # Verde
        confirm_text = font.render("Conferma", True, (255, 255, 255))
        screen.blit(confirm_text, (confirm_rect.centerx - confirm_text.get_width() // 2, confirm_rect.centery - confirm_text.get_height() // 2))
        
        #Bottone Annulla
        cancel_rect = pygame.Rect(screen.get_width() // 2 + 40, y_final, 180, 50)
        pygame.draw.rect(screen, (150, 0, 0), cancel_rect)           # Rosso
        cancel_text = font.render("Annulla", True, (255, 255, 255))
        screen.blit(cancel_text, (cancel_rect.centerx - cancel_text.get_width() // 2, cancel_rect.centery - cancel_text.get_height() // 2))

        pygame.display.flip()
        
        #Gestione degli eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return  #Esce senza salvare
            
            #Gestione input tastiera durante editing episodi
            if event.type == pygame.KEYDOWN and editing_episodi:
                if event.key == pygame.K_RETURN:
                    try:
                        if episodi_input.strip():  #Solo se c'è del testo
                            val = int(episodi_input)
                            num_episodi = max(1, min(10000, val))  #Range esteso
                        
                        #Se vuoto, mantieni valore attuale
                    except ValueError:
                        pass
                    editing_episodi = False

                elif event.key == pygame.K_ESCAPE:
                    episodi_input = str(num_episodi)  # Ripristina valore originale
                    editing_episodi = False

                elif event.key == pygame.K_BACKSPACE:
                    episodi_input = episodi_input[:-1]

                else:
                    # Accetta solo cifre, max 5 caratteri (fino a 3000)
                    if event.unicode.isdigit() and len(episodi_input) < 5:
                        episodi_input += event.unicode
            
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
                
                #Controllo cambio percorso auto (0%-90%, step del 10%)
                if auto_less_rect.collidepoint(pos):
                    prob_change_auto = max(0.0, prob_change_auto - 0.10)
                
                if auto_more_rect.collidepoint(pos):
                    prob_change_auto = min(0.9, prob_change_auto + 0.10)

                if episodi_input_rect.collidepoint(pos):
                    editing_episodi = True
                    episodi_input = str(num_episodi)
                
                #Controllo numero episodi (step dinamico: Shift=±100, altrimenti ±10)
                #Nota:Se stai editando da tastiera, i bottoni continuano a funzionare.
                mods = pygame.key.get_mods()
                step = 100 if (mods & pygame.KMOD_SHIFT) else 10

                if episodi_less_rect.collidepoint(pos):
                    num_episodi = max(1, num_episodi - step)

                if episodi_more_rect.collidepoint(pos):
                    num_episodi = min(10000, num_episodi + step)

                #BOTTONE TOGGLE MODALITÀ REALISTICA
                if toggle_rect.collidepoint(pos):
                    env.realistic_mode = not getattr(env, 'realistic_mode', False)
                    
                    #Reinizializza la Q-table per evitare shape mismatch (5D vs 6D)
                    if hasattr(env, 'reinitialize_q_values'):
                        env.reinitialize_q_values()

                #Bottone per la conferma
                if confirm_rect.collidepoint(pos):
                    if editing_episodi:
                        try:
                            val = int(episodi_input) if episodi_input.strip() != "" else num_episodi
                            num_episodi = max(1, min(10000, val))
                        except ValueError:
                            pass
                        editing_episodi = False

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
    prob_change_percorso = env.route_change_probability
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
    if(env.realistic_mode):
        modalita = "Realistica"
    else:    
        modalita = "Semplificata"

    info_line2 = f"Episodi: {num_episodes} | Goal raggiunti: {goals_reached} ({success_rate:.1f}%) | Collisioni: {total_collisions} | Reward medio: {avg_reward:.1f} | Reward max: {max_reward:.1f} | Modalità: {modalita}"

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
    pygame.display.set_caption("Traffic Learner")
    pygame.event.pump()#Forza aggiornamento della finestra
    font =  pygame.font.Font("Progetto Tesi Privitera/assets/PixeloidSansBold.ttf", 20)

    #Inizializzo l'ambiente
    env = Map1Environment(48, 25, 32, screen,                    
        num_pedoni=0,           
        pedone_error_prob=0.0,              
        route_change_probability=0.2,  
        num_episodi = 2000,
        realistic_mode=False,
        seed=88
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
                current_realistic_mode = getattr(env, 'realistic_mode', False) if 'env' in locals() else False
                current_seed = getattr(env, 'seed', None) if 'env' in locals() else None

                env = selected_environment_class(
                    48, 25, 32, screen,
                    #Le successive cinque righe mantengono le impostazioni correnti
                    num_pedoni=current_num_pedoni,           
                    pedone_error_prob=current_error_prob,    
                    route_change_probability=current_route_prob,  
                    num_episodi=current_num_episodi,
                    realistic_mode=current_realistic_mode,
                    seed= current_seed
                )

        elif action == "exit":
            running = False

        elif action == "settings":
            show_settings(screen, font, env)

        elif action == "traffic_training":
            pass
    pygame.quit()

if __name__ == "__main__":
    
    try:
        main()
    
    except Exception as e:
        print(f"Errore imprevisto: {e}")

