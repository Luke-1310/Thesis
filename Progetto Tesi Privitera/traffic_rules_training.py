import numpy as np
import pygame
import sys
from datetime import datetime

def train_traffic_rules(env, font):
    """Training per regole della strada su tutti i percorsi - Versione silenziosa"""
    
    # Recupera tutti i percorsi
    if hasattr(env, 'traffic_training_routes'):
        training_routes = env.traffic_training_routes
    else:
        return  # Nessun percorso disponibile

    # Parametri training
    epsilon = 1.0
    discount_factor = 0.9 
    learning_rate = 0.1 
    num_episodes = getattr(env, 'num_episodes', 100)

    episode_data = []
    
    # Loop di allenamento principale
    for episode in range(num_episodes):
        
        # Gestione eventi pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        
        # Rotazione tra tutti i percorsi
        current_route = training_routes[episode % len(training_routes)]

        # Reset ambiente per percorso corrente
        env.reset_for_traffic_rules(current_route['start'], current_route['end'])

        # Variabili episodio
        episode_reward = 0
        episode_violations = 0
        steps = 0
        max_steps = 100  # Timeout per evitare loop infiniti

        # Loop episodio con timeout
        while not is_route_completed(env, current_route) and steps < max_steps:
            
            # Update ambiente
            env.update_traffic_lights()  # Aggiorna semafori
            env.update_pedoni(env.pedoni)  # Aggiorna pedoni

            # Scelta azione tramite funzione dedicata traffic rules
            action_index = env.get_next_action_traffic_rules(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.is_car_in_vision())

            # Esecuzione movimento tramite funzione dedicata
            is_valid = env.get_next_location_traffic_rules(action_index)

            # Calcolo reward
            if is_valid:
                reward, violations = calculate_traffic_reward_dedicated(env, old_position, current_route)
                episode_violations += violations
            else:
                reward = -10  # Penalità movimento non valido

            episode_reward += reward
            steps += 1

            # Q-learning update standard
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.is_car_in_vision())])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            
            # Display visuale
            display_traffic_training_dedicated(env, font, episode, current_route, episode_violations)
            pygame.time.wait(50)  # Controllo velocità visualizzazione

        # Registrazione statistiche episodio
        completed = is_route_completed(env, current_route)
        episode_data.append({
            'episode': episode,
            'route': current_route['name'],
            'reward': episode_reward,
            'violations': episode_violations,
            'steps': steps,
            'completed': completed
        })

        # Decay epsilon per exploration/exploitation
        epsilon = max(0.01, epsilon * 0.995)

    # Cleanup finale
    env.set_traffic_rules_mode(False)
    
    # TODO: Implementare statistiche grafiche come in q_learning
    # Per ora statistiche base
    show_final_results_summary(episode_data)

def reset_for_traffic_training(env, route):
    """Reset ambiente per percorso specifico"""
    
    # Posizionamento agente
    env.agent_position = list(route['start'])
    
    # Goal temporaneo per questo percorso
    env.current_traffic_goal = route['end']

    # Attivazione modalità traffic rules (disabilita logica automatica semafori)
    env.traffic_rules_mode = True

    # Reset standard ambiente
    env.reset_game()

def is_route_completed(env, route):
    """Verifica completamento percorso tramite distanza Manhattan"""
    
    distance = abs(env.agent_position[0] - route['end'][0]) + abs(env.agent_position[1] - route['end'][1])
    return distance <= 2  # Margine tolleranza

def calculate_traffic_reward_dedicated(env, old_position, route):
    """Sistema reward dedicato per regole stradali"""
    
    base_reward = 1
    violations = 0
    
    # Bonus progressione verso goal
    progress_bonus = calculate_progress_bonus_dedicated(env, old_position, route['end'])
    
    # Controllo violazioni regole strada
    traffic_violations = check_traffic_rules_violations_dedicated(env)
    violations += traffic_violations['count']
    
    total_reward = base_reward + progress_bonus + traffic_violations['penalty']
    
    return total_reward, violations

def check_traffic_rules_violations_dedicated(env):
    """Sistema controllo violazioni regole stradali"""
    
    violations = {'count': 0, 'penalty': 0}
    agent_pos = env.agent_position
    
    # REGOLA 1: Violazione semaforo rosso
    if env.check_traffic_light_violation_at_position(agent_pos):
        violations['count'] += 1
        violations['penalty'] -= 20
    
    # REGOLA 2: Mantenimento corsia destra
    if check_right_lane_rule_dedicated(env):
        violations['count'] += 1
        violations['penalty'] -= 5
    
    # REGOLA 3: Sicurezza pedoni
    for pedone in env.pedoni:
        distance = abs(agent_pos[0] - pedone.position[0]) + abs(agent_pos[1] - pedone.position[1])
        if distance <= 1:
            violations['count'] += 1
            violations['penalty'] -= 10
            break  # Una violazione per step
    
    return violations

def check_right_lane_rule_dedicated(env):
    """Controllo regola mantenimento destra"""
    
    agent_pos = env.agent_position
    x, y = agent_pos
    
    # Verifica posizione su strada valida
    if env.map[y][x] != 1:
        return False  # Non su strada
    
    # Controllo posizione corsia
    return check_lane_position_simple(env, x, y)

def check_lane_position_simple(env, x, y):
    """Controllo semplificato posizione corsia"""
    
    # Conta celle strada adiacenti
    road_count = 0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.width and 0 <= ny < env.height:
            if env.map[ny][nx] == 1:
                road_count += 1
    
    # Violazione se al centro di strada larga (circondato da molte strade)
    return road_count >= 3

def calculate_progress_bonus_dedicated(env, old_pos, goal):
    """Calcolo bonus progressione verso goal"""
    
    old_distance = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
    new_distance = abs(env.agent_position[0] - goal[0]) + abs(env.agent_position[1] - goal[1])
    
    if new_distance < old_distance:
        return 2    # Bonus avvicinamento
    elif new_distance > old_distance:
        return -1   # Penalità allontanamento
    return 0        # Neutro

def display_traffic_training_dedicated(env, font, episode, route, violations):
    """Display visuale training con informazioni percorso"""
    
    # Display ambiente principale
    if hasattr(env, 'display_traffic_rules_info'):
        env.display_traffic_rules_info(episode)
    else:
        env.display(episode)
    
    # Overlay informazioni training
    screen = env.screen
    
    # Info principali
    draw_text(screen, f"TRAFFIC RULES TRAINING", 10, 10, font, (0, 0, 200))
    draw_text(screen, f"Episode: {episode}", 10, 40, font, (0, 0, 0))
    draw_text(screen, f"Route: {route['name']}", 10, 70, font, (0, 0, 0))
    draw_text(screen, f"Violations: {violations}", 10, 100, font, 
             (255, 0, 0) if violations > 0 else (0, 150, 0))
    
    # Visualizzazione goal (giallo) se non già gestito da funzione dedicata
    if not hasattr(env, 'current_traffic_goal'):
        goal = route['end']
        goal_rect = pygame.Rect(goal[0] * env.cell_size, goal[1] * env.cell_size, 
                               env.cell_size, env.cell_size)
        pygame.draw.rect(screen, (255, 255, 0), goal_rect, 3)
    
    # Visualizzazione start (verde)
    start = route['start']
    start_rect = pygame.Rect(start[0] * env.cell_size, start[1] * env.cell_size, 
                            env.cell_size, env.cell_size)
    pygame.draw.rect(screen, (0, 255, 0), start_rect, 3)
    
    pygame.display.flip()

def draw_text(screen, text, x, y, font, color=(0, 0, 0)):
    """Utility rendering testo"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def show_final_results_summary(episode_data):
    """Riepilogo finale risultati (versione minimale)"""
    
    if not episode_data:
        return  # Nessun dato
    
    # Calcoli statistiche base
    completed_count = sum(1 for ep in episode_data if ep['completed'])
    total_episodes = len(episode_data)
    completion_rate = (completed_count / total_episodes) * 100 if total_episodes > 0 else 0
    
    total_violations = sum(ep['violations'] for ep in episode_data)
    avg_violations = total_violations / total_episodes if total_episodes > 0 else 0
    
    # TODO: Implementare display grafico risultati come in q_learning_training
    # Per ora skip output console
    pass