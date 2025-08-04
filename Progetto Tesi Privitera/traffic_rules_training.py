import numpy as np
import pygame
import sys
from datetime import datetime

def train_traffic_rules(env, font):
    """Training per regole della strada su tutti i percorsi"""
    
    print("🚀 Avvio Traffic Rules Training...")
    
    # Recupera tutti i percorsi
    if hasattr(env, 'traffic_training_routes'):
        training_routes = env.traffic_training_routes
        print(f"📍 {len(training_routes)} percorsi caricati:")
        
        for i, route in enumerate(training_routes):
            print(f"  {i+1}. {route['name']} - {route['start']} → {route['end']}")
    else:
        print("⚠️ Nessun percorso di allenamento disponibile.")
        return

    # Parametri
    epsilon = 1.0
    discount_factor = 0.9 
    learning_rate = 0.1 
    num_episodes = getattr(env, 'num_episodes', 100)

    episode_data = []
    
    print(f"\n🎯 Inizio training: {num_episodes} episodi su {len(training_routes)} percorsi")
    
    # Loop di allenamento
    for episode in range(num_episodes):
        
        # Gestione eventi pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        
        # ✅ Rotazione tra tutti i percorsi
        current_route = training_routes[episode % len(training_routes)]
        print(f"\nEpisodio {episode}: {current_route['name']} - {current_route['start']} → {current_route['end']}")

        env.reset_for_traffic_rules(current_route['start'], current_route['end'])

        episode_reward = 0
        episode_violations = 0
        steps = 0
        max_steps = 100  # ✅ IMPORTANTE: Timeout per evitare loop infiniti

        # ✅ Loop episodio con timeout
        while not is_route_completed(env, current_route) and steps < max_steps:
            
            # Update ambiente
            env.update_traffic_lights()  #aggiorna le luce dei semafori
            env.update_pedoni(env.pedoni)

            # Azioni
            action_index = env.get_next_action_traffic_rules(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.is_car_in_vision())

            # Movimento
            is_valid = env.get_next_location_traffic_rules(action_index)

            # Reward
            if is_valid:
                reward, violations = calculate_traffic_reward_dedicated(env, old_position, current_route)
                episode_violations += violations
            else:
                reward = -10  # Penalità per azione non valida

            episode_reward += reward
            steps += 1

            # Q-learning update
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.is_car_in_vision())])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            
            # ✅ Display per vedere cosa succede
            display_traffic_training_dedicated(env, font, episode, current_route, episode_violations)
            pygame.time.wait(50)  # Rallenta per vedere

        # ✅ Statistiche episodio
        completed = is_route_completed(env, current_route)
        episode_data.append({
            'episode': episode,
            'route': current_route['name'],
            'reward': episode_reward,
            'violations': episode_violations,
            'steps': steps,
            'completed': completed
        })

        # Status report
        status = "✅ COMPLETATO" if completed else "❌ TIMEOUT"
        print(f"  {status} | Steps: {steps:2d} | Reward: {episode_reward:6.1f} | Violazioni: {episode_violations}")

        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)

    env.set_traffic_rules_mode(False)
    print_final_stats(episode_data)

def reset_for_traffic_training(env, route):
    """Reset ambiente per percorso specifico"""
    
    # Posiziona agente all'inizio del percorso
    env.agent_position = list(route['start'])
    
    # Salva goal temporaneo
    env.current_traffic_goal = route['end']

    #Vogliamo mettere un flag per capire che siamo in modalità training
    #Ricordiamo che i semafori nel trova parcheggio sono gestiti in modo diverso (forzato da codice)
    env.traffic_rules_mode = True

    # Reset normale
    env.reset_game()

def is_route_completed(env, route):
    """Verifica se il percorso è completato"""
    
    distance = abs(env.agent_position[0] - route['end'][0]) + abs(env.agent_position[1] - route['end'][1])
    return distance <= 2

def calculate_traffic_reward_dedicated(env, old_position, route):
    """Sistema reward dedicato per traffic rules"""
    
    base_reward = 1
    violations = 0
    
    # Bonus progresso
    progress_bonus = calculate_progress_bonus_dedicated(env, old_position, route['end'])
    
    # Violazioni con funzioni dedicate
    traffic_violations = check_traffic_rules_violations_dedicated(env)
    violations += traffic_violations['count']
    
    total_reward = base_reward + progress_bonus + traffic_violations['penalty']
    
    return total_reward, violations

def check_traffic_rules_violations_dedicated(env):
    """Sistema violazioni dedicato usando le nuove funzioni"""
    
    violations = {'count': 0, 'penalty': 0}
    agent_pos = env.agent_position
    
    # ✅ REGOLA 1: Semaforo rosso - con funzione dedicata
    if env.check_traffic_light_violation_at_position(agent_pos):
        violations['count'] += 1
        violations['penalty'] -= 20
        print(f"🚨 VIOLAZIONE: Passaggio col rosso!")
    
    # ✅ REGOLA 2: Mantenimento destra
    if check_right_lane_rule_dedicated(env):
        violations['count'] += 1
        violations['penalty'] -= 5
        print(f"🚨 VIOLAZIONE: Non mantieni la destra!")
    
    # ✅ REGOLA 3: Pedoni
    for pedone in env.pedoni:
        distance = abs(agent_pos[0] - pedone.position[0]) + abs(agent_pos[1] - pedone.position[1])
        if distance <= 1:
            violations['count'] += 1
            violations['penalty'] -= 10
            print(f"🚨 VIOLAZIONE: Investito pedone!")
            break
    
    return violations

def check_right_lane_rule_dedicated(env):
    """Controllo regola destra con funzioni dedicate"""
    
    agent_pos = env.agent_position
    x, y = agent_pos
    
    # Verifica se siamo su strada
    if env.map[y][x] != 1:
        return False
    
    # Logica semplificata per mantenimento destra
    # (implementazione dettagliata come prima)
    return check_lane_position_simple(env, x, y)

def check_lane_position_simple(env, x, y):
    """Controllo semplificato posizione corsia"""
    
    # Per ora implementazione base
    # Conta strade intorno per determinare se siamo al centro
    road_count = 0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.width and 0 <= ny < env.height:
            if env.map[ny][nx] == 1:
                road_count += 1
    
    # Se circondati da molte strade, probabilmente siamo al centro
    return road_count >= 3

def calculate_progress_bonus_dedicated(env, old_pos, goal):
    """Bonus progresso con goal dedicato"""
    
    old_distance = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
    new_distance = abs(env.agent_position[0] - goal[0]) + abs(env.agent_position[1] - goal[1])
    
    if new_distance < old_distance:
        return 2
    elif new_distance > old_distance:
        return -1
    return 0

def check_safety_violations(env):
    """Controllo violazioni sicurezza"""
    
    penalty = 0
    violations = 0
    
    # Distanza sicurezza da auto
    for car in env.cars:
        distance = abs(env.agent_position[0] - car['position'][0]) + abs(env.agent_position[1] - car['position'][1])
        if distance < 2:
            penalty -= 15
            violations += 1
            break
    
    return penalty, violations

def display_traffic_training_dedicated(env, font, episode, route, violations):
    """Display dedicato per traffic rules"""
    
    # Display con funzione dedicata se esiste
    if hasattr(env, 'display_traffic_rules_info'):
        env.display_traffic_rules_info(episode)
    else:
        env.display(episode)
    
    # Overlay informazioni
    screen = env.screen
    
    draw_text(screen, f"TRAFFIC RULES (FUNZIONI DEDICATE)", 10, 10, font, (0, 0, 200))
    draw_text(screen, f"Episode: {episode}", 10, 40, font, (0, 0, 0))
    draw_text(screen, f"Route: {route['name']}", 10, 70, font, (0, 0, 0))
    draw_text(screen, f"Violations: {violations}", 10, 100, font, 
             (255, 0, 0) if violations > 0 else (0, 150, 0))
    
    # Goal e start (se non già disegnati dalla funzione dedicata)
    if not hasattr(env, 'current_traffic_goal'):
        goal = route['end']
        goal_rect = pygame.Rect(goal[0] * env.cell_size, goal[1] * env.cell_size, env.cell_size, env.cell_size)
        pygame.draw.rect(screen, (255, 255, 0), goal_rect, 3)
    
    start = route['start']
    start_rect = pygame.Rect(start[0] * env.cell_size, start[1] * env.cell_size, env.cell_size, env.cell_size)
    pygame.draw.rect(screen, (0, 255, 0), start_rect, 3)
    
    pygame.display.flip()

def draw_text(screen, text, x, y, font, color=(0, 0, 0)):
    """Utility per disegnare testo"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def print_final_stats(episode_data):
    """Stampa statistiche finali"""
    
    if not episode_data:
        print("❌ Nessun dato disponibile!")
        return
    
    completed_count = sum(1 for ep in episode_data if ep['completed'])
    total_episodes = len(episode_data)
    completion_rate = (completed_count / total_episodes) * 100 if total_episodes > 0 else 0
    
    total_violations = sum(ep['violations'] for ep in episode_data)
    avg_violations = total_violations / total_episodes if total_episodes > 0 else 0
    
    avg_reward = sum(ep['reward'] for ep in episode_data) / total_episodes if total_episodes > 0 else 0
    avg_steps = sum(ep['steps'] for ep in episode_data) / total_episodes if total_episodes > 0 else 0
    
    print("\n" + "="*60)
    print("🏁 TRAFFIC RULES TRAINING - RISULTATI FINALI")
    print("="*60)
    print(f"📊 Episodi completati: {completed_count}/{total_episodes}")
    print(f"📈 Tasso completamento: {completion_rate:.1f}%")
    print(f"🚨 Violazioni totali: {total_violations}")
    print(f"📉 Violazioni medie: {avg_violations:.2f}")
    print(f"🎯 Reward medio: {avg_reward:.2f}")
    print(f"👣 Steps medi: {avg_steps:.1f}")
    print("="*60)
    
    # Statistiche per percorso
    routes_stats = {}
    for ep in episode_data:
        route_name = ep['route']
        if route_name not in routes_stats:
            routes_stats[route_name] = {'completed': 0, 'total': 0, 'violations': 0}
        
        routes_stats[route_name]['total'] += 1
        routes_stats[route_name]['violations'] += ep['violations']
        if ep['completed']:
            routes_stats[route_name]['completed'] += 1
    
    print("\n📋 PERFORMANCE PER PERCORSO:")
    for route_name, stats in routes_stats.items():
        if stats['total'] > 0:
            completion_rate = (stats['completed'] / stats['total']) * 100
            avg_violations = stats['violations'] / stats['total']
            print(f"  {route_name}: {completion_rate:.1f}% completamento, {avg_violations:.1f} violazioni medie")