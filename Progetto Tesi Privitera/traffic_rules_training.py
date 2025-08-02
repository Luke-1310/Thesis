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

        # Reset per questo percorso
        reset_for_traffic_training(env, current_route)

        episode_reward = 0
        episode_violations = 0
        steps = 0
        max_steps = 100  # ✅ IMPORTANTE: Timeout per evitare loop infiniti

        # ✅ Loop episodio con timeout
        while not is_route_completed(env, current_route) and steps < max_steps:
            
            # Update ambiente
            env.update_traffic_lights()
            env.update_pedoni(env.pedoni)

            # Azioni
            action_index = env.get_next_action(epsilon)
            old_position = env.agent_position[:]
            old_car_in_vision = int(env.is_car_in_vision())

            # Movimento
            is_valid = env.get_next_location(action_index)

            # Reward
            if is_valid:
                reward, violations = calculate_traffic_reward(env, old_position, current_route)
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
            display_traffic_training(env, font, episode, current_route, episode_violations)
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
    
    # ✅ Risultati finali
    print_final_stats(episode_data)

# ✅ FUNZIONI DI SUPPORTO:

def reset_for_traffic_training(env, route):
    """Reset ambiente per percorso specifico"""
    
    # Posiziona agente all'inizio del percorso
    env.agent_position = list(route['start'])
    
    # Salva goal temporaneo
    env.current_traffic_goal = route['end']
    
    # Reset normale
    env.reset_game()

def is_route_completed(env, route):
    """Verifica se il percorso è completato"""
    
    distance = abs(env.agent_position[0] - route['end'][0]) + abs(env.agent_position[1] - route['end'][1])
    return distance <= 2

def calculate_traffic_reward(env, old_position, route):
    """Sistema reward per regole strada"""
    
    base_reward = 1
    violations = 0
    
    # Bonus progresso verso goal
    progress_bonus = calculate_progress_bonus(env, old_position, route['end'])
    
    # Penalità violazioni
    safety_penalty, safety_violations = check_safety_violations(env)
    violations += safety_violations
    
    total_reward = base_reward + progress_bonus + safety_penalty
    
    return total_reward, violations

def calculate_progress_bonus(env, old_pos, goal):
    """Bonus avvicinamento"""
    
    old_distance = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
    new_distance = abs(env.agent_position[0] - goal[0]) + abs(env.agent_position[1] - goal[1])
    
    if new_distance < old_distance:
        return 2  # Bonus
    elif new_distance > old_distance:
        return -1  # Penalità
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

def display_traffic_training(env, font, episode, route, violations):
    """Display con info training"""
    
    # Display normale
    env.display(episode)
    
    # Overlay informazioni
    screen = env.screen
    
    # Info episodio
    draw_text(screen, f"TRAFFIC RULES - Episode {episode}", 10, 10, font, (0, 0, 0))
    draw_text(screen, f"Route: {route['name']}", 10, 40, font, (0, 0, 0))
    draw_text(screen, f"Goal: {route['end']}", 10, 70, font, (0, 0, 150))
    draw_text(screen, f"Violations: {violations}", 10, 100, font, 
             (255, 0, 0) if violations > 0 else (0, 150, 0))
    
    # Visualizza goal con bordo giallo
    goal = route['end']
    goal_rect = pygame.Rect(goal[0] * env.cell_size, goal[1] * env.cell_size, env.cell_size, env.cell_size)
    pygame.draw.rect(screen, (255, 255, 0), goal_rect, 3)
    
    # Visualizza start con bordo verde
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