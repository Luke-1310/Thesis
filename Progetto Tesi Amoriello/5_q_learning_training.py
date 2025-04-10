import numpy as np
import pygame
from virtual_environment_5 import VirtualEnvironment
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(rewards, collisions):
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

    # Grafico delle collisioni cumulative (rimane invariato)
    ax2.plot(range(1, len(collisions) + 1), collisions)
    ax2.set_title('Collisioni Cumulative')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Numero di Collisioni')
    ax2.grid(True)

    # Aggiusta lo spazio tra i sottografici
    plt.tight_layout(pad=3.0)

    plt.savefig('training_results.png')
    plt.show()

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def print_q_table(q_table):
    print("Q-Table:")
    print(q_table)

def train_agent(env):
    epsilon = 1
    discount_factor = 0.9
    learning_rate = 0.1
    episode_rewards = []  # Lista per memorizzare le ricompense totali per ogni episodio
    cumulative_collisions = []  # Nuova lista per tenere traccia delle collisioni cumulative
    total_collisions = 0

    for episode in range(10000):
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
                reward = -100
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value
            env.display(episode)
            pygame.time.wait(1)  # Breve pausa per gestire gli eventi

            total_reward += reward
            steps += 1
            if steps > 1000:  # Previeni episodi troppo lunghi
                break
            
           # print_q_table(env.q_values)
        if env.check_loss():  # Aggiungi questa condizione per incrementare il contatore delle collisioni
            total_collisions += 1

        episode_rewards.append(total_reward)  # Memorizza la ricompensa totale per questo episodio
        cumulative_collisions.append(total_collisions)  # Aggiungi il numero cumulativo di collisioni

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

    return episode_rewards, cumulative_collisions

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

def main():
    env = VirtualEnvironment(48, 25, 32)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.display()                # Visualizza l'ambiente

        print("\nScegli un'opzione:")
        print("1. Allenare l'agente")
        print("2. Mostrare risultati con Q-table esistente")
        print("3. Uscire")
        scelta = input("Inserisci il numero dell'opzione: ")

        if scelta == "1":
            rewards, collisions = train_agent(env)
            plot_results(rewards, collisions)
        elif scelta == "2":
            show_results(env)
        elif scelta == "3":
            running = False
        else:
            print("Opzione non valida. Riprova.")

    pygame.quit()

if __name__ == "__main__":
    main()
