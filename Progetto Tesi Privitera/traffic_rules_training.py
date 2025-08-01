def train_traffic_rules(env, font):
    """Training per imparare regole della strada"""
    # - Obiettivo: Seguire percorsi rispettando regole stradali
    # - Reward: Basato su violazioni regole stradali
    # - Termina quando: Percorso completato o max steps
    # - Q-learning per comportamento stradale corretto
    # il percorso va calcolato con A* e poi l'agente deve seguirlo rispettando le regole

    #PRIMA LO CALCOLO E POI LUI SI ALLENA dopo averlo calcolato

    #Parametri
    epsilon = 1 
    discount_factor = 0.9 
    learning_rate = 0.1 
    num_episodes = getattr(env, 'num_episodes', 2000)
    episode_data = []
    
    #Loop di allenamento
    for episode in range(num_episodes):
        env.reset_game()  #DA FARE

        while not is_route_completed(env, current_route): #Condizione di stop

            #azioni
            action_index = env.get_next_action_index(epsilon)  #Scegli la prossima azione DA FARE
            old_position = env.agent_position[:]

            #movimento
            is_valid = env.get_next_action(action_index)  #Come si muove l'agente?

            #reward
            if is_valid:
                reward = calculate_traffic_reward(env, old_position, action_index)  #Calcolo reward
            else:
                reward = -10  #Penalità per azione non valida

            #q-learning update
            old_q_value = env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index]
            temporal_difference = reward + (discount_factor * np.max(env.q_values[env.agent_position[1], env.agent_position[0], int(env.car_in_vision)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            env.q_values[old_position[1], old_position[0], old_car_in_vision, action_index] = new_q_value