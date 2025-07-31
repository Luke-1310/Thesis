def train_traffic_rules(env, font):
    """Training per imparare regole della strada"""
    # - Obiettivo: Seguire percorsi rispettando regole stradali
    # - Reward: Basato su violazioni regole stradali
    # - Termina quando: Percorso completato o max steps
    # - Q-learning per comportamento stradale corretto
    # il percorso va calcolato con A* e poi l'agente deve seguirlo rispettando le regole

    #PRIMA LO CALCOLO E POI LUI SI ALLENA dopo averlo calcolato

