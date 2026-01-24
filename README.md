# Breve Introduzione Progetto Tesi

Questo `README` ha lo scopo di dare una breve introduzione sulla struttura del progetto senza dilungarsi molto sui dettagli. Di base la struttura del progetto è la seguente:

```
Progetto Tesi Privitera
|
|_________ "assets/" -> contiene tutti gli assets del progetto come il font e "imgs/" che contiene le immagini e il file .psd per Photoshop
|
|_________ "environments" -> contiene la classe astratta per l'ambiente dove viene addestrato l'agente, le sue due classe derivate (mappe)
|                           e "pedone.py" riguardante il funzionamento dell'entità pedone 
|
|_________ "q_tables" -> contiene una q-table per ciascuna mappa
|
|_________ "q_learning_training.py" -> main del progetto contenente l'algoritmo per addestrare l'agente e le funzioni di setup/save del progetto 
```