# Guida all’uso dello script WordNet-ConceptNet

---

##  Descrizione

Questo script elabora un insieme di parole prese da WordNet, estrae i loro significati (synsets) e le relazioni semantiche da ConceptNet, e genera due output principali:

- Un file JSON (`wordnet_to_conceptnet_sample.json`) con i dati elaborati.
- Un file Markdown (`word_results.md`) con una visualizzazione leggibile e tabellare delle informazioni per ogni parola.

---

## 🚀 Come eseguire lo script

### Prerequisiti

- Python 3.7 o superiore
- Pacchetti Python installati:
  - `nltk`
  - `requests`
  - `tqdm`
  - `pandas` (opzionale, se usi visualizzazioni aggiuntive)

Puoi installare i pacchetti necessari con:

```bash
pip install nltk requests tqdm pandas
```

### Esecuzione
Lancia lo script con:

```
py Esercizio1.py
```
Durante l’esecuzione vedrai:

1. Una barra di progresso con il numero di parole processate

2. Ogni parola in elaborazione stampata a video

### Output prodotti
**wordnet_to_conceptnet_sample.json**
File JSON contenente per ogni parola:

Lista di synsets con nome e definizione

Lista di relazioni da ConceptNet (relazione, target)

**word_results.md**
File Markdown con struttura leggibile e tabelle per ogni parola, utile per una lettura  più agevole.

**Grafo delle relazioni**
Viene generato un grafo visivo per ognuna delle relazioni fra le parole di Wordnet e ConceptNet