import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import string
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
from itertools import combinations
from collections import defaultdict

# Scarica modelli e risorse
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("it_core_news_sm")
fasttext.util.download_model('it', if_exists='ignore')
model = fasttext.load_model('cc.it.300.bin')

# Leggi il CSV
df = pd.read_csv('file_definizioni.csv', sep=';', header=None, names=['argomento', 'frase'])
frasi = df['frase'].tolist()

# Preprocessing
f = []
for frase in frasi:
    tokens = word_tokenize(frase, language='italian')
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('italian'))
    tokens_senza_stopwords = [token for token in tokens if token not in stop_words]
    doc = nlp(" ".join(tokens_senza_stopwords))
    lemmatizzati = [token.lemma_ for token in doc]
    f.append(lemmatizzati)

# Gruppi
gruppi = {
    'Telecomando': {'index': (0, 10), 'conc': 'concreto', 'spec': 'generico'},
    'Pendrive':    {'index': (10, 20), 'conc': 'concreto', 'spec': 'specifico'},
    'Ansia':       {'index': (20, 30), 'conc': 'astratto', 'spec': 'generico'},
    'Bias':        {'index': (30, 40), 'conc': 'astratto', 'spec': 'specifico'},
}

# Funzioni
def vettore_medio(frase):
    vettori = [model.get_word_vector(parola) for parola in frase if parola]
    if vettori:
        return np.mean(vettori, axis=0)
    else:
        return np.zeros(model.get_dimension())

def jaccard_similarity(lista1, lista2):
    set1, set2 = set(lista1), set(lista2)
    if set1 or set2:
        return len(set1 & set2) / len(set1 | set2)
    else:
        return 0.0

def calcola_simlex_simsem(gruppo_def):
    simsem = []
    simlex = []
    for i, j in combinations(range(len(gruppo_def)), 2):
        v1 = vettore_medio(gruppo_def[i])
        v2 = vettore_medio(gruppo_def[j])
        simsem.append(cosine_similarity([v1], [v2])[0][0])
        simlex.append(jaccard_similarity(gruppo_def[i], gruppo_def[j]))
    return np.mean(simsem), np.mean(simlex)

# Calcolo
risultati = defaultdict(list)

for nome, info in gruppi.items():
    start, end = info['index']
    gruppo = f[start:end]
    simsem_m, simlex_m = calcola_simlex_simsem(gruppo)
    risultati[info['conc']].append((simsem_m, simlex_m))
    risultati[info['spec']].append((simsem_m, simlex_m))
print(risultati)

# Media per concretezza/specificità
print("\n### RISULTATI RAGGRUPPATI ###\n")
for categoria in risultati:
    simsem_media = np.mean([x[0] for x in risultati[categoria]])
    simlex_media = np.mean([x[1] for x in risultati[categoria]])
    print(f"{categoria.upper()} — simsem: {simsem_media:.3f}, simlex: {simlex_media:.3f}")
