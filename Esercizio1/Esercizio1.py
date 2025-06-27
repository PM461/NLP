import json
import random
import logging
import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from typing import List, Dict, Any, Optional, TypedDict
import requests
from requests.exceptions import RequestException
from nltk.corpus.reader import Synset
from nltk.corpus import wordnet as wn

# --- Setup NLTK data ---
# Verifica che il pacchetto 'wordnet' di NLTK sia scaricato, altrimenti lo scarica
nltk_packages = ["wordnet"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# --- Definizione di tipi ---
class SynsetData(TypedDict):
    name: str
    definition: str

class ConceptNetRelation(TypedDict):
    relation: str
    target: str

class WordNetToConceptNetResult(TypedDict):
    synsets: List[SynsetData]
    relations: List[ConceptNetRelation]

# --- Setup logging---
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

# --- Funzione che chiama l'API di ConceptNet per ottenere i dati su una parola ---
def get_conceptnet_entries(word: str) -> List[Dict[str, Any]]:
    try:
        # Normalizza la parola per l'URL (minuscolo, spazi con underscore)
        normalized_word = word.lower().replace(" ", "_")
        url = f"http://api.conceptnet.io/c/en/{normalized_word}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Solleva eccezione se status code è errore
        data = response.json()
        if 'edges' not in data:
            raise ValueError("Missing 'edges' in response")
        return data['edges']  # Ritorna la lista delle relazioni (edges)
    except (RequestException, ValueError) as e:
        logging.error(f"ConceptNet error for '{word}': {e}")
        raise

# --- Funzione che, data una parola, restituisce dati combinati WordNet + ConceptNet ---
def wordnet_to_conceptnet(word: str) -> Optional[WordNetToConceptNetResult]:
    try:
        synsets: List[Synset] = wn.synsets(word)  # Ottiene i synset da WordNet
        if not synsets:
            return None  # Se non ci sono synset, ritorna None

        conceptnet_data = get_conceptnet_entries(word)  # Ottiene relazioni da ConceptNet
        relations: List[ConceptNetRelation] = []

        # Estrae solo le relazioni in cui start ed end sono in inglese
        for edge in conceptnet_data:
            try:
                if (edge["start"]["language"] == "en" and edge["end"]["language"] == "en"):
                    # Ottiene il tipo di relazione e i label dei nodi start/end
                    rel = edge["rel"]["@id"].split("/")[-1]
                    start_label = edge["start"]["label"].lower()
                    end_label = edge["end"]["label"].lower()
                    word_lower = word.lower()

                    # Sceglie come "target" il nodo che NON è la parola cercata
                    if end_label != word_lower:
                        target = edge["end"]["label"]
                    elif start_label != word_lower:
                        target = edge["start"]["label"]
                    else:
                        continue  # Salta se entrambi sono la parola stessa

                    relations.append({"relation": rel, "target": target})
            except (KeyError, AttributeError):
                continue  # Ignora edges malformati

        # Prepara il dizionario risultato con synsets e relazioni
        return {
            "synsets": [{"name": syn.name(), "definition": syn.definition()} for syn in synsets],
            "relations": relations
        }
    except Exception as e:
        logging.error(f"Processing failed for '{word}': {e}")
        return None

# --- Funzione che mostra su console una tabella di synsets e relazioni ---
def show_table(word: str, data: WordNetToConceptNetResult):
    print(f"\n Word: {word}")
    if data["synsets"]:
        print("\n Synsets:")
        print(pd.DataFrame(data["synsets"]))  # Usa pandas per formattare tabella
    if data["relations"]:
        print("\n ConceptNet Relations:")
        print(pd.DataFrame(data["relations"]))

# --- Funzione che disegna un grafo delle relazioni ConceptNet usando NetworkX ---
def draw_concept_graph(word: str, data: WordNetToConceptNetResult):
    G = nx.DiGraph()
    G.add_node(word, color='green')  # Nodo centrale (la parola)

    # Aggiunge nodi e archi per ogni relazione
    for rel in data.get("relations", []):
        G.add_node(rel["target"])
        G.add_edge(word, rel["target"], label=rel["relation"])

    pos = nx.spring_layout(G, seed=42)  # Layout del grafo (posizione nodi)
    edge_labels = nx.get_edge_attributes(G, 'label')  # Etichette archi (relazioni)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f" ConceptNet relations for '{word}'")
    plt.show()
    
def draw_combined_graph(results: Dict[str, WordNetToConceptNetResult]):
    G = nx.DiGraph()

    wordnet_nodes = set(results.keys())  # Parole da WordNet

    for word, data in results.items():
        if not data:
            continue
        G.add_node(word)  # Nodo WordNet
        for rel in data.get("relations", []):
            G.add_node(rel["target"])  # Nodo ConceptNet
            G.add_edge(word, rel["target"], label=rel["relation"])

    pos = nx.spring_layout(G, seed=42, scale=5, iterations=100)

    # Colore nodi: verde se da WordNet, blu se da ConceptNet
    node_colors = []
    for node in G.nodes():
        if node in wordnet_nodes:
            node_colors.append('green')
        else:
            node_colors.append('blue')

    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(16, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1500,
        font_size=10,
        font_weight='bold',
        edge_color='gray'
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legenda: WordNet = verde, ConceptNet = blu
    wordnet_patch = mpatches.Patch(color='green', label='Parole da WordNet')
    conceptnet_patch = mpatches.Patch(color='blue', label='Concetti da ConceptNet')
    plt.legend(handles=[wordnet_patch, conceptnet_patch], loc='upper right', fontsize=10)

    plt.title("Combined ConceptNet (blu) and WordNet (verde) Relations", fontsize=16)
    plt.axis('off')
    plt.show()



# --- Funzione principale: esegue tutto il flusso ---
def main() -> None:
    setup_logging()
    try:
        all_words = set()
        logging.info("Collecting words from WordNet...")
        # Prende tutte le parole da tutti i synset di WordNet
        for synset in wn.all_synsets():
            for lemma in synset.lemmas():
                all_words.add(lemma.name())
        word_list = list(all_words)
        random.shuffle(word_list)  # Mescola per scelta casuale
        selected_words = word_list[:min(20, len(word_list))]  # Prende 20 parole per demo
        results = {}

        logging.info(f"Processing {len(selected_words)} words...\n")
        # Per ogni parola selezionata, chiama la funzione wordnet_to_conceptnet
        for word in tqdm(sorted(selected_words), desc="Processing", unit="word"):
            print(f" Processing: {word}")
            try:
                result = wordnet_to_conceptnet(word)
                results[word] = result
            except Exception:
                logging.warning(f" Failed to process word: {word}")
                results[word] = None

        # Scrive i risultati in un file Markdown con formattazione tabellare
        with open("word_results.md", "w", encoding="utf-8") as md:
            for word, data in results.items():
                if not data:
                    continue

                md.write(f"## 🔤 Word: {word}\n\n")

                # Scrive i synset in tabella markdown
                md.write("### 🧠 Synsets\n")
                if data["synsets"]:
                    md.write("| name | definition |\n|------|------------|\n")
                    for syn in data["synsets"]:
                        name = syn["name"].replace("|", " ")
                        definition = syn["definition"].replace("|", " ")
                        md.write(f"| {name} | {definition} |\n")
                else:
                    md.write("_No synsets found._\n")

                # Scrive le relazioni ConceptNet in tabella markdown
                md.write("\n\n### 🔗 ConceptNet Relations\n")
                if data["relations"]:
                    md.write("| relation | target |\n|----------|--------|\n")
                    for rel in data["relations"]:
                        r = rel["relation"].replace("|", " ")
                        t = rel["target"].replace("|", " ")
                        md.write(f"| {r} | {t} |\n")
                else:
                    md.write("_No ConceptNet relations found._\n")

                md.write("\n\n")

        # Salva anche i risultati raw in JSON per ulteriori elaborazioni
        with open("wordnet_to_conceptnet_sample.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logging.info("\n Processing complete!")

        # Visualizza grafo tra Conceptnet e wordnet delle relazioni
        if(results):
            draw_combined_graph(results)

               

    except KeyboardInterrupt:
        # Se si interrompe con Ctrl+C, salva i risultati parziali
        logging.warning("\n Interruption received, saving partial results...")
        with open("wordnet_to_conceptnet_partial.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("Partial results saved.")

# Entry point
if __name__ == "__main__":
    main()
