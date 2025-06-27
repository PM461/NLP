# Pseudoword Ambiguity Reduction Analysis
# ======================================
# This notebook processes multilingual pseudowords, retrieves synsets from BabelNet,
# computes ambiguity reduction, and visualizes the results.

# --- 1. SETUP ---

import os
import csv
import json
import logging
import requests
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional, Tuple, Any, Set
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
    logging.info(f"Loaded .env from {dotenv_path}")
else:
    logging.warning("No .env file found.")

API_KEY = os.getenv("BABELNET_API_KEY")
INPUT_FILE = os.getenv("WORD_PAIRS")
LANGS = [lang.strip().upper() for lang in os.getenv("LANGUAGES", "").split(',') if lang.strip()]

# --- 2. HELPER FUNCTIONS ---

def load_word_tuples(filepath: str) -> List[Tuple[str, ...]]:
    with open(filepath, newline='', encoding='utf-8') as f:
        return [tuple(word.strip() for word in row if word.strip())
                for row in csv.reader(f) if len(row) >= 2]

def get_sense(lemma: str, targetLang: List[str], key: str, source: str = "WIKI") -> Optional[List[Dict[str, Any]]]:
    try:
        response = requests.get(
            'https://babelnet.io/v9/getSenses',
            params={
                'lemma': lemma,
                'searchLang': targetLang[0],
                'targetLang': targetLang,
                'key': key,
                'source': source
            }, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"API error: {e}")
        return None

def find_synset_language_dict(synsets: List[dict]) -> Dict[str, Set[str]]:
    result = {}
    for s in synsets:
        props = s.get("properties", {})
        sid = props.get("synsetID", {}).get("id")
        lang = props.get("language", "").upper()
        if sid and lang:
            result.setdefault(lang, set()).add(sid)
    return result

def extract_lemma_for_lang(synsets: List[Dict[str, Any]], synset_id: str, lang: str) -> str:
    for s in synsets:
        props = s.get("properties", {})
        if props.get("synsetID", {}).get("id") == synset_id and props.get("language", "").upper() == lang.upper():
            return props.get("fullLemma") or props.get("simpleLemma") or "N/A"
    return "N/A"

def save_pseudoword_csv(words: Tuple[str, ...], synsets: List[Dict[str, Any]], common_synsets: Set[str], langs: List[str]):
    pseudoword = '-'.join(words)
    path = f"rsrc/pseudowords_{pseudoword}.csv"
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = ['pseudoword'] + [f"{lang.lower()}_word,{lang.lower()}_sense" for lang in langs]
        writer.writerow(["pseudoword"] + [f"{lang}_word" for lang in langs] + [f"{lang}_sense" for lang in langs] + ["common_synset_id"])
        for sid in common_synsets:
            row = [pseudoword]
            for i, lang in enumerate(langs):
                row.append(words[i])
                row.append(extract_lemma_for_lang(synsets, sid, lang))
            row.append(sid)
            writer.writerow(row)

def process_word_tuple(words: Tuple[str, ...], langs: List[str], api_key: str) -> Optional[dict]:
    synsets = get_sense(words[0], langs, api_key)
    if not synsets:
        return None
    lang_synsets = find_synset_language_dict(synsets)
    synset_sets = [lang_synsets.get(lang, set()) for lang in langs]
    if not all(synset_sets):
        return None
    common = set.intersection(*synset_sets)
    total = sum(len(s) for s in synset_sets)
    reduction = (total - len(common) * len(langs)) / total if total else 0.0
    save_pseudoword_csv(words, synsets, common, langs)
    return {"pseudoword": '-'.join(words), "ambiguity_reduction": round(reduction, 3)}

# --- 3. PROCESSING ---

word_tuples = load_word_tuples(INPUT_FILE)
valid_tuples = [wt for wt in word_tuples if len(wt) == len(LANGS)]

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    func = partial(process_word_tuple, langs=LANGS, api_key=API_KEY)
    futures = {executor.submit(func, wt): wt for wt in valid_tuples}
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
        if res:
            results.append(res)

# --- 4. SAVE RESULTS ---

with open('rsrc/ambiguity_scores.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# --- 5. PLOTTING ---

if results:
    sorted_data = sorted(results, key=lambda x: x['ambiguity_reduction'])
    pseudowords = [d['pseudoword'] for d in sorted_data]
    scores = [d['ambiguity_reduction'] for d in sorted_data]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(pseudowords))
    plt.barh(y_pos, scores, color=plt.cm.viridis(np.linspace(0, 1, len(pseudowords))))
    plt.yticks(y_pos, pseudowords)
    plt.xlabel('Ambiguity Reduction')
    plt.title('Ambiguity Reduction per Pseudoword')
    for i, score in enumerate(scores):
        plt.text(score, i, f'{score:.2f}', va='center', ha='left')
    plt.tight_layout()
    plt.savefig('ambiguity_reduction_plot.png')
    plt.show()
else:
    logging.warning("No results to plot.")
