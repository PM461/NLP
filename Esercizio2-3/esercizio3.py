import pandas as pd
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

concetti_genus = {
    "Telecomando": "remote_control",
    "Pendrive": "flash_drive",
    "Ansia": "anxiety",
    "Bias": "bias"
}

def trova_synset_tradotto(definizione_it, concetto_en):
    # Traduci la definizione italiana in inglese
    definizione_en = GoogleTranslator(source='it', target='en').translate(definizione_it)
    
    possibili_synset = wn.synsets(concetto_en)
    if not possibili_synset:
        print(f"❌ No synset found for '{concetto_en}'")
        return
    
    def similitudine_testo(s1, s2):
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0
        return len(set1 & set2) / len(set1 | set2)
    
    miglior_synset = None
    miglior_sim = 0
    
    for syn in possibili_synset:
        gloss_en = syn.definition()
        sim = similitudine_testo(definizione_en, gloss_en)
        if sim > miglior_sim:
            miglior_sim = sim
            miglior_synset = syn
    
    if miglior_synset:
        gloss_it = GoogleTranslator(source='en', target='it').translate(miglior_synset.definition())
        print(f"🔎 Definition EN: {definizione_en}")
        print(f"✔ Synset found for '{concetto_en}': {miglior_synset.name()}")
        print(f"  Gloss IT: {gloss_it}")
    else:
        print(f"🔎 Definition EN: {definizione_en}")
        print(f"❌ No relevant synset found for '{concetto_en}'")

df = pd.read_csv('file_definizioni.csv', sep=';', header=None, names=['concetto', 'definizione'])

gruppi = df.groupby('concetto')

for concetto_it, group in gruppi:
    print(f"\n=== CONCEPT: {concetto_it.upper()} ===\n")
    concetto_en = concetti_genus.get(concetto_it)
    if not concetto_en:
        print(f"⚠ No English mapping for {concetto_it}")
        continue
    
    for definizione_it in group['definizione']:
        try:
            trova_synset_tradotto(definizione_it, concetto_en)
        except Exception as e:
            print(f"Error during translation or synset search: {e}")
