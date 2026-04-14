import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SYNONYMS = {
    "bourbon vanilla": "vanilla", "madagascar vanilla": "vanilla", "vanilla absolute": "vanilla",
    "ambergris": "amber", "amberwood": "amber", "amber resin": "amber",
    "rose de mai": "rose", "damask rose": "rose", "turkish rose": "rose", "bulgarian rose": "rose",
    "tonka bean": "tonka", "benzoin resin": "benzoin", "white musk": "musk",
    "musk ketone": "musk", "cacao": "chocolate", "cocoa": "chocolate",
    "oud wood": "oud", "agarwood": "oud", "patchouli leaf": "patchouli",
    "cashmeran": "cashmere wood", "sandalwood oil": "sandalwood", "vetiver oil": "vetiver",
    "green apple": "apple", "bergamot peel": "bergamot", "lemon zest": "lemon",
    "mandarin orange": "mandarin", "tangerine": "mandarin", "orange blossom absolute": "orange blossom",
    "pink peppercorn": "pink pepper", "pepper essence": "pepper"
}

class DupeFinder:
    def __init__(self, data_path="data/fragrantica_cleaned.csv"):
        logging.info(f"Loading fragrance dataset from {data_path}")
        self.df = pd.read_csv(data_path, sep=';')
        self._prepare_data()

    def _clean_and_split(self, text):
        if pd.isna(text):
            return []
        return [x.strip().lower() for x in str(text).split(',')]

    def _normalize_notes(self, note_list):
        normalized = []
        for note in note_list:
            note_clean = note.strip().lower()
            normalized.append(SYNONYMS.get(note_clean, note_clean))
        return normalized

    def _prepare_data(self):
        logging.info("Preprocessing notes and accords...")
        
        # Parse notes
        for col in ['Top', 'Middle', 'Base']:
            self.df[col] = self.df[col].apply(self._clean_and_split).apply(self._normalize_notes)
        self.df['All_Notes'] = self.df['Top'] + self.df['Middle'] + self.df['Base']

        # Parse accords
        accord_cols = ['mainaccord1','mainaccord2','mainaccord3','mainaccord4','mainaccord5']
        self.df['Accords'] = self.df[accord_cols].apply(lambda x: [str(i).lower() for i in x if pd.notna(i)], axis=1)

        # Compute rarity weights
        all_notes = []
        for col in ['Top', 'Middle', 'Base']:
            all_notes.extend([note for notes in self.df[col] for note in notes])
        
        freq = Counter(all_notes)
        max_freq = max(freq.values())
        self.rarity_weights = {note: np.log(1 + (max_freq / count)) for note, count in freq.items()}
        self.all_unique_notes = sorted(set(all_notes))
        
        # Search key
        self.df['SearchKey'] = (self.df['Brand'].str.lower() + " " + self.df['Perfume'].str.lower().str.replace("-", " "))

    def _build_note_vector(self, notes_dict):
        vec = np.zeros(len(self.all_unique_notes))
        for layer in notes_dict.values():
            for note in layer:
                if note in self.all_unique_notes:
                    idx = self.all_unique_notes.index(note)
                    vec[idx] += self.rarity_weights.get(note, 1.0)
        return vec

    def _calculate_similarity(self, vec_a, vec_b, accords_a, accords_b, brand_match, year_match):
        from sklearn.metrics.pairwise import cosine_similarity
        note_sim = cosine_similarity([vec_a], [vec_b])[0][0]
        
        union_accords = len(set(accords_a) | set(accords_b))
        accord_overlap = len(set(accords_a) & set(accords_b)) / max(union_accords, 1)

        score = (note_sim * 0.7) + (accord_overlap * 0.2)
        if brand_match: score += 0.05
        if year_match: score += 0.05
        return min(score * 100, 100)

    def find_dupes(self, query, top_n=10, gender_filter=None):
        query_norm = str(query).lower().replace("-", " ")
        
        # RapidFuzz match
        from rapidfuzz import process
        matches = process.extract(query_norm, self.df['SearchKey'].tolist(), limit=1, scorer=fuzz.token_set_ratio)
        if not matches:
            logging.warning(f"No match found for query: {query}")
            return None
            
        best_match_key = matches[0][0]
        score = matches[0][1]
        
        idx = self.df[self.df['SearchKey'] == best_match_key].index[0]
        base = self.df.iloc[idx]
        
        logging.info(f"Matched input to: {base['Perfume']} by {base['Brand']} (Confidence: {score:.1f}%)")

        notes_a = {'top': base['Top'], 'mid': base['Middle'], 'base': base['Base']}
        accords_a = base['Accords']
        vec_a = self._build_note_vector(notes_a)

        results = []
        for i, row in self.df.iterrows():
            if i == idx:
                continue
            if gender_filter and str(row['Gender']).lower() != gender_filter.lower():
                continue
                
            notes_b = {'top': row['Top'], 'mid': row['Middle'], 'base': row['Base']}
            accords_b = row['Accords']
            vec_b = self._build_note_vector(notes_b)
            
            brand_match = (str(row['Brand']).strip().lower() == str(base['Brand']).strip().lower())
            year_match = (row['Year'] == base['Year'])
            
            sim_score = self._calculate_similarity(vec_a, vec_b, accords_a, accords_b, brand_match, year_match)
            
            results.append({
                'Perfume': row['Perfume'],
                'Brand': row['Brand'],
                'Year': row['Year'],
                'Gender': row['Gender'],
                'Similarity': round(sim_score, 2),
                'Rating Value': row['Rating Value'],
                'url': row['url']
            })

        results.sort(key=lambda x: x['Similarity'], reverse=True)
        return pd.DataFrame(results[:top_n])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        finder = DupeFinder()
        dupes = finder.find_dupes(query)
        if dupes is not None:
            print(f"\nTop matches for '{query}':\n")
            print(dupes.to_string(index=False))
    else:
        print("Usage: python dupe_finder.py <perfume_name>")
