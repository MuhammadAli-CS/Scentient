import pandas as pd
import numpy as np
import logging
from collections import Counter
try:
    from rapidfuzz import process, fuzz
except ImportError:
    pass # Let it fail gracefully if someone tries without rapidfuzz

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

DUPE_BRANDS = ["lattafa", "armaf", "afnan", "fragrance world", "maison alhambra", "zara", "paris corner"]

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
        return list(set(normalized)) # Remove duplicates within layer

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
        # Use log scale so super rare notes don't completely dominate the result, but still matter heavily
        self.rarity_weights = {note: np.log(1 + (max_freq / count)) for note, count in freq.items()}
        
        # Search key
        self.df['SearchKey'] = (self.df['Brand'].str.lower() + " " + self.df['Perfume'].str.lower().str.replace("-", " "))

    def _calculate_cross_layer_sim(self, notes_a, notes_b):
        """
        Calculates similarity with cross-layer penalties.
        Matches in the exact same layer (top-top) = 1.0 weight
        Matches across layers (top-base) = 0.6 weight
        """
        score = 0
        max_score = 0
        cross_penalty = 0.6
        
        # Get unique sets of all notes present
        all_notes_a = set(notes_a['top'] + notes_a['mid'] + notes_a['base'])
        all_notes_b = set(notes_b['top'] + notes_b['mid'] + notes_b['base'])
        
        for note in all_notes_a:
            w = self.rarity_weights.get(note, 1.0)
            max_score += w
            
            if note in all_notes_b:
                # Find best matching layer
                best_match = 0
                for layer in ['top', 'mid', 'base']:
                    if note in notes_a[layer]:
                        if note in notes_b[layer]:
                            best_match = max(best_match, 1.0)
                        elif note in all_notes_b:
                            best_match = max(best_match, cross_penalty)
                score += w * best_match

        # Add remaining notes in B to max_score as penalty for missing notes
        for note in all_notes_b:
            if note not in all_notes_a:
                max_score += self.rarity_weights.get(note, 1.0)
                
        return score / max_score if max_score > 0 else 0

    def _calculate_accord_sim(self, accords_a, accords_b):
        if not accords_a or not accords_b:
            return 0
        union = len(set(accords_a) | set(accords_b))
        return len(set(accords_a) & set(accords_b)) / union if union > 0 else 0

    def find_dupes(self, query, top_n=10, note_weight=0.75, accord_weight=0.25):
        query_norm = str(query).lower().replace("-", " ")
        
        # RapidFuzz match
        matches = process.extract(query_norm, self.df['SearchKey'].tolist(), limit=1, scorer=fuzz.token_set_ratio)
        if not matches:
            logging.warning(f"No match found for query: {query}")
            return None
            
        best_match_key = matches[0][0]
        confidence = matches[0][1]
        
        idx = self.df[self.df['SearchKey'] == best_match_key].index[0]
        base = self.df.iloc[idx]
        
        logging.info(f"Matched input to: {base['Perfume']} by {base['Brand']} (Confidence: {confidence:.1f}%)")

        notes_a = {'top': base['Top'], 'mid': base['Middle'], 'base': base['Base']}
        accords_a = base['Accords']

        results = []
        for i, row in self.df.iterrows():
            if i == idx:
                continue
                
            notes_b = {'top': row['Top'], 'mid': row['Middle'], 'base': row['Base']}
            accords_b = row['Accords']
            
            note_sim = self._calculate_cross_layer_sim(notes_a, notes_b)
            accord_sim = self._calculate_accord_sim(accords_a, accords_b)
            
            # Base similarity
            sim_score = (note_sim * note_weight) + (accord_sim * accord_weight)
            
            # Post-processing heuristics
            brand_a = str(base['Brand']).strip().lower()
            brand_b = str(row['Brand']).strip().lower()
            
            # 1. Flanker Penalty
            if brand_b == brand_a:
                sim_score *= 0.9  
            # 2. Clone House Boost
            elif brand_b in DUPE_BRANDS:
                sim_score *= 1.08 
            
            # 3. Gender Alignment
            if row['Gender'] == base['Gender']:
                sim_score *= 1.05
            
            results.append({
                'Perfume': row['Perfume'],
                'Brand': row['Brand'],
                'Year': row['Year'],
                'Gender': row['Gender'],
                'Similarity (%)': round(min(sim_score * 100, 100), 2),
                'Rating Value': row['Rating Value'],
                'url': row['url']
            })

        results.sort(key=lambda x: x['Similarity (%)'], reverse=True)
        return pd.DataFrame(results[:top_n])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        finder = DupeFinder()
        dupes = finder.find_dupes(query)
        if dupes is not None:
            print(f"\nTop matches for '{query}':\n")
            print(dupes[['Perfume', 'Brand', 'Similarity (%)', 'Rating Value', 'Gender']].to_string(index=False))
    else:
        print("Usage: python dupe_finder.py <perfume_name>")
