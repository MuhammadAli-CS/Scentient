import pandas as pd
from rdkit import Chem
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_smiles(csv_path):
    df = pd.read_csv(csv_path)
    molecules = []
    for index, row in df.iterrows():
        name = row['name']
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append((name, mol))
        else:
            logging.warning(f"Invalid SMILES at row {index}: {row['smiles']}")
    return molecules
