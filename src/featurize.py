import logging

import pandas as pd
from mordred import Calculator, descriptors

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_CALC = None


def get_calculator():
    """Build the 2D Mordred calculator once; construction is slow."""
    global _CALC
    if _CALC is None:
        _CALC = Calculator(descriptors, ignore_3D=True)
    return _CALC


def featurize(molecules, output_csv):
    calc = get_calculator()
    results = []

    for name, mol in molecules:
        try:
            desc_dict = calc(mol).asdict()
            desc_dict['name'] = name
            results.append(desc_dict)
        except Exception as e:
            logging.error(f"Error calculating descriptors for {name}: {e}")

    df = pd.DataFrame(results)
    cols = ['name'] + [c for c in df.columns if c != 'name']
    df = df[cols]
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(df)} molecules to {output_csv}")
