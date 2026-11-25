[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](./LICENSE)
# Scentient

A Streamlit app for exploring perfumes and predicting how molecules smell.

- **Browse**: filter 24,000 Fragrantica perfumes by note, house, rating and audience.
- **Search**: rank perfumes by TF-IDF similarity to a list of notes and accords.
- **Alternatives**: find perfumes with a similar note structure. Rare notes weigh more, same-layer matches beat cross-layer ones, and flankers are pushed down.
- **Odor predictor**: parse a SMILES string with RDKit, compute Mordred descriptors and classify it into one of eleven odor families with a random forest.
- **Model**: cross-validated accuracy, per-family F1, the most informative descriptors, and a button to retrain.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Odor model

Training data lives in `data/sample_smiles.csv` and `data/odor_labels.csv`: 93 common odorants, each SMILES checked against its known molecular formula. Retrain from the Model page or the command line:

```bash
python src/main.py --train
python src/main.py --predict "COc1cc(C=O)ccc1O"
python src/main.py --dupe "baccarat rouge 540"
```

Accuracy is measured with stratified 5-fold cross-validation, with descriptor selection refit inside each fold, and written to `models/metrics.json`.
