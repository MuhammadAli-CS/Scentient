import base64
import html
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dupe_finder import DUPE_BRANDS, DupeFinder  # noqa: E402
from paths import (IMPORTANCE_CSV, LABELS_CSV, METRICS_JSON,  # noqa: E402
                   PERFUMES_CSV, SMILES_CSV)

# RDKit draws structures; Mordred computes the descriptors the classifier needs.
# Either can be missing on a minimal install, so each feature degrades on its own.
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdDepictor, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from predict import predict_smiles_proba
    PREDICT_AVAILABLE = True
except Exception:
    PREDICT_AVAILABLE = False

st.set_page_config(page_title="Scentient", page_icon=":material/science:", layout="wide")

ACCENT = "#9a3d1c"
INK = "#22201c"
MUTED = "#6f695f"
RULE = "#dcd6ca"

st.html(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1 {{ font-weight: 500 !important; letter-spacing: -0.01em; }}
  .lede {{ color: {MUTED}; font-size: 1.02rem; max-width: 46rem; margin: -0.4rem 0 1.4rem; line-height: 1.55; }}
  .pf {{ border-top: 1px solid {RULE}; padding: 14px 0 18px; }}
  .pf-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
  .pf-name {{ font-family: 'Newsreader', serif; font-size: 1.32rem; line-height: 1.2; }}
  .pf-name a {{ color: {INK}; text-decoration: none; }}
  .pf-name a:hover {{ color: {ACCENT}; text-decoration: underline; }}
  .pf-score {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: {MUTED}; white-space: nowrap; }}
  .pf-score b {{ color: {INK}; font-weight: 500; }}
  .pf-match {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; color: {ACCENT}; white-space: nowrap; }}
  .pf-meta {{ color: {MUTED}; font-size: 0.86rem; margin: 2px 0 10px; }}
  .pf-notes {{ display: grid; grid-template-columns: 3.6rem 1fr; gap: 3px 10px; font-size: 0.9rem; margin: 0; }}
  .pf-notes dt {{ color: {MUTED}; font-size: 0.78rem; padding-top: 2px; }}
  .pf-notes dd {{ margin: 0; color: {INK}; }}
  .pf-notes mark {{ background: none; color: {ACCENT}; font-weight: 600; }}
  .pf-accords {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .pf-accords span {{ font-size: 0.75rem; color: {MUTED}; border: 1px solid {RULE}; border-radius: 3px; padding: 1px 7px; }}
  .count {{ color: {MUTED}; font-size: 0.88rem; margin: 6px 0 4px; }}
  .mol {{ display: flex; justify-content: center; border: 1px solid {RULE}; border-radius: 4px; background: #fff; padding: 8px; }}
  .verdict {{ font-family: 'Newsreader', serif; font-size: 2.6rem; line-height: 1; margin: 2px 0 6px; color: {ACCENT}; }}
  .small {{ color: {MUTED}; font-size: 0.85rem; }}
</style>
""")


# ----------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load_perfumes():
    df = pd.read_csv(PERFUMES_CSV, sep=";")
    df["Rating Value"] = pd.to_numeric(df["Rating Value"].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)
    df["Rating Count"] = pd.to_numeric(df["Rating Count"].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Indexing notes and accords…")
def get_dupe_finder():
    return DupeFinder(PERFUMES_CSV)


@st.cache_resource(show_spinner="Indexing notes and accords…")
def get_search_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = load_perfumes()

    def notes(col):
        return df[col].fillna("").str.lower()

    accords = df[[f"mainaccord{i}" for i in range(1, 6)]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    corpus = notes("Top") + " " + notes("Middle") + " " + notes("Base") + " " + accords + " " + accords
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z]+\b", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    return vectorizer, vectorizer.fit_transform(corpus)


@st.cache_data(show_spinner=False)
def dupes_for(idx):
    # Score generously, then filter in the page; one pass is ~2s.
    return get_dupe_finder().find_dupes(base_idx=idx, top_n=300)


def load_metrics():
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            return json.load(f)
    return None


# ----------------------------------------------------------------- rendering helpers

def nice(slug):
    return str(slug).replace("-", " ").title()


def split_notes(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return [n.strip() for n in str(value).split(",") if n.strip()]


def notes_html(notes, shared=None, limit=6):
    if not notes:
        return f"<span style='color:{MUTED}'>not listed</span>"
    out = []
    for n in notes[:limit]:
        text = html.escape(n)
        out.append(f"<mark>{text}</mark>" if shared and n.lower() in shared else text)
    more = f" <span style='color:{MUTED}'>+{len(notes) - limit}</span>" if len(notes) > limit else ""
    return ", ".join(out) + more


def perfume_card(row, right_html, shared=None):
    year = f" · {int(row['Year'])}" if pd.notna(row.get("Year")) else ""
    accords = [a for a in (row.get(f"mainaccord{i}") for i in range(1, 6)) if isinstance(a, str)]
    accord_html = "".join(f"<span>{html.escape(a)}</span>" for a in accords)
    return f"""
    <div class="pf">
      <div class="pf-head">
        <div class="pf-name"><a href="{html.escape(str(row['url']))}" target="_blank">{html.escape(nice(row['Perfume']))}</a></div>
        {right_html}
      </div>
      <div class="pf-meta">{html.escape(nice(row['Brand']))}{year} · {html.escape(str(row['Gender']).title())}</div>
      <dl class="pf-notes">
        <dt>Top</dt><dd>{notes_html(split_notes(row['Top']), shared)}</dd>
        <dt>Heart</dt><dd>{notes_html(split_notes(row['Middle']), shared)}</dd>
        <dt>Base</dt><dd>{notes_html(split_notes(row['Base']), shared)}</dd>
      </dl>
      <div class="pf-accords">{accord_html}</div>
    </div>"""


def rating_html(row):
    return (f"<div class='pf-score'><b>{row['Rating Value']:.2f}</b> / 5 · "
            f"{int(row['Rating Count']):,} ratings</div>")


def render_grid(rows, right_fn, shared=None):
    cols = st.columns(2, gap="large")
    for i, (_, row) in enumerate(rows.iterrows()):
        cols[i % 2].html(perfume_card(row, right_fn(row), shared))


def bar_chart(df, x, y, x_title, height, fmt=".2f", domain=None):
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    return (
        alt.Chart(df)
        .mark_bar(color=ACCENT, height={"band": 0.7})
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, scale=scale),
            y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=[alt.Tooltip(y), alt.Tooltip(x, format=fmt)],
        )
        .properties(height=height)
        .configure(background="transparent")
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=RULE, domainColor=RULE,
                        tickColor=RULE, labelFont="IBM Plex Sans", titleFont="IBM Plex Sans",
                        labelFontSize=12, titleFontWeight="normal", labelLimit=220)
        .configure_view(strokeWidth=0)
    )


def page_header(title, lede):
    st.title(title)
    st.html(f"<p class='lede'>{lede}</p>")


# ----------------------------------------------------------------- pages

def page_browse():
    df = load_perfumes()
    page_header(
        "Browse the catalogue",
        f"{len(df):,} perfumes from {df['Brand'].nunique():,} houses, scraped from Fragrantica. "
        "Filter by note, house or rating. Each name links to its Fragrantica page.",
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    note = c1.text_input("Contains note", placeholder="oud, iris, fig…")
    brand = c2.text_input("House", placeholder="Guerlain, Lattafa…")
    gender = c3.selectbox("For", ["Anyone", "Women", "Men", "Unisex"])
    sort = c4.selectbox("Sort by", ["Most rated", "Highest rated", "Newest"])

    c5, c6, _ = st.columns([1.2, 1.2, 2])
    min_rating = c5.slider("Minimum rating", 0.0, 5.0, 3.5, 0.1)
    min_votes = c6.select_slider("Minimum number of ratings", [0, 10, 50, 100, 500, 1000, 5000], value=50)

    view = df[(df["Rating Value"] >= min_rating) & (df["Rating Count"] >= min_votes)]
    if gender != "Anyone":
        view = view[view["Gender"].str.lower() == gender.lower()]
    if brand:
        view = view[view["Brand"].str.replace("-", " ").str.contains(brand.strip(), case=False, na=False)]
    if note:
        needle = note.strip()
        view = view[view[["Top", "Middle", "Base"]].apply(
            lambda c: c.str.contains(needle, case=False, na=False, regex=False)).any(axis=1)]

    sort_cols = {"Most rated": ["Rating Count"], "Highest rated": ["Rating Value", "Rating Count"], "Newest": ["Year", "Rating Count"]}
    view = view.sort_values(sort_cols[sort], ascending=False)

    per_page = 20
    pages = max(1, int(np.ceil(len(view) / per_page)))
    if len(view) == 0:
        st.info("Nothing matches those filters. Try a lower rating or vote threshold.")
        return
    top_l, top_r = st.columns([3, 1])
    page = top_r.number_input(f"Page (of {pages:,})", 1, pages, 1)
    top_l.html(f"<p class='count'>{len(view):,} perfumes · showing {(page - 1) * per_page + 1}–{min(page * per_page, len(view))}</p>")
    render_grid(view.iloc[(page - 1) * per_page: page * per_page], rating_html)


SEARCH_EXAMPLES = {
    "Smoky leather": "leather smoke birch tobacco",
    "Summer citrus": "bergamot lemon grapefruit neroli fresh",
    "Gourmand": "vanilla tonka caramel coffee",
    "Green and earthy": "vetiver galbanum fig leaf green",
    "Powdery iris": "iris orris violet powdery",
}


def page_search():
    df = load_perfumes()
    page_header(
        "Search by description",
        "Type the notes or accords you want. Results are ranked by TF-IDF similarity over each "
        "perfume's listed notes and main accords, so concrete words like “fig” or “incense” work "
        "better than moods like “cosy”.",
    )

    if "search_q" not in st.session_state:
        st.session_state.search_q = SEARCH_EXAMPLES["Gourmand"]

    def use_example():
        pick = st.session_state.search_pick
        if pick:
            st.session_state.search_q = SEARCH_EXAMPLES[pick]

    st.pills("Examples", list(SEARCH_EXAMPLES), key="search_pick", on_change=use_example)
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Notes and accords", key="search_q")
    gender = c2.selectbox("For", ["Anyone", "Women", "Men", "Unisex"])
    min_votes = c3.select_slider("Min. ratings", [0, 10, 50, 100, 500, 1000], value=50)

    if not query.strip():
        return
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer, matrix = get_search_index()
    sims = cosine_similarity(vectorizer.transform([query.lower()]), matrix).ravel()
    view = df.assign(score=sims)
    view = view[(view["score"] > 0) & (view["Rating Count"] >= min_votes)]
    if gender != "Anyone":
        view = view[view["Gender"].str.lower() == gender.lower()]
    view = view.sort_values(["score", "Rating Count"], ascending=False).head(20)

    if view.empty:
        st.info("None of those words appear in the note lists. Try specific notes such as rose, amber or cedar.")
        return
    words = {w for w in query.lower().replace(",", " ").split()}
    shared = {n.lower() for _, r in view.iterrows() for n in split_notes(r["Top"]) + split_notes(r["Middle"]) + split_notes(r["Base"])
              if any(w in n.lower().split() for w in words)}
    st.html(f"<p class='count'>Top {len(view)} matches. Notes that match your words are highlighted.</p>")
    render_grid(view, lambda r: f"<div class='pf-match'>{r['score']:.2f}</div>", shared)


DUPE_EXAMPLES = {
    "Creed Aventus": "creed aventus",
    "Baccarat Rouge 540": "maison francis kurkdjian baccarat rouge 540",
    "PdM Althaïr": "parfums de marly althair",
    "Tom Ford Lost Cherry": "tom ford lost cherry",
    "YSL Black Opium": "yves saint laurent black opium",
}


def page_alternatives():
    finder = get_dupe_finder()
    page_header(
        "Find alternatives",
        "Pick a perfume and get others with a similar note structure. Rare notes count for more than "
        "common ones, a note in the same layer scores higher than one in a different layer, "
        "and the same house's flankers are pushed down.",
    )

    if "dupe_q" not in st.session_state:
        st.session_state.dupe_q = DUPE_EXAMPLES["Baccarat Rouge 540"]

    def use_example():
        pick = st.session_state.dupe_pick
        if pick:
            st.session_state.dupe_q = DUPE_EXAMPLES[pick]

    st.pills("Examples", list(DUPE_EXAMPLES), key="dupe_pick", on_change=use_example)
    c1, c2 = st.columns([1, 1])
    query = c1.text_input("Perfume name", key="dupe_q")
    candidates = finder.search(query, limit=8) if query.strip() else []
    if not candidates:
        st.info("Type a perfume name, optionally with the house.")
        return
    idx = c2.selectbox("Closest catalogue entries", [i for i, _ in candidates], format_func=finder.describe)

    f1, f2, f3 = st.columns([1, 1, 1])
    clones_only = f1.toggle("Clone houses only", help="Lattafa, Armaf, Afnan, Zara and similar: " + ", ".join(DUPE_BRANDS))
    hide_same = f2.toggle("Hide the same house", value=True)
    min_votes = f3.select_slider("Min. ratings", [0, 10, 50, 100, 500], value=10)

    base = finder.df.iloc[idx]
    perfumes = load_perfumes()
    base_full = perfumes.iloc[idx]
    # Highlight using the raw note names shown on the cards, not the matcher's normalized ones.
    base_notes = {n.lower() for c in ("Top", "Middle", "Base") for n in split_notes(base_full[c])}
    st.html(f"<p class='count'>Your pick</p>{perfume_card(base_full, rating_html(base_full))}")

    with st.spinner("Comparing against the whole catalogue…"):
        results = dupes_for(idx)
    if results is None or results.empty:
        st.warning("No comparable perfumes found.")
        return
    results = results.merge(perfumes[["Rating Count"]], left_on="index", right_index=True)
    results = results[results["Rating Count"] >= min_votes]
    if clones_only:
        results = results[results["Brand"].str.replace("-", " ").str.lower().isin(DUPE_BRANDS)]
    if hide_same:
        results = results[results["Brand"] != base["Brand"]]
    results = results.head(12)
    if results.empty:
        st.info("No matches pass those filters. Loosen them to see more.")
        return

    rows = perfumes.loc[results["index"]].assign(match=results["Similarity (%)"].values)
    st.html(f"<p class='count'>{len(rows)} closest by note structure. Notes shared with {html.escape(nice(base['Perfume']))} are highlighted.</p>")
    render_grid(rows, lambda r: f"<div class='pf-match'>{r['match']:.0f}% match</div>", base_notes)


MOLECULES = {
    "Vanillin": "COc1cc(C=O)ccc1O",
    "Limonene": "CC1=CCC(CC1)C(=C)C",
    "Menthol": "CC(C)C1CCC(C)CC1O",
    "Ambroxan": "CC1(C)CCCC2(C)C1CCC1(C)OCCC12",
    "Ethyl cinnamate": "CCOC(=O)/C=C/c1ccccc1",
    "Geranyl formate": "CC(C)=CCC/C(C)=C/COC=O",
    "Damascenone": "CC=CC(=O)C1=C(C)C=CCC1(C)C",
}


def molecule_svg(mol):
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # st.html sanitizes inline <svg>, so embed it as an image instead.
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def page_predictor():
    page_header(
        "Predict an odor from structure",
        "Enter a SMILES string. The model computes Mordred descriptors, keeps the 250 most informative, "
        "and a random forest assigns one of eleven odor families. It was trained on fewer than a hundred "
        "molecules, so treat the output as a rough guess.",
    )
    if not RDKIT_AVAILABLE:
        st.warning("RDKit is not installed in this environment, so structures can't be parsed. "
                   "Install the packages in requirements.txt to enable this page.")
        return

    if "smiles" not in st.session_state:
        st.session_state.smiles = MOLECULES["Vanillin"]

    def use_example():
        pick = st.session_state.mol_pick
        if pick:
            st.session_state.smiles = MOLECULES[pick]

    st.pills("Examples", list(MOLECULES), key="mol_pick", on_change=use_example,
             help="The first four are in the training set. The last three are not.")
    smiles = st.text_input("SMILES", key="smiles").strip()
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("RDKit couldn't parse that SMILES string. Check brackets, ring closures and charges.")
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html(f"<div class='mol'><img src='{molecule_svg(Chem.Mol(mol))}' width='420' height='300' alt='2D structure'></div>")
        props = pd.DataFrame(
            [
                ("Formula", rdMolDescriptors.CalcMolFormula(mol)),
                ("Molecular weight", f"{Descriptors.MolWt(mol):.1f} g/mol"),
                ("logP (Crippen)", f"{Crippen.MolLogP(mol):.2f}"),
                ("Heavy atoms", str(mol.GetNumHeavyAtoms())),
                ("Rotatable bonds", str(rdMolDescriptors.CalcNumRotatableBonds(mol))),
                ("H-bond donors / acceptors", f"{rdMolDescriptors.CalcNumHBD(mol)} / {rdMolDescriptors.CalcNumHBA(mol)}"),
            ],
            columns=["Property", "Value"],
        )
        st.dataframe(props, hide_index=True, width="stretch")
        if Descriptors.MolWt(mol) > 300:
            st.caption("Most odorants weigh under 300 g/mol. Heavier molecules are usually not volatile enough to smell.")

    with right:
        if not PREDICT_AVAILABLE:
            st.warning("Mordred is not installed, so the classifier can't run. Install the packages in requirements.txt.")
            return
        try:
            with st.spinner("Computing descriptors…"):
                proba = predict_smiles_proba(smiles)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        st.html(f"<div class='small'>Most likely family</div><div class='verdict'>{html.escape(proba.index[0])}</div>"
                f"<div class='small'>{proba.iloc[0]:.0%} of trees agree. Runner-up: {html.escape(proba.index[1])} ({proba.iloc[1]:.0%}).</div>")
        top = proba.head(6).rename_axis("family").reset_index(name="probability")
        st.altair_chart(bar_chart(top, "probability", "family", "Share of trees", 220, ".0%", [0, 1]), width="stretch")
        training = pd.read_csv(SMILES_CSV)
        canon = Chem.MolToSmiles(mol)
        hit = [n for n, s in zip(training["name"], training["smiles"])
               if (m := Chem.MolFromSmiles(s)) is not None and Chem.MolToSmiles(m) == canon]
        if hit:
            label = pd.read_csv(LABELS_CSV).set_index("name").loc[hit[0], "odor"]
            st.caption(f"This is {hit[0]}, which is in the training set labelled “{label}”. Try an unseen molecule for a fair test.")


def page_model():
    page_header(
        "How the odor model works",
        "Training data, accuracy and the descriptors the classifier relies on. Accuracy comes from "
        "stratified cross-validation, with descriptor selection refit inside each fold.",
    )
    metrics = load_metrics()
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cross-validated accuracy", f"{metrics['cv_accuracy']:.0%}")
        m2.metric("Random-guess baseline", f"{metrics['chance_accuracy']:.0%}")
        m3.metric("Training molecules", metrics["n_molecules"])
        m4.metric("Odor families", metrics["n_classes"])
        st.caption(f"{metrics['classifier']}, {metrics['n_estimators']} trees, {metrics['n_features']} descriptors, "
                   f"{metrics['cv_folds']}-fold CV. Last trained {metrics['trained_at']}.")
    else:
        st.info("No metrics yet. Retrain the model below.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Accuracy by family", anchor=False)
        if metrics:
            per = pd.DataFrame([{"family": k, "F1": v["f1"], "molecules": v["support"]} for k, v in metrics["per_class"].items()])
            st.altair_chart(bar_chart(per, "F1", "family", "F1 score (out-of-fold)", 330, ".2f", [0, 1]), width="stretch")
    with right:
        st.subheader("Most informative descriptors", anchor=False)
        if os.path.exists(IMPORTANCE_CSV):
            imp = pd.read_csv(IMPORTANCE_CSV).head(15)
            st.altair_chart(bar_chart(imp, "mi_score", "feature", "Mutual information with odor label", 330), width="stretch")
            st.caption("Names are Mordred descriptor codes; see the Mordred documentation for definitions.")

    with st.expander("Training molecules"):
        data = pd.read_csv(SMILES_CSV).merge(pd.read_csv(LABELS_CSV), on="name")
        st.dataframe(data, hide_index=True, width="stretch", height=360)
        st.caption("Each SMILES was checked against its known molecular formula. Edit data/sample_smiles.csv and "
                   "data/odor_labels.csv to add molecules, then retrain.")

    st.subheader("Retrain", anchor=False)
    st.write("Recomputes descriptors for every training molecule, re-selects features, cross-validates and "
             "refits the forest. Takes about half a minute.")
    if st.button("Retrain the model", type="primary", disabled=not PREDICT_AVAILABLE):
        from pipeline import run_train_pipeline
        with st.status("Retraining…", expanded=True) as status:
            try:
                new = run_train_pipeline(progress=lambda i, n, msg: status.write(f"{i}/{n} · {msg}"))
                status.update(label=f"Done. Cross-validated accuracy {new['cv_accuracy']:.0%}.", state="complete")
            except Exception as e:
                status.update(label="Retraining failed", state="error")
                st.exception(e)
    if not PREDICT_AVAILABLE:
        st.caption("Retraining needs RDKit and Mordred, which aren't installed here.")


nav = st.navigation(
    [
        st.Page(page_browse, title="Browse", default=True),
        st.Page(page_search, title="Search", url_path="search"),
        st.Page(page_alternatives, title="Alternatives", url_path="alternatives"),
        st.Page(page_predictor, title="Odor predictor", url_path="predict"),
        st.Page(page_model, title="Model", url_path="model"),
    ],
    position="top",
)
nav.run()
