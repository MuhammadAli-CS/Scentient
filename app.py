import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import altair as alt
import textwrap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def clean_html(html_str):
    dedented = textwrap.dedent(html_str)
    return " ".join(line.strip() for line in dedented.splitlines())


# Ensure src/ is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dupe_finder import DupeFinder
from predict import predict_smiles, featurize_smiles_in_memory
from train_model import train_model

# Page config
st.set_page_config(
    page_title="Scentient - Molecular Olfaction Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (ultra-minimalist luxury Gold & Obsidian theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .brand-title, .main-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    .sub-title, .section-title, .widget-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    /* Deep Obsidian Black Matte Background */
    .stApp {
        background: #050507;
        color: #a1a1b5;
    }
    
    /* Sidebar Overhaul: Ultra-Clean luxury Menu */
    section[data-testid="stSidebar"] {
        background-color: #030304 !important;
        border-right: 1px solid #111115 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #111115 !important;
    }
    
    /* Navigation Radio Items - Clean Minimal Links */
    div[data-testid="stSidebar"] div.stRadio > div {
        background-color: transparent !important;
        padding: 5px !important;
    }
    div[data-testid="stSidebar"] div.stRadio label {
        background: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 10px 0px 10px 15px !important;
        margin-bottom: 12px !important;
        color: #717188 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        border-left: 1px solid transparent !important;
    }
    div[data-testid="stSidebar"] div.stRadio label:hover {
        color: #e2e8f0 !important;
        padding-left: 20px !important;
    }
    div[data-testid="stSidebar"] div.stRadio label[data-checked="true"] {
        color: #d4af37 !important;
        font-weight: 500 !important;
        border-left: 2px solid #d4af37 !important;
        padding-left: 20px !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] div.stRadio label > div:first-child {
        display: none !important; /* Hide native radio circles */
    }
    
    /* Minimalist Inputs and Selectboxes */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #0b0b0d !important;
        border: 1px solid #16161f !important;
        border-radius: 6px !important;
        color: #cbd5e1 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    div[data-testid="stNumberInput"] input:hover {
        border-color: #2b2b38 !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #d4af37 !important;
        box-shadow: none !important;
    }
    
    /* Solid Flat Burnished Gold Buttons */
    div.stButton > button {
        background: #b28b12 !important;
        color: #000000 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: 1.5px !important;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #d4af37 !important;
        transform: none !important;
        box-shadow: none !important;
        color: #000000 !important;
    }
    div.stButton > button:active {
        background: #90700d !important;
    }
    
    /* Refined Gold Slider */
    div.stSlider > div {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    div.stSlider [data-testid="stThumb"] {
        background-color: #d4af37 !important;
        border: 1px solid #000 !important;
        box-shadow: none !important;
        border-radius: 50% !important;
        width: 14px !important;
        height: 14px !important;
    }
    div.stSlider [style*="background-color: rgb(255, 75, 75)"] {
        background-color: #d4af37 !important;
    }
    div.stSlider [style*="background-color: rgb(244, 63, 94)"] {
        background-color: #d4af37 !important;
    }
    div.stSlider [data-testid="stSliderTrack"] > div {
        background-color: #16161f !important;
    }
    
    /* Sidebar Branding */
    .brand-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 35px;
        padding: 20px 10px;
        background: transparent;
        border-radius: 0px;
        border: none;
        border-bottom: 1px solid #111115;
        box-shadow: none;
    }
    .brand-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 24px;
        font-weight: 300;
        color: #e2e8f0;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin: 0px;
    }
    
    /* Matte Luxury Cards */
    .glass-card {
        background: #0b0b0e;
        border: 1px solid #14141a;
        border-radius: 12px;
        padding: 24px;
        box-shadow: none;
        margin-bottom: 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: #2b2b36;
    }
    
    /* Rating & Gender labels - Monochromatic Minimalist */
    .rating-pill {
        background: transparent;
        color: #d4af37;
        font-weight: 500;
        padding: 0px;
        border: none;
        font-size: 12px;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }
    
    .gender-pill {
        background: transparent;
        color: #717188;
        padding: 0;
        border: none;
        font-size: 11px;
        text-transform: uppercase;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 1px;
    }

    /* Links */
    .fragrantica-link {
        color: #cbd5e1;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .fragrantica-link:hover {
        color: #d4af37;
        text-decoration: underline;
    }

    /* Molecules container */
    .mol-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #08080a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #14141a;
        box-shadow: none;
    }
    
    /* Clean Typography Olfactory Pyramid */
    .pyramid-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 15px;
    }
    .pyramid-tier {
        border-bottom: 1px solid #111116;
        padding-bottom: 10px;
    }
    .pyramid-tier:last-child {
        border-bottom: none;
    }
    .tier-header {
        font-family: 'Outfit', sans-serif;
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }
    .tier-top { color: #fda4af; }
    .tier-mid { color: #a7f3d0; }
    .tier-base { color: #fde68a; }
    
    /* Clean text notes */
    .notes-list {
        font-size: 13.5px;
        color: #cbd5e1;
        line-height: 1.5;
        font-weight: 300;
    }
    
    /* Accords list */
    .accord-text {
        font-family: 'Outfit', sans-serif;
        font-size: 9px;
        font-weight: 400;
        text-transform: uppercase;
        color: #d4af37;
        letter-spacing: 1px;
        margin-right: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED RESOURCES & DATA LOADING -----------------

@st.cache_resource
def get_dupe_finder():
    try:
        return DupeFinder(data_path="data/fragrantica_cleaned.csv")
    except Exception as e:
        st.error(f"Error loading DupeFinder: {e}")
        return None

@st.cache_resource
def load_perfume_data():
    df = pd.read_csv("data/fragrantica_cleaned.csv", sep=';')
    # Clean Rating Value: European comma decimal → float
    if 'Rating Value' in df.columns:
        df['Rating Value'] = pd.to_numeric(
            df['Rating Value'].astype(str).str.replace(',', '.'),
            errors='coerce'
        ).fillna(0.0)
    # Clean Rating Count and Year to numeric
    for col in ['Rating Count', 'Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_resource
def load_search_engine(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Clean lists of notes
    def process_notes(text):
        if pd.isna(text): return ''
        return ' '.join([x.strip().lower() for x in str(text).split(',')])
    
    top_clean = df['Top'].apply(process_notes)
    mid_clean = df['Middle'].apply(process_notes)
    base_clean = df['Base'].apply(process_notes)
    
    accord_cols = ['mainaccord1','mainaccord2','mainaccord3','mainaccord4','mainaccord5']
    accords = df[accord_cols].apply(lambda x: ' '.join([str(i).lower() for i in x if pd.notna(i)]), axis=1)
    
    # Concatenate corpus texts (notes repeated to balance importances)
    corpus = (
        "top: " + top_clean + " " +
        "middle: " + mid_clean + " " +
        "base: " + base_clean + " " +
        "accords: " + accords
    )
    
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b') # Keep single-character notes if any
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

# Load assets
df_perfumes = load_perfume_data()
dupe_finder = get_dupe_finder()

# Helper for physical tiered notes display
def get_pyramid_html(top_notes, mid_notes, base_notes, accords):
    top_str = ", ".join(top_notes) if top_notes else "N/A"
    mid_str = ", ".join(mid_notes) if mid_notes else "N/A"
    base_str = ", ".join(base_notes) if base_notes else "N/A"
    
    accords_html = "".join([f"<span class='accord-text'>• {a.upper()}</span>" for a in accords])
    
    return clean_html(f"""
    <div class="pyramid-container">
        <div class="pyramid-tier">
            <div class="tier-header tier-top">Top Notes</div>
            <div class="notes-list">{top_str}</div>
        </div>
        <div class="pyramid-tier">
            <div class="tier-header tier-mid">Heart Notes</div>
            <div class="notes-list">{mid_str}</div>
        </div>
        <div class="pyramid-tier">
            <div class="tier-header tier-base">Base Notes</div>
            <div class="notes-list">{base_str}</div>
        </div>
        <div style="margin-top: 5px; padding-top: 8px;">
            {accords_html}
        </div>
    </div>
    """)

# ----------------- SIDEBAR BRANDING & NAVIGATION -----------------

with st.sidebar:
    st.markdown(clean_html("""
    <div class="brand-container">
        <span class="brand-title">SCENTIENT</span>
    </div>
    """), unsafe_allow_html=True)
    
    st.markdown("<p style='font-family: \"Outfit\", sans-serif; font-size: 10px; color: #d4af37; margin-top: -24px; margin-bottom: 30px; letter-spacing: 2px; text-transform: uppercase; font-weight: 400; text-align: center;'>Molecular Olfaction</p>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; color: #717188; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; padding-left: 15px;'>Navigation</h4>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Go to page:",
        [
            "Home & Discovery Explorer", 
            "Semantic Scent Search", 
            "Dupe & Alternative Discovery",
            "Molecular Odor Predictor",
            "ML Dashboard & Features"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; color: #d4af37; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; padding-left: 15px;'>Olfactive Repository</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 13px; color: #a1a1b5; padding-left: 15px; line-height: 1.6;'><b>Fragrances:</b> {len(df_perfumes):,}<br><b>Brands:</b> {df_perfumes['Brand'].nunique():,}<br><b>Demographics:</b> Unisex, Pour Homme, Pour Femme</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='font-size: 10px; color: #4b4b5c; text-align: center; font-weight: 300;'>Scentient Platform • Developed by Muhammad Ali<br>Powered by RDKit, Mordred & Random Forests</div>", unsafe_allow_html=True)

# ----------------- RDKIT MOLECULE RENDERING -----------------

def render_molecule_svg(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        Chem.rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSvg(400, 320)
        
        # Style Options for Premium Look (Dark background, clear contrast)
        opts = drawer.drawOptions()
        opts.backgroundColour = (0.04, 0.04, 0.06, 1.0) # Match dark card style
        opts.legendFontSize = 13
        opts.annotationFontSize = 13
        opts.multipleBondOffset = 0.15
        opts.bondThickness = 2.0
        
        # Color atoms clearly
        opts.useBWDirectly = False
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        return f"Error drawing molecule: {e}"

# ----------------- PAGES IMPLEMENTATION -----------------

# Page 1: Home & Discovery Explorer
if menu == "Home & Discovery Explorer":
    # Stunning editorial split hero layout
    hero_col1, hero_col2 = st.columns([5, 4])
    
    with hero_col1:
        st.markdown("""
        <h1 style="font-family: 'Cormorant Garamond', serif; font-size: 55px; font-weight: 300; line-height: 1.0; margin-top: 20px; margin-bottom: 10px; color: #f3d060; letter-spacing: 2px; text-transform: uppercase;">
            SCENTIENT
        </h1>
        <h3 class="sub-title" style="font-size: 11px; font-weight: 400; color: #717188; margin-top:0px; margin-bottom: 30px; letter-spacing: 4px; text-transform: uppercase;">
            Molecular Olfaction Platform
        </h3>
        <p style="font-size: 15px; line-height: 1.8; color: #a1a1b5; margin-bottom: 20px; font-weight: 300;">
            Welcome to the intersection of molecular organic chemistry and sensory aesthetics. Scentient decodes the olfactive universe by bridging mathematical compound geometry and the natural vocabulary of scent.
        </p>
        <p style="font-size: 14px; line-height: 1.8; color: #717188; margin-bottom: 30px; font-weight: 300;">
            Utilizing topological graph descriptors via <b>RDKit</b>, high-dimensional machine learning estimators, and semantic vector indexing, Scentient parses chemical molecules, executes note-weighted searches, and retrieves direct budget-friendly olfactive equivalents.
        </p>
        """, unsafe_allow_html=True)
        
    with hero_col2:
        img_file = "generic_scent.png" if os.path.exists("generic_scent.png") else "minimalist_perfume.png"
        if os.path.exists(img_file):
            st.image(img_file, use_container_width=True)
            
    st.markdown("<hr style='border-color: #111115; margin: 35px 0px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 28px; font-weight: 300; color: #f3d060; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;'>Olfactive Discovery Library</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #717188; margin-bottom: 30px; font-weight:300; font-size:14.5px;'>Explore, filter, and dissect the notes of over 24,000 elite luxury, designer, and boutique perfumes.</p>", unsafe_allow_html=True)
    
    # Scent Discovery Filters
    filter_card = st.container()
    with filter_card:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender_filter = st.selectbox("Olfactive Demography", ["All", "Unisex", "Men", "Women"])
        with col2:
            rating_filter = st.slider("Minimum Quality Score", 0.0, 5.0, 3.8, 0.1)
        with col3:
            search_brand = st.text_input("Brand / House Search", placeholder="e.g. Tom Ford, Creed...")
        with col4:
            search_note = st.text_input("Olfactory Note Key", placeholder="e.g. Vanilla, Oud, Bergamot...")
            
    # Query logic
    filtered_df = df_perfumes.copy()
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['Gender'].str.lower() == gender_filter.lower()]
    filtered_df = filtered_df[filtered_df['Rating Value'] >= rating_filter]
    if search_brand:
        filtered_df = filtered_df[filtered_df['Brand'].str.contains(search_brand, case=False, na=False)]
    if search_note:
        note_match = (
            filtered_df['Top'].str.contains(search_note, case=False, na=False) |
            filtered_df['Middle'].str.contains(search_note, case=False, na=False) |
            filtered_df['Base'].str.contains(search_note, case=False, na=False)
        )
        filtered_df = filtered_df[note_match]
        
    st.markdown(f"<p style='font-family: \"Outfit\", sans-serif; font-size: 12px; color: #d4af37; margin-top: 15px; text-transform: uppercase; letter-spacing: 1px;'>Catalogued matches: <b>{len(filtered_df):,}</b> perfumes found</p>", unsafe_allow_html=True)
    
    # Paginate results for clean loading
    limit = 12
    pages = max(1, int(np.ceil(len(filtered_df) / limit)))
    
    col_p1, col_p2 = st.columns([1, 6])
    with col_p1:
        page_num = st.number_input("Page", min_value=1, max_value=pages, step=1, value=1)
    
    start_idx = (page_num - 1) * limit
    end_idx = start_idx + limit
    display_df = filtered_df.iloc[start_idx:end_idx]
    
    # Display Grid
    if not display_df.empty:
        col_grid1, col_grid2 = st.columns(2)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            target_col = col_grid1 if idx % 2 == 0 else col_grid2
            
            top_notes = [x.strip() for x in str(row['Top']).split(',') if pd.notna(row['Top'])][:4]
            mid_notes = [x.strip() for x in str(row['Middle']).split(',') if pd.notna(row['Middle'])][:4]
            base_notes = [x.strip() for x in str(row['Base']).split(',') if pd.notna(row['Base'])][:4]
            
            accord_list = [row[f'mainaccord{i}'] for i in range(1, 6) if pd.notna(row[f'mainaccord{i}'])]
            
            pyramid_html = get_pyramid_html(top_notes, mid_notes, base_notes, accord_list)
            
            card_content = clean_html(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                    <div>
                        <h4 style="margin: 0px 0px 4px 0px; font-size: 22px; font-weight: 500;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                        <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 11px; color: #d4af37; font-weight: 400; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).title()}</p>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <span class="gender-pill">{row['Gender']}</span>
                        <span class="rating-pill">{row['Rating Value']:.2f} Rating</span>
                    </div>
                </div>
                {pyramid_html}
            </div>
            """)
            target_col.markdown(card_content, unsafe_allow_html=True)
    else:
        st.info("No fragrances match your filter parameters. Try expanding your search criteria!")

# Page 2: Semantic Scent Search
elif menu == "Semantic Scent Search":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px; font-weight: 300; text-transform: uppercase; letter-spacing: 1px;'>Semantic Scent Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #717188; margin-bottom: 25px;'>Describe your desired olfactive atmosphere in natural language to retrieve matches.</p>", unsafe_allow_html=True)
    
    vectorizer, tfidf_matrix = load_search_engine(df_perfumes)
    
    st.markdown(
        """
        <div class="glass-card">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Olfactive PYRAMID Vectorization</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.7; color: #a1a1b5; font-weight: 300;">
                We represent each perfume as a structured document capturing its exact olfactive layers and principal accords. Using a customized TF-IDF Vectorizer, we convert your descriptions into high-dimensional keyword representations and execute a Cosine Similarity search against our entire 24,000+ library.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Describe the atmosphere or specific notes you seek:", "sweet chocolate coffee winter perfume with vanilla base")
    with col_s2:
        s_gender = st.selectbox("Olfactive Demography", ["All", "Unisex", "Men", "Women"])
        
    num_results = st.slider("Result Count Limit", 4, 30, 8, 2)
    
    if query:
        from sklearn.metrics.pairwise import cosine_similarity
        
        st.markdown(f"<h3 style='color: #f3d060; font-size: 22px; font-weight: 300; margin-top: 25px; text-transform: uppercase; letter-spacing: 1px;'>Olfactive Matches for: <i>\"{query}\"</i></h3>", unsafe_allow_html=True)
        
        # Transform query
        query_vec = vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        search_df = df_perfumes.copy()
        search_df['Similarity'] = sims * 100
        
        if s_gender != "All":
            search_df = search_df[search_df['Gender'].str.lower() == s_gender.lower()]
            
        search_df = search_df.sort_values(by='Similarity', ascending=False).head(num_results)
        
        if not search_df.empty and search_df['Similarity'].max() > 0:
            col_res1, col_res2 = st.columns(2)
            
            for i, (_, row) in enumerate(search_df.iterrows()):
                target_col = col_res1 if i % 2 == 0 else col_res2
                
                top_notes = [x.strip() for x in str(row['Top']).split(',') if pd.notna(row['Top'])][:4]
                mid_notes = [x.strip() for x in str(row['Middle']).split(',') if pd.notna(row['Middle'])][:4]
                base_notes = [x.strip() for x in str(row['Base']).split(',') if pd.notna(row['Base'])][:4]
                
                accord_list = [row[f'mainaccord{k}'] for k in range(1, 6) if pd.notna(row[f'mainaccord{k}'])]
                
                pyramid_html = get_pyramid_html(top_notes, mid_notes, base_notes, accord_list)
                
                card_content = clean_html(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 22px; font-weight: 500;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 11px; color: #d4af37; font-weight: 400; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).title()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 4px;">
                            <span class="rating-pill">{row['Similarity']:.1f}% Similarity</span>
                            <span class="gender-pill">{row['Rating Value']:.2f} Rating ({row['Gender']})</span>
                        </div>
                    </div>
                    {pyramid_html}
                </div>
                """)
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.warning("No matches found. Try entering alternative scent descriptors!")

# Page 3: Dupe & Alternative Discovery
elif menu == "Dupe & Alternative Discovery":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px; font-weight: 300; text-transform: uppercase; letter-spacing: 1px;'>Dupe & Alternative Discovery</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #717188; margin-bottom: 25px;'>Discover highly accurate, cost-effective equivalents of prestigious luxury fragrances.</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Olfactory Mapping Architecture</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.7; color: #a1a1b5; font-weight: 300;">
                Our algorithm rates component rarity mathematically (rare notes weigh more heavily), penalizes cross-tier shifts (matching top notes directly to base notes), and integrates dedicated clone house filters (Lattafa, Armaf, Afnan) to target direct equivalents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Default high-end luxury searches
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 11px; color: #d4af37; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;'>Prestigious Fragrances</h4>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    query_dupe = "baccarat-rouge-540"
    if col_btn1.button("Creed Aventus"):
        query_dupe = "aventus"
    if col_btn2.button("Baccarat Rouge 540"):
        query_dupe = "baccarat-rouge-540"
    if col_btn3.button("Parfums de Marly Althair"):
        query_dupe = "althair"
    if col_btn4.button("Tom Ford Lost Cherry"):
        query_dupe = "lost-cherry"
        
    query_search = st.text_input("Or query a specific prestigious model title:", query_dupe)
    
    if query_search and dupe_finder:
        with st.spinner("Executing similarity mapping..."):
            dupe_df = dupe_finder.find_dupes(query_search, top_n=10)
            
        if dupe_df is not None and not dupe_df.empty:
            query_norm = str(query_search).lower().replace("-", " ")
            try:
                from rapidfuzz import process, fuzz
                matches = process.extract(query_norm, dupe_finder.df['SearchKey'].tolist(), limit=1, scorer=fuzz.token_set_ratio)
                best_match_key = matches[0][0]
                base_idx = dupe_finder.df[dupe_finder.df['SearchKey'] == best_match_key].index[0]
                base_perfume = dupe_finder.df.iloc[base_idx]
            except Exception:
                base_perfume = None
                
            if base_perfume is not None:
                st.markdown(
                    f"""
                    <div style="background: transparent; border: 1px solid #22222d; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
                        <h4 style="margin: 0px 0px 6px 0px; font-family: 'Cormorant Garamond', serif; font-size: 22px; color: #f3d060; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Selected Luxury Fragrance: {str(base_perfume['Perfume']).replace("-", " ").title()} by {str(base_perfume['Brand']).title()}</h4>
                        <p style="margin:0px 0px 10px 0px; font-family: 'Outfit', sans-serif; font-size:11px; color:#717188; text-transform: uppercase; letter-spacing: 1px;">Demography: <b>{base_perfume['Gender']}</b> | Score: <b>{float(base_perfume['Rating Value']):.2f} Rating</b> | Year: <b>{int(base_perfume['Year']) if pd.notna(base_perfume['Year']) else 'N/A'}</b></p>
                        <p style="margin:0px; font-size:13.5px; color:#cbd5e1; line-height:1.6; font-weight: 300;"><b>Olfactive Pyramid:</b> Top ({', '.join(base_perfume['Top'])}) • Heart ({', '.join(base_perfume['Middle'])}) • Base ({', '.join(base_perfume['Base'])})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("<h3 style='color: #f3d060; font-size: 24px; font-weight: 300; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;'>Olfactive Equivalents</h3>", unsafe_allow_html=True)
            
            # Display results in columns
            col_dupe1, col_dupe2 = st.columns(2)
            for i, (_, row) in enumerate(dupe_df.iterrows()):
                target_col = col_dupe1 if i % 2 == 0 else col_dupe2
                
                # Fetch full data of the dupe perfume to get notes
                full_dupe = df_perfumes[df_perfumes['Perfume'] == row['Perfume']].iloc[0]
                
                dupe_top = [x.strip() for x in str(full_dupe['Top']).split(',') if pd.notna(full_dupe['Top'])]
                dupe_mid = [x.strip() for x in str(full_dupe['Middle']).split(',') if pd.notna(full_dupe['Middle'])]
                dupe_base = [x.strip() for x in str(full_dupe['Base']).split(',') if pd.notna(full_dupe['Base'])]
                
                # Highlight overlapping notes with luxury perfume
                overlap_html = []
                if base_perfume is not None:
                    lux_notes = set(base_perfume['Top'] + base_perfume['Middle'] + base_perfume['Base'])
                    
                    for layer_name, notes_list, color_class in [("Top", dupe_top, "tier-top"), ("Mid", dupe_mid, "tier-mid"), ("Base", dupe_base, "tier-base")]:
                        for note in notes_list[:4]:
                            is_overlap = False
                            for ln in lux_notes:
                                if note.strip().lower() in ln or ln in note.strip().lower():
                                    is_overlap = True
                                    break
                            
                            style_overlap = "color: #d4af37; font-weight: 500;" if is_overlap else "color: #8c8c9e;"
                            overlap_html.append(f"<span style='font-size: 13.5px; margin-right: 12px; display: inline-block; {style_overlap}'>• {note}</span>")
                
                acc_list = [full_dupe[f'mainaccord{k}'] for k in range(1, 6) if pd.notna(full_dupe[f'mainaccord{k}'])]
                accords_html = "".join([f"<span class='accord-text'>• {a.upper()}</span>" for a in acc_list])
                
                card_content = clean_html(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 22px; font-weight: 500;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 11px; color: #f3d060; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).upper()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 4px;">
                            <span class="rating-pill" style="font-size: 13px;">{row['Similarity (%)']:.1f}% Match</span>
                            <span class="gender-pill">{float(row['Rating Value']):.2f} Rating ({row['Gender']})</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="font-family: 'Outfit', sans-serif; font-size: 10px; color: #717188; margin-bottom: 8px; font-weight:600; text-transform: uppercase; letter-spacing: 1px;">Olfactive Overlap (Gold denotes exact luxury elements):</div>
                        <div style="margin-bottom: 12px; line-height: 1.6;">
                            {"".join(overlap_html) if overlap_html else '<span style="font-size:12px; color:#64748b;">N/A</span>'}
                        </div>
                        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #111116;">
                            {accords_html}
                        </div>
                    </div>
                </div>
                """)
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.error(f"Could not locate matching perfumes for '{query_search}'. Please refine your search entry.")

# Page 4: Molecular Odor Predictor
elif menu == "Molecular Odor Predictor":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px; font-weight: 300; text-transform: uppercase; letter-spacing: 1px;'>Molecular Odor Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #717188; margin-bottom: 25px;'>Decode structural SMILES formulas to predict chemical odor categories.</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Molecular Geometry Featurization</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.7; color: #a1a1b5; font-weight: 300;">
                We parse your organic SMILES strings in real-time via RDKit, generating 1,800+ structural, electrostatic, and topological graph descriptors using Mordred. After collinearity pruning, the top 250 mutual information descriptors compile a Random Forest Classifier to output primary raw odor classifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Examples selection
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 11px; color: #d4af37; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;'>Common Molecules</h4>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    selected_smiles = "COC1=CC=C(C=C1)C=O" # Vanillin default
    
    if col_m1.button("Vanillin (Vanilla)"):
        selected_smiles = "COC1=CC=C(C=C1)C=O"
    if col_m2.button("Limonene (Citrus)"):
        selected_smiles = "C=C(C)CCC1=CC=C(C=C1)C"
    if col_m3.button("Menthol (Mint)"):
        selected_smiles = "CC(C)C1CCC(C)CC1O"
    if col_m4.button("Phenylethyl Alcohol (Rose)"):
        selected_smiles = "C1=CC=C(C=C1)CCO"
    if col_m5.button("Alpha-Pinene (Pine)"):
        selected_smiles = "CC1=CCC2CC1C2(C)C"
        
    input_smiles = st.text_input("Or input a custom molecular SMILES string:", selected_smiles)
    
    if input_smiles:
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 14px; color: #d4af37; text-transform: uppercase; letter-spacing: 1px; margin-bottom:12px;'>Molecular Topology</h4>", unsafe_allow_html=True)
            svg_text = render_molecule_svg(input_smiles)
            if svg_text:
                st.markdown(f"<div class='mol-container'>{svg_text}</div>", unsafe_allow_html=True)
            else:
                st.error("Invalid SMILES structure. Please verify your organic chemical formula.")
                
        with col_pred2:
            st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 14px; color: #d4af37; text-transform: uppercase; letter-spacing: 1px; margin-bottom:12px;'>Odor Classifier Output</h4>", unsafe_allow_html=True)
            
            try:
                with st.spinner("Computing high-dimensional descriptors..."):
                    predicted_class = predict_smiles(input_smiles, models_dir="models")
                    
                st.markdown(
                    f"""
                    <div class="glass-card" style="border: 1px solid #22222d;">
                        <h5 style="margin: 0px 0px 8px 0px; font-family: 'Outfit', sans-serif; color:#717188; font-size: 10px; text-transform: uppercase; letter-spacing: 1px;">Predicted Classification</h5>
                        <div style="font-family: 'Cormorant Garamond', serif; font-size: 38px; font-weight: 300; color: #f3d060; text-transform: uppercase; margin-bottom:15px; letter-spacing: 1px;">
                            {predicted_class}
                        </div>
                        <p style="margin:0px; font-size:13.5px; color:#cbd5e1; line-height:1.7; font-weight: 300;">
                            Our random forest classifier analyzed the 250 high-impact topological and electrostatic molecular descriptors computed for this molecule. The predicted odor classification represents the strongest olfactory group associated with this chemical's specific geometry and orbital properties.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                mol_obj = Chem.MolFromSmiles(input_smiles)
                if mol_obj:
                    st.markdown("<h5 style='font-family: \"Outfit\", sans-serif; font-size: 13px; color: #cbd5e1; margin-top:15px; text-transform: uppercase; letter-spacing: 1px;'>Properties</h5>", unsafe_allow_html=True)
                    st.markdown(f"- **Formula:** `{Chem.rdMolDescriptors.CalcMolFormula(mol_obj)}`")
                    st.markdown(f"- **Mass:** `{Chem.rdMolDescriptors.CalcExactMolWt(mol_obj):.2f} g/mol`")
                    st.markdown(f"- **Heavy Atoms:** `{mol_obj.GetNumHeavyAtoms()}`")
                    st.markdown(f"- **Rotatable Bonds:** `{Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_obj)}`")
                    
            except Exception as e:
                st.exception(e)

# Page 5: ML Dashboard & Features
elif menu == "ML Dashboard & Features":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px; font-weight: 300; text-transform: uppercase; letter-spacing: 1px;'>ML Dashboard & Features</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #717188; margin-bottom: 25px;'>Analyze feature weighting metrics and execute model retraining loops.</p>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        st.markdown("<h3 style='color: #f3d060; font-size: 22px; font-weight: 300; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>Topological Descriptor Weightings</h3>", unsafe_allow_html=True)
        st.markdown("Top 15 molecular descriptors ranked by Mutual Information Score in classifying chemical odors.")
        
        # Load feature importance
        if os.path.exists("data/feature_importance.csv"):
            importance_df = pd.read_csv("data/feature_importance.csv")
            top_importance = importance_df.head(15)
            
            chart = alt.Chart(top_importance).mark_bar(
                cornerRadiusTopRight=0,
                cornerRadiusBottomRight=0,
                color="#b28b12" # Solid gold bars
            ).encode(
                x=alt.X('mi_score:Q', title='Mutual Information Score'),
                y=alt.Y('feature:N', sort='-x', title='Mordred Descriptor'),
                tooltip=['feature', 'mi_score']
            ).properties(
                height=450
            ).configure_axis(
                labelColor='#717188',
                titleColor='#a1a1b5',
                grid=False
            ).configure_view(
                strokeOpacity=0
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No feature importance data found. Please run the training pipeline first.")
            
    with col_d2:
        st.markdown("<h3 style='color: #f3d060; font-size: 22px; font-weight: 300; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>Hyperparameters</h3>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="glass-card">
                <h5 style="margin-top: 0px; font-family: 'Outfit', sans-serif; color: #f3d060; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Odor Estimator Profile</h5>
                <p style="margin-bottom: 8px; font-size:13.5px; font-weight: 300;"><b>Classifier:</b> Random Forest Classifier</p>
                <p style="margin-bottom: 8px; font-size:13.5px; font-weight: 300;"><b>Trees:</b> 200 Estimators</p>
                <p style="margin-bottom: 8px; font-size:13.5px; font-weight: 300;"><b>Selected Features:</b> 250 Descriptors</p>
                <p style="margin-bottom: 0px; font-size:13.5px; font-weight: 300;"><b>Cross-Val Accuracy:</b> 60.0%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<h3 style='color: #f3d060; font-size: 22px; font-weight: 300; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>Retraining Loop</h3>", unsafe_allow_html=True)
        st.markdown("Initiate a comprehensive end-to-end retraining cycle over the olfactive chemistry dataset. This recomputes descriptors, executes mutual information filtering, and fits a clean estimator.")
        
        if st.button("Execute Retraining"):
            with st.spinner("Executing optimization pipeline..."):
                try:
                    FINAL_DATASET_CSV = "data/final_dataset.csv"
                    train_model(FINAL_DATASET_CSV)
                    st.success("ML Pipeline Retrained and Saved Successfully!")
                    st.toast("Model saved to models/odor_model.pkl")
                except Exception as e:
                    st.error(f"Retraining failed: {e}")
                    
        st.markdown("<h3 style='color: #f3d060; font-size: 20px; font-weight: 300; margin-top: 25px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;'>Glossary</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            - **`MW`**: Exact compound molecular weight.
            - **`LogP`**: Octanol-water partition coefficient representing lipophilicity.
            - **`nHBDon` / `nHBAcc`**: Absolute counts of Hydrogen Bond Donors & Acceptors.
            - **`RingCount`**: Total carbon ring formations.
            - **`nAcid` / `nBase`**: Acidic and basic atom centers.
            """
        )
