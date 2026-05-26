import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import altair as alt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Ensure src/ is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dupe_finder import DupeFinder
from predict import predict_smiles, featurize_smiles_in_memory
from train_model import train_model

# Page config
st.set_page_config(
    page_title="Scentient - AI Fragrance Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (luxury gold/obsidian theme, custom fonts, overrides)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .brand-title, .main-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .sub-title, .section-title, .widget-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Elegant Dark Obsidian Background */
    .stApp {
        background: #06050a;
        color: #e2e8f0;
    }
    
    /* Floating Ambient Glow Orbs */
    .ambient-glow-1 {
        position: fixed;
        top: -15%;
        left: -15%;
        width: 50vw;
        height: 50vh;
        background: radial-gradient(circle, rgba(212, 175, 55, 0.05) 0%, rgba(139, 92, 246, 0.02) 50%, rgba(0,0,0,0) 80%);
        z-index: -1;
        filter: blur(100px);
        pointer-events: none;
        animation: pulseGlow1 15s ease-in-out infinite alternate;
    }
    .ambient-glow-2 {
        position: fixed;
        bottom: -15%;
        right: -15%;
        width: 55vw;
        height: 55vh;
        background: radial-gradient(circle, rgba(244, 63, 94, 0.03) 0%, rgba(212, 175, 55, 0.02) 50%, rgba(0,0,0,0) 80%);
        z-index: -1;
        filter: blur(120px);
        pointer-events: none;
        animation: pulseGlow2 20s ease-in-out infinite alternate;
    }
    @keyframes pulseGlow1 {
        0% { transform: scale(1) translate(0px, 0px); }
        100% { transform: scale(1.15) translate(40px, -30px); }
    }
    @keyframes pulseGlow2 {
        0% { transform: scale(1) translate(0px, 0px); }
        100% { transform: scale(1.1) translate(-40px, 30px); }
    }

    /* Streamlit UI Custom Styling Overrides */
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #030305 !important;
        border-right: 1px solid rgba(212, 175, 55, 0.08) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(212, 175, 55, 0.08) !important;
    }
    
    /* Navigation Radio Items as Custom Buttons */
    div[data-testid="stSidebar"] div.stRadio > div {
        background-color: transparent !important;
        padding: 5px !important;
    }
    div[data-testid="stSidebar"] div.stRadio label {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        color: #94a3b8 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    div[data-testid="stSidebar"] div.stRadio label:hover {
        background: rgba(212, 175, 55, 0.05) !important;
        border-color: rgba(212, 175, 55, 0.2) !important;
        color: #f3d060 !important;
        transform: translateX(4px) !important;
    }
    div[data-testid="stSidebar"] div.stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
        border-color: rgba(212, 175, 55, 0.35) !important;
        color: #f3d060 !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.06) !important;
    }
    div[data-testid="stSidebar"] div.stRadio label > div:first-child {
        display: none !important; /* Hide native radio circles */
    }
    
    /* Inputs, selectboxes, and number input boxes overrides */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: rgba(13, 12, 18, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    div[data-testid="stNumberInput"] input:hover {
        border-color: rgba(212, 175, 55, 0.25) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: rgba(212, 175, 55, 0.5) !important;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.15) !important;
    }
    
    /* Elegant Custom Gold Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #aa820a 100%) !important;
        color: #07070a !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.1) !important;
        text-transform: uppercase;
        font-size: 12px !important;
        letter-spacing: 0.8px !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.25), 0 0 15px rgba(212, 175, 55, 0.1) !important;
        color: #000000 !important;
        border-color: #f3d060 !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Override standard red Streamlit Slider */
    div.stSlider > div {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    div.stSlider [data-testid="stThumb"] {
        background-color: #d4af37 !important;
        border: 2px solid #050408 !important;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.4) !important;
    }
    div.stSlider [style*="background-color: rgb(255, 75, 75)"] {
        background-color: #d4af37 !important;
    }
    div.stSlider [style*="background-color: rgb(244, 63, 94)"] {
        background-color: #d4af37 !important;
    }
    
    /* Header branding */
    .brand-container {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        padding: 16px;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border-radius: 16px;
        border: 1px solid rgba(212, 175, 55, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(90deg, #f3d060 0%, #d4af37 50%, #aa820a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-left: 10px;
        letter-spacing: 2px;
    }
    
    /* Luxury Cards */
    .glass-card {
        background: rgba(13, 12, 18, 0.45);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.6);
        margin-bottom: 24px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.03) 0%, rgba(255, 255, 255, 0) 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(212, 175, 55, 0.18);
        box-shadow: 0 20px 50px -15px rgba(212, 175, 55, 0.08), 0 0 30px -5px rgba(0, 0, 0, 0.6);
    }
    .glass-card:hover::before {
        opacity: 1;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        margin: 3px;
        letter-spacing: 0.5px;
        font-family: 'Outfit', sans-serif;
    }
    .badge-top { background: rgba(244, 63, 94, 0.06); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.2); }
    .badge-mid { background: rgba(16, 185, 129, 0.06); color: #a7f3d0; border: 1px solid rgba(16, 185, 129, 0.2); }
    .badge-base { background: rgba(245, 158, 11, 0.06); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-accord { background: rgba(212, 175, 55, 0.06); color: #fde68a; border: 1px solid rgba(212, 175, 55, 0.25); }
    
    .rating-pill {
        background: rgba(212, 175, 55, 0.08);
        color: #f3d060;
        border: 1px solid rgba(212, 175, 55, 0.25);
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-family: 'Outfit', sans-serif;
    }
    
    .gender-pill {
        background: rgba(255, 255, 255, 0.04);
        color: #94a3b8;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        text-transform: capitalize;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-family: 'Outfit', sans-serif;
    }

    /* Links */
    .fragrantica-link {
        color: #cbd5e1;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .fragrantica-link:hover {
        color: #d4af37;
    }

    /* Molecules container */
    .mol-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #050408;
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(212, 175, 55, 0.08);
        box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.7);
    }
    
    /* Olfactory Pyramid Container Styling */
    .pyramid-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 15px;
    }
    .pyramid-tier {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 14px;
        padding: 12px 18px;
        transition: all 0.3s ease;
    }
    .pyramid-tier:hover {
        background: rgba(212, 175, 55, 0.02);
        border-color: rgba(212, 175, 55, 0.1);
    }
    .tier-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .tier-top { color: #fda4af; }
    .tier-mid { color: #a7f3d0; }
    .tier-base { color: #fde68a; }
</style>

<!-- Floating Background Divs -->
<div class="ambient-glow-1"></div>
<div class="ambient-glow-2"></div>
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
    top_html = "".join([f"<span class='badge badge-top'>{n}</span>" for n in top_notes])
    mid_html = "".join([f"<span class='badge badge-mid'>{n}</span>" for n in mid_notes])
    base_html = "".join([f"<span class='badge badge-base'>{n}</span>" for n in base_notes])
    accords_html = "".join([f"<span class='badge badge-accord'>{a}</span>" for a in accords])
    
    return f"""
    <div class="pyramid-container">
        <div class="pyramid-tier">
            <div class="tier-header tier-top">✨ Opening • Top Notes</div>
            <div>{top_html if top_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
        </div>
        <div class="pyramid-tier">
            <div class="tier-header tier-mid">🌿 Heart • Middle Notes</div>
            <div>{mid_html if mid_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
        </div>
        <div class="pyramid-tier">
            <div class="tier-header tier-base">🪵 Anchor • Base Notes</div>
            <div>{base_html if base_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
        </div>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.03);">
            {accords_html}
        </div>
    </div>
    """

# ----------------- SIDEBAR BRANDING & NAVIGATION -----------------

with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <span style="font-size: 30px;">🧪</span>
        <span class="brand-title">SCENTIENT</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-family: \"Outfit\", sans-serif; font-size: 12px; color: #d4af37; margin-top: -18px; margin-bottom: 25px; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 500; text-align: center;'>The Chemistry of Luxury Scent</p>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; color: #e2e8f0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>🗺️ Navigate Platform</h4>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Go to page:",
        [
            "🏠 Home & Discovery Explorer", 
            "🔍 Semantic Scent Search", 
            "💎 Dupe & Alternative Discovery",
            "🔬 Molecular Odor Predictor",
            "📊 ML Dashboard & Features"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; color: #d4af37; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;'>📊 OLFACTIVE REPOSITORY</h4>", unsafe_allow_html=True)
    st.markdown(f"**Total Fragrances:** `{len(df_perfumes):,}`")
    st.markdown(f"**Total Brands:** `{df_perfumes['Brand'].nunique():,}`")
    st.markdown(f"**Demographics:** `Unisex, Pour Homme, Pour Femme`")
    
    st.markdown("---")
    st.markdown("<div style='font-size: 11px; color: #4b4b5c; text-align: center;'>Scentient Platform • Developed by Muhammad Ali<br>Powered by RDKit, Mordred & Random Forests</div>", unsafe_allow_html=True)

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
        opts.backgroundColour = (0.05, 0.04, 0.08, 1.0) # Match dark card style
        opts.legendFontSize = 14
        opts.annotationFontSize = 14
        opts.multipleBondOffset = 0.15
        opts.bondThickness = 3.0
        
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
if menu == "🏠 Home & Discovery Explorer":
    # Stunning editorial split hero layout
    hero_col1, hero_col2 = st.columns([5, 4])
    
    with hero_col1:
        st.markdown("""
        <h1 style="font-size: 60px; font-weight: 700; line-height: 1.1; margin-top: 15px; margin-bottom: 5px; background: linear-gradient(90deg, #f5d060 0%, #d4af37 50%, #aa820a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            SCENTIENT
        </h1>
        <h3 class="sub-title" style="font-size: 16px; font-weight: 400; color: #a6a6bd; margin-top:0px; margin-bottom: 25px; letter-spacing: 3px; text-transform: uppercase;">
            Decoding the Chemistry of Olfaction
        </h3>
        <p style="font-size: 16px; line-height: 1.75; color: #b2b2cb; margin-bottom: 20px;">
            Welcome to the intersection of molecular science and luxury perfumery. Scentient decodes the olfactive universe by bridging structural organic chemistry and sensory language.
        </p>
        <p style="font-size: 15px; line-height: 1.75; color: #9494a8; margin-bottom: 25px;">
            Through compute-dense topological descriptors (<b>RDKit & Mordred</b>), machine learning odor models (<b>Random Forests</b>), and semantic natural language representations, you can catalog compounds, query notes, and discover custom high-fidelity perfume alternatives.
        </p>
        """, unsafe_allow_html=True)
        
    with hero_col2:
        if os.path.exists("luxury_perfume.png"):
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; padding: 10px; background: rgba(13,12,18,0.3); border: 1px solid rgba(212,175,55,0.08); border-radius: 24px; box-shadow: 0 15px 40px rgba(0,0,0,0.5);">
            """, unsafe_allow_html=True)
            st.image("luxury_perfume.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("<hr style='border-color: rgba(212, 175, 55, 0.08); margin: 35px 0px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 32px; color: #f3d060; margin-bottom: 10px;'>🔎 Olfactive Discovery Library</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 30px;'>Explore, filter, and dissect the notes of over 24,000 elite luxury, designer, and boutique perfumes.</p>", unsafe_allow_html=True)
    
    # Scent Discovery Filters
    filter_card = st.container()
    with filter_card:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender_filter = st.selectbox("Olfactive Demography", ["All", "Unisex", "Men", "Women"])
        with col2:
            rating_filter = st.slider("Minimum Quality Score (Stars)", 0.0, 5.0, 3.8, 0.1)
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
        
    st.markdown(f"<p style='font-family: \"Outfit\", sans-serif; font-size: 14px; color: #d4af37; margin-top: 15px;'>Catalogued matches: <b>{len(filtered_df):,}</b> perfumes found</p>", unsafe_allow_html=True)
    
    # Paginate results for clean loading
    limit = 12
    pages = max(1, int(np.ceil(len(filtered_df) / limit)))
    
    col_p1, col_p2 = st.columns([1, 6])
    with col_p1:
        page_num = st.number_input("Page Selector", min_value=1, max_value=pages, step=1, value=1)
    
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
            
            card_content = f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <div>
                        <h4 style="margin: 0px 0px 4px 0px; font-size: 21px; font-weight: 600;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                        <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 13px; color: #d4af37; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).title()}</p>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="gender-pill">{row['Gender']}</span>
                        <span class="rating-pill">⭐ {row['Rating Value']:.2f}</span>
                    </div>
                </div>
                {pyramid_html}
            </div>
            """
            target_col.markdown(card_content, unsafe_allow_html=True)
    else:
        st.info("No fragrances match your filter parameters. Try expanding your search criteria!")

# Page 2: Semantic Scent Search
elif menu == "🔍 Semantic Scent Search":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px;'>🔍 Semantic Scent Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Describe your dream sensory experience to retrieve matching olfactive profiles.</p>", unsafe_allow_html=True)
    
    vectorizer, tfidf_matrix = load_search_engine(df_perfumes)
    
    st.markdown(
        """
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.04) 0%, rgba(255,255,255,0.01) 100%);">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">💡 NLP Weighted Engine</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.6; color: #b2b2cb;">
                We represent each perfume as a structured document capturing its exact olfactive layers (Top, Mid, Base) and principal accords. 
                Using a customized <b>TF-IDF Vectorizer</b>, we convert your rich natural descriptions into high-dimensional keyword representations and execute a <b>Cosine Similarity</b> search against our entire 24,000+ library.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Describe the scent or atmosphere you desire:", "sweet chocolate coffee winter perfume with vanilla base")
    with col_s2:
        s_gender = st.selectbox("Target Demography", ["All", "Unisex", "Men", "Women"])
        
    num_results = st.slider("Results Count", 4, 30, 8, 2)
    
    if query:
        from sklearn.metrics.pairwise import cosine_similarity
        
        st.markdown(f"<h3 style='color: #f3d060; font-size: 24px; margin-top: 25px;'>🎯 Top Semantic Matches for: <i>\"{query}\"</i></h3>", unsafe_allow_html=True)
        
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
                
                card_content = f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 21px; font-weight: 600;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 13px; color: #d4af37; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).title()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 6px;">
                            <span class="rating-pill" style="background: rgba(16, 185, 129, 0.08); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25);">⚡ {row['Similarity']:.1f}% Match</span>
                            <span class="gender-pill" style="font-size: 11px;">⭐ {row['Rating Value']:.2f} ({row['Gender']})</span>
                        </div>
                    </div>
                    {pyramid_html}
                </div>
                """
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.warning("No matches found. Try entering alternative scent descriptors!")

# Page 3: Dupe & Alternative Discovery
elif menu == "💎 Dupe & Alternative Discovery":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px;'>💎 Dupe & Alternative Discovery</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Find highly similar affordable alternatives to your favorite luxury and niche scents</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.04) 0%, rgba(255,255,255,0.01) 100%);">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">📊 Weighted Comparison Heuristics</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.6; color: #b2b2cb;">
                Our discovery algorithm scores notes dynamically: rare components scale higher (TF-IDF weights), cross-tier pairings are penalized (e.g. matching a Top note to a Base note receives a 40% reduction), and specialized clone house boosts (e.g. Lattafa, Armaf, Afnan) prioritize direct budget equivalents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Default high-end luxury searches
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 14px; color: #d4af37; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>🌟 Luxury Favorites Quick Match</h4>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    query_dupe = "baccarat-rouge-540"
    if col_btn1.button("🥃 Creed Aventus"):
        query_dupe = "aventus"
    if col_btn2.button("🔮 Baccarat Rouge 540"):
        query_dupe = "baccarat-rouge-540"
    if col_btn3.button("🍨 Parfums de Marly Althair"):
        query_dupe = "althair"
    if col_btn4.button("🍒 Tom Ford Lost Cherry"):
        query_dupe = "lost-cherry"
        
    query_search = st.text_input("Or input a custom luxury perfume title below:", query_dupe)
    
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
                    <div style="background: rgba(212, 175, 55, 0.03); border: 1px dashed rgba(212, 175, 55, 0.25); border-radius: 16px; padding: 20px; margin-bottom: 30px;">
                        <h4 style="margin: 0px 0px 6px 0px; font-family: 'Cormorant Garamond', serif; font-size: 24px; color: #f3d060; font-weight: 600;">Luxury Perfume Profile: {str(base_perfume['Perfume']).replace("-", " ").title()} by {str(base_perfume['Brand']).title()}</h4>
                        <p style="margin:0px 0px 10px 0px; font-family: 'Outfit', sans-serif; font-size:13px; color:#94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Demography: <b>{base_perfume['Gender']}</b> | Quality Score: <b>⭐ {float(base_perfume['Rating Value']):.2f}</b> | Year: <b>{int(base_perfume['Year']) if pd.notna(base_perfume['Year']) else 'N/A'}</b></p>
                        <p style="margin:0px; font-size:13.5px; color:#cbd5e1; line-height:1.6;"><b>Olfactive Pyramid:</b> Top ({', '.join(base_perfume['Top'])}) • Heart ({', '.join(base_perfume['Middle'])}) • Base ({', '.join(base_perfume['Base'])})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("<h3 style='color: #f3d060; font-size: 26px; margin-bottom: 20px;'>🏆 Ranked Affordable Alternatives</h3>", unsafe_allow_html=True)
            
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
                    
                    for layer_name, notes_list, b_class in [("Top", dupe_top, "badge-top"), ("Mid", dupe_mid, "badge-mid"), ("Base", dupe_base, "badge-base")]:
                        for note in notes_list[:4]:
                            is_overlap = False
                            for ln in lux_notes:
                                if note.strip().lower() in ln or ln in note.strip().lower():
                                    is_overlap = True
                                    break
                            
                            style_border = "border: 2px solid rgba(212, 175, 55, 0.7); box-shadow: 0 0 8px rgba(212, 175, 55, 0.25);" if is_overlap else ""
                            overlap_html.append(f"<span class='badge {b_class}' style='{style_border}'>{note}</span>")
                
                acc_list = [full_dupe[f'mainaccord{k}'] for k in range(1, 6) if pd.notna(full_dupe[f'mainaccord{k}'])]
                accords_html = "".join([f"<span class='badge badge-accord'>{a}</span>" for a in acc_list])
                
                card_content = f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 21px; font-weight: 600;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-family: 'Outfit', sans-serif; font-size: 13px; color: #f3d060; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">{str(row['Brand']).upper()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 6px;">
                            <span class="rating-pill" style="font-size: 13px; background: rgba(212, 175, 55, 0.1); color: #f3d060; border: 1px solid rgba(212, 175, 55, 0.35);">💎 {row['Similarity (%)']:.1f}% Match</span>
                            <span class="gender-pill" style="font-size: 11px;">⭐ {float(row['Rating Value']):.2f} ({row['Gender']})</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="font-family: 'Outfit', sans-serif; font-size: 11px; color: #cbd5e1; margin-bottom: 8px; font-weight:600; text-transform: uppercase; letter-spacing: 0.5px;">Note Matches (Highlighted borders represent luxurious overlap):</div>
                        <div style="margin-bottom: 12px;">
                            {"".join(overlap_html) if overlap_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}
                        </div>
                        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.03);">
                            {accords_html}
                        </div>
                    </div>
                </div>
                """
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.error(f"Could not locate matching perfumes for '{query_search}'. Please refine your search entry.")

# Page 4: Molecular Odor Predictor
elif menu == "🔬 Molecular Odor Predictor":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px;'>🔬 Molecular Odor Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Input a molecular SMILES structure to predict raw olfactive profiles using advanced scikit-learn random forests</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.04) 0%, rgba(255,255,255,0.01) 100%);">
            <h5 style="margin-top: 0px; color: #f3d060; font-family: 'Outfit', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">🔬 The Organic Chemistry Pipeline</h5>
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.6; color: #b2b2cb;">
                We parse your organic SMILES strings in real-time via <b>RDKit</b>. We then map <b>1,800+</b> molecular descriptors using <b>Mordred</b>. After removing zero-variance and multi-collinear columns, the top <b>250 mutual information features</b> feed a trained <b>Random Forest Classifier</b> to predict the chemical's primary olfactory note.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Examples selection
    st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 14px; color: #d4af37; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>💡 Dynamic Molecular Standards</h4>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    selected_smiles = "COC1=CC=C(C=C1)C=O" # Vanillin default
    
    if col_m1.button("🧁 Vanillin (Vanilla)"):
        selected_smiles = "COC1=CC=C(C=C1)C=O"
    if col_m2.button("🍋 Limonene (Citrus)"):
        selected_smiles = "C=C(C)CCC1=CC=C(C=C1)C"
    if col_m3.button("🌿 Menthol (Minty)"):
        selected_smiles = "CC(C)C1CCC(C)CC1O"
    if col_m4.button("🌹 Phenylethyl Alcohol (Rose)"):
        selected_smiles = "C1=CC=C(C=C1)CCO"
    if col_m5.button("🌲 Alpha-Pinene (Pine)"):
        selected_smiles = "CC1=CCC2CC1C2(C)C"
        
    input_smiles = st.text_input("Or enter a custom IUPAC SMILES string:", selected_smiles)
    
    if input_smiles:
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 18px; color: #d4af37; margin-bottom:12px;'>🔬 Dynamic 2D Structural Map</h4>", unsafe_allow_html=True)
            svg_text = render_molecule_svg(input_smiles)
            if svg_text:
                st.markdown(f"<div class='mol-container'>{svg_text}</div>", unsafe_allow_html=True)
            else:
                st.error("Invalid SMILES structure. Please verify your organic chemical formula.")
                
        with col_pred2:
            st.markdown("<h4 style='font-family: \"Outfit\", sans-serif; font-size: 18px; color: #d4af37; margin-bottom:12px;'>🧠 AI Scent Profile Analysis</h4>", unsafe_allow_html=True)
            
            try:
                with st.spinner("Computing high-dimensional descriptors..."):
                    predicted_class = predict_smiles(input_smiles, models_dir="models")
                    
                st.markdown(
                    f"""
                    <div class="glass-card" style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.08) 0%, rgba(139, 92, 246, 0.03) 100%); border: 1px solid rgba(212,175,55,0.2); margin-top:2px;">
                        <h5 style="margin: 0px 0px 8px 0px; font-family: 'Outfit', sans-serif; color:#f3d060; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">CLASSIFICATION RESULT</h5>
                        <div style="font-family: 'Cormorant Garamond', serif; font-size: 38px; font-weight: 700; color: #ffffff; text-transform: capitalize; margin-bottom:15px; letter-spacing: 0.5px;">
                            ✨ {predicted_class}
                        </div>
                        <p style="margin:0px; font-size:13.5px; color:#cbd5e1; line-height:1.65;">
                            Our random forest classifier analyzed the 250 high-impact topological and electrostatic molecular descriptors computed for this molecule. The predicted odor classification represents the strongest olfactory group associated with this chemical's specific geometry and orbital properties.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                mol_obj = Chem.MolFromSmiles(input_smiles)
                if mol_obj:
                    st.markdown("<h5 style='font-family: \"Outfit\", sans-serif; font-size: 15px; color: #e2e8f0; margin-top:15px;'>🧬 Molecular Properties</h5>", unsafe_allow_html=True)
                    st.markdown(f"- **Chemical Formula:** `{Chem.rdMolDescriptors.CalcMolFormula(mol_obj)}`")
                    st.markdown(f"- **Molecular Mass:** `{Chem.rdMolDescriptors.CalcExactMolWt(mol_obj):.2f} g/mol`")
                    st.markdown(f"- **Heavy Atom Count:** `{mol_obj.GetNumHeavyAtoms()}`")
                    st.markdown(f"- **Rotatable Bonds:** `{Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_obj)}`")
                    
            except Exception as e:
                st.exception(e)

# Page 5: ML Dashboard & Features
elif menu == "📊 ML Dashboard & Features":
    st.markdown("<h1 style='margin-bottom: 5px; font-size: 45px;'>📊 ML Dashboard & Feature Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Dissect classifier parameters, topological weight matrices, and execute real-time model retraining.</p>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        st.markdown("<h3 style='color: #f3d060; font-size: 24px; margin-bottom: 15px;'>📈 Topological Descriptor Ranking</h3>", unsafe_allow_html=True)
        st.markdown("Visualizing the top **15 molecular descriptors** ranked by **Mutual Information Score** in classifying chemical odors.")
        
        # Load feature importance
        if os.path.exists("data/feature_importance.csv"):
            importance_df = pd.read_csv("data/feature_importance.csv")
            top_importance = importance_df.head(15)
            
            chart = alt.Chart(top_importance).mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6,
                color="#d4af37" # Gold bar charts!
            ).encode(
                x=alt.X('mi_score:Q', title='Mutual Information Score'),
                y=alt.Y('feature:N', sort='-x', title='Mordred Descriptor'),
                tooltip=['feature', 'mi_score']
            ).properties(
                height=450
            ).configure_axis(
                labelColor='#94a3b8',
                titleColor='#cbd5e1',
                grid=False
            ).configure_view(
                strokeOpacity=0
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No feature importance data found. Please run the training pipeline first.")
            
    with col_d2:
        st.markdown("<h3 style='color: #f3d060; font-size: 24px; margin-bottom: 15px;'>⚙️ Hyperparameters</h3>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="glass-card" style="margin-bottom: 25px;">
                <h5 style="margin-top: 0px; font-family: 'Outfit', sans-serif; color: #f3d060; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Model Configurations</h5>
                <p style="margin-bottom: 6px; font-size:13.5px;"><b>Core Estimator:</b> Random Forest Classifier</p>
                <p style="margin-bottom: 6px; font-size:13.5px;"><b>Decision Trees:</b> 200 Estimators</p>
                <p style="margin-bottom: 6px; font-size:13.5px;"><b>Pruned Descriptors:</b> 250 Top Features</p>
                <p style="margin-bottom: 0px; font-size:13.5px;"><b>Cross-Val Accuracy:</b> 60.0%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<h3 style='color: #f3d060; font-size: 24px; margin-bottom: 15px;'>🔄 Retraining Trigger</h3>", unsafe_allow_html=True)
        st.markdown("Initiate a comprehensive end-to-end retraining cycle over the olfactive chemistry dataset. This recomputes descriptors, executes mutual information filtering, and fits a clean estimator.")
        
        if st.button("🚀 Execute Retraining Loop"):
            with st.spinner("Featurizing compounds and running scikit-learn optimization..."):
                try:
                    FINAL_DATASET_CSV = "data/final_dataset.csv"
                    train_model(FINAL_DATASET_CSV)
                    st.success("🎉 ML Pipeline Retrained and Saved Successfully!")
                    st.toast("Model saved to models/odor_model.pkl")
                except Exception as e:
                    st.error(f"Retraining failed: {e}")
                    
        st.markdown("<h3 style='color: #f3d060; font-size: 22px; margin-top: 25px; margin-bottom: 12px;'>📚 Topological Glossary</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            - **`MW`**: Exact compound molecular weight.
            - **`LogP`**: Octanol-water partition coefficient representing lipophilicity.
            - **`nHBDon` / `nHBAcc`**: Absolute counts of Hydrogen Bond Donors & Acceptors.
            - **`RingCount`**: Total carbon ring formations.
            - **`nAcid` / `nBase`**: Acidic and basic atom centers.
            """
        )
