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

# Custom premium CSS styling (harmonious dark mode, custom fonts, glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Elegant Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #09090e 0%, #12121a 50%, #1a1525 100%);
        color: #e2e8f0;
    }
    
    /* Header branding */
    .brand-container {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .brand-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-left: 10px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.2);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        margin: 2px;
        letter-spacing: 0.5px;
    }
    .badge-top { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-mid { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-base { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-accord { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }
    
    .rating-pill {
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.3);
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 12px;
    }
    
    .gender-pill {
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 12px;
        text-transform: capitalize;
    }

    /* Links */
    .fragrantica-link {
        color: #f472b6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .fragrantica-link:hover {
        color: #ec4899;
        text-decoration: underline;
    }

    /* Molecules container */
    .mol-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #0d0d12;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
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

# ----------------- SIDEBAR BRANDING & NAVIGATION -----------------

with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <span style="font-size: 32px;">🧪</span>
        <span class="brand-title">SCENTIENT</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-top: -15px; margin-bottom: 25px;'>ML-Powered Fragrance Analysis & Recommendation Platform</p>", unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Navigation")
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
    st.markdown("### 📊 Dataset Quick Stats")
    st.markdown(f"**Total Fragrances:** `{len(df_perfumes):,}`")
    st.markdown(f"**Total Brands:** `{df_perfumes['Brand'].nunique():,}`")
    st.markdown(f"**Genders Covered:** `Unisex, Men, Women`")
    
    st.markdown("---")
    st.markdown("<div style='font-size: 11px; color: #64748b;'>Developed by Muhammad Ali • Built with RDKit, Mordred, scikit-learn & Streamlit</div>", unsafe_allow_html=True)

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
        opts.backgroundColour = (0.05, 0.05, 0.08, 1.0) # Match dark card style
        opts.legendFontSize = 13
        opts.annotationFontSize = 13
        opts.multipleBondOffset = 0.15
        opts.bondThickness = 2.5
        
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
    st.markdown("<h1 style='margin-bottom: 0px;'>🏠 Home & Scent Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Explore metadata and notes across 24,000+ luxury, designer, and niche fragrances</p>", unsafe_allow_html=True)
    
    # Welcome banner
    st.markdown(
        """
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);">
            <h3 style="color: #ec4899; margin-top: 0px;">Welcome to Scentient!</h3>
            <p style="margin-bottom: 0px; font-size: 14.5px; line-height: 1.6;">
                Scentient bridges the gap between molecular chemistry and natural language scent experiences. 
                Using high-dimensional molecular descriptors computed via <b>RDKit/Mordred</b>, machine learning odor models, 
                and advanced text retrieval algorithms, Scentient allows you to analyze molecules, perform semantic 
                scent queries, and discover budget-friendly alternatives to luxury fragrances.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Scent Discovery Filters
    st.markdown("### 🔎 Browse & Filter Fragrances")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender_filter = st.selectbox("Target Demography", ["All", "Unisex", "Men", "Women"])
    with col2:
        rating_filter = st.slider("Minimum Rating Value", 0.0, 5.0, 3.5, 0.1)
    with col3:
        search_brand = st.text_input("Filter by Brand (e.g. Creed, Tom Ford)", "")
    with col4:
        search_note = st.text_input("Contains Specific Note (e.g. Vanilla, Oud)", "")
        
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
        
    st.markdown(f"**Found `{len(filtered_df):,}` matching perfumes**")
    
    # Paginate results for clean loading
    limit = 12
    pages = max(1, int(np.ceil(len(filtered_df) / limit)))
    page_num = st.number_input("Page", min_value=1, max_value=pages, step=1, value=1)
    
    start_idx = (page_num - 1) * limit
    end_idx = start_idx + limit
    display_df = filtered_df.iloc[start_idx:end_idx]
    
    # Display Grid
    if not display_df.empty:
        # Create columns
        col_grid1, col_grid2 = st.columns(2)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            target_col = col_grid1 if idx % 2 == 0 else col_grid2
            
            top_notes = [x.strip() for x in str(row['Top']).split(',') if pd.notna(row['Top'])][:4]
            mid_notes = [x.strip() for x in str(row['Middle']).split(',') if pd.notna(row['Middle'])][:4]
            base_notes = [x.strip() for x in str(row['Base']).split(',') if pd.notna(row['Base'])][:4]
            
            accord_list = [row[f'mainaccord{i}'] for i in range(1, 6) if pd.notna(row[f'mainaccord{i}'])]
            
            top_html = "".join([f"<span class='badge badge-top'>{n}</span>" for n in top_notes])
            mid_html = "".join([f"<span class='badge badge-mid'>{n}</span>" for n in mid_notes])
            base_html = "".join([f"<span class='badge badge-base'>{n}</span>" for n in base_notes])
            accords_html = "".join([f"<span class='badge badge-accord'>{a}</span>" for a in accord_list])
            
            card_content = f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <h4 style="margin: 0px 0px 4px 0px; font-size: 18px;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                        <p style="margin: 0px; font-size: 13px; color: #a78bfa; font-weight: 500;">by {str(row['Brand']).title()}</p>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="gender-pill">{row['Gender']}</span>
                        <span class="rating-pill">⭐ {row['Rating Value']:.2f}</span>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <div style="margin-bottom: 8px;"><span style="font-size:12px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Top:</span> {top_html if top_html else '<span style="font-size:12px; color:#64748b;">N/A</span>'}</div>
                    <div style="margin-bottom: 8px;"><span style="font-size:12px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Middle:</span> {mid_html if mid_html else '<span style="font-size:12px; color:#64748b;">N/A</span>'}</div>
                    <div style="margin-bottom: 8px;"><span style="font-size:12px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Base:</span> {base_html if base_html else '<span style="font-size:12px; color:#64748b;">N/A</span>'}</div>
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                        {accords_html}
                    </div>
                </div>
            </div>
            """
            target_col.markdown(card_content, unsafe_allow_html=True)
    else:
        st.info("No fragrances match your filter parameters. Try expanding your search criteria!")

# Page 2: Semantic Scent Search
elif menu == "🔍 Semantic Scent Search":
    st.markdown("<h1 style='margin-bottom: 0px;'>🔍 Semantic Scent Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Search the database using rich natural descriptions and notes</p>", unsafe_allow_html=True)
    
    vectorizer, tfidf_matrix = load_search_engine(df_perfumes)
    
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 25px;">
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.5; color: #cbd5e1;">
                💡 <b>How it works:</b> We construct an NLP text corpus representing the exact olfactive pyramid 
                (Top, Mid, Base notes) and major accords of each fragrance. 
                Using a <b>TF-IDF Vectorizer</b>, we convert your search query and the perfumes into TF-IDF vector representations, 
                then compute their <b>Cosine Similarity</b> to retrieve the closest contextual matches!
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Describe your perfect scent (e.g. 'sweet chocolate coffee winter perfume with vanilla base')", "fresh citrus summer sporty bergamot lemon")
    with col_s2:
        s_gender = st.selectbox("Filter Gender", ["All", "Unisex", "Men", "Women"])
        
    num_results = st.slider("Number of results to retrieve", 5, 50, 10, 5)
    
    if query:
        from sklearn.metrics.pairwise import cosine_similarity
        
        st.markdown(f"### 🎯 Best Matches for: *\"{query}\"*")
        
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
                
                top_html = "".join([f"<span class='badge badge-top'>{n}</span>" for n in top_notes])
                mid_html = "".join([f"<span class='badge badge-mid'>{n}</span>" for n in mid_notes])
                base_html = "".join([f"<span class='badge badge-base'>{n}</span>" for n in base_notes])
                accords_html = "".join([f"<span class='badge badge-accord'>{a}</span>" for a in accord_list])
                
                card_content = f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 18px;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-size: 13px; color: #a78bfa; font-weight: 500;">by {str(row['Brand']).title()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 4px;">
                            <span class="rating-pill" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">⚡ {row['Similarity']:.1f}% Match</span>
                            <span class="gender-pill" style="font-size: 10px;">⭐ {row['Rating Value']:.2f} ({row['Gender']})</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="margin-bottom: 6px;"><span style="font-size:11px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Top:</span> {top_html if top_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
                        <div style="margin-bottom: 6px;"><span style="font-size:11px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Middle:</span> {mid_html if mid_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
                        <div style="margin-bottom: 6px;"><span style="font-size:11px; font-weight:600; color:#94a3b8; display:inline-block; width:50px;">Base:</span> {base_html if base_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}</div>
                        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05);">
                            {accords_html}
                        </div>
                    </div>
                </div>
                """
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.warning("No matches found. Try entering alternative scent descriptors!")

# Page 3: Dupe & Alternative Discovery
elif menu == "💎 Dupe & Alternative Discovery":
    st.markdown("<h1 style='margin-bottom: 0px;'>💎 Dupe & Alternative Discovery</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Find highly similar affordable alternatives to your favorite luxury and niche scents</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 25px;">
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.5; color: #cbd5e1;">
                📊 <b>Custom Comparison Algorithm:</b> We analyze Note Rarity (rare notes carry higher logarithmic weights), 
                compute cross-layer penalties (notes matching Top-to-Top score 100%, while Top-to-Base cross matches are scaled to 60%), 
                and apply specialized heuristics (flanker penalties, clone-house boosts like Lattafa/Armaf/Afnan, and gender demographic alignment).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Default high-end luxury searches
    st.markdown("### 🌟 Quick Selections (Luxury Favorites)")
    
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
        
    query_search = st.text_input("Or search custom luxury perfume name:", query_dupe)
    
    if query_search and dupe_finder:
        # Perform match
        with st.spinner("Finding matches..."):
            dupe_df = dupe_finder.find_dupes(query_search, top_n=10)
            
        if dupe_df is not None and not dupe_df.empty:
            # Re-fetch the luxury base perfume details
            # To find matching index
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
                    <div style="background: rgba(167, 139, 250, 0.05); border: 1px dashed rgba(167, 139, 250, 0.3); border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                        <h4 style="margin: 0px 0px 4px 0px; color:#c084fc;">Target Luxury Perfume matched: {str(base_perfume['Perfume']).replace("-", " ").title()} by {str(base_perfume['Brand']).title()}</h4>
                        <p style="margin:0px; font-size:13px; color:#94a3b8;"><b>Target Gender:</b> {base_perfume['Gender']} | <b>Rating:</b> ⭐ {float(base_perfume['Rating Value']):.2f} | <b>Year:</b> {int(base_perfume['Year']) if pd.notna(base_perfume['Year']) else 'N/A'}</p>
                        <p style="margin:5px 0px 0px 0px; font-size:13px; color:#cbd5e1;"><b>Notes Profile:</b> Top ({', '.join(base_perfume['Top'])}) • Mid ({', '.join(base_perfume['Middle'])}) • Base ({', '.join(base_perfume['Base'])})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("### 🏆 Top Affordable Alternatives")
            
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
                    normalized_lux_notes = [dupe_finder.rarity_weights.get(n, 1.0) for n in lux_notes] # normalized
                    
                    # Check overlap (case-insensitive & synonym mapping)
                    for layer_name, notes_list, b_class in [("Top", dupe_top, "badge-top"), ("Mid", dupe_mid, "badge-mid"), ("Base", dupe_base, "badge-base")]:
                        for note in notes_list[:4]:
                            norm_note = dupe_finder.rarity_weights.get(note.strip().lower(), 1.0) # check synonym mapping
                            # For simple display, let's check exact or partial overlap
                            is_overlap = False
                            for ln in lux_notes:
                                if note.strip().lower() in ln or ln in note.strip().lower():
                                    is_overlap = True
                                    break
                            
                            style_border = "border: 2px solid #a78bfa;" if is_overlap else ""
                            overlap_html.append(f"<span class='badge {b_class}' style='{style_border}'>{note}</span>")
                
                acc_list = [full_dupe[f'mainaccord{k}'] for k in range(1, 6) if pd.notna(full_dupe[f'mainaccord{k}'])]
                accords_html = "".join([f"<span class='badge badge-accord'>{a}</span>" for a in acc_list])
                
                card_content = f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h4 style="margin: 0px 0px 4px 0px; font-size: 18px;"><a href="{row['url']}" target="_blank" class="fragrantica-link">{str(row['Perfume']).replace("-", " ").title()}</a></h4>
                            <p style="margin: 0px; font-size: 13px; color: #ec4899; font-weight: 600;">by {str(row['Brand']).upper()}</p>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: end; gap: 4px;">
                            <span class="rating-pill" style="font-size: 14px; background: rgba(167, 139, 250, 0.15); color: #c084fc; border: 1px solid rgba(167, 139, 250, 0.3);">💎 {row['Similarity (%)']:.1f}% Dupe</span>
                            <span class="gender-pill" style="font-size: 10px;">⭐ {float(row['Rating Value']):.2f} ({row['Gender']})</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="font-size: 11px; color: #a78bfa; margin-bottom: 6px; font-weight:600;">NOTE COMPONENT MATCHES (Highlighted matches luxury base):</div>
                        <div style="margin-bottom: 10px;">
                            {"".join(overlap_html) if overlap_html else '<span style="font-size:11px; color:#64748b;">N/A</span>'}
                        </div>
                        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05);">
                            {accords_html}
                        </div>
                    </div>
                </div>
                """
                target_col.markdown(card_content, unsafe_allow_html=True)
        else:
            st.error(f"Could not find matching perfume for '{query_search}'. Please try another search term.")

# Page 4: Molecular Odor Predictor
elif menu == "🔬 Molecular Odor Predictor":
    st.markdown("<h1 style='margin-bottom: 0px;'>🔬 Molecular Odor Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Input molecular structures (SMILES) to predict odor profiles using scikit-learn random forests</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 25px;">
            <p style="margin: 0px; font-size: 13.5px; line-height: 1.5; color: #cbd5e1;">
                🧪 <b>The Molecular Pipeline:</b> We parse SMILES structures dynamically via <b>RDKit</b>. 
                Then, <b>Mordred</b> evaluates 1,800+ topological, geometric, and electrostatic descriptors. 
                Constant and highly-correlated columns are removed via our feature selection pipeline, leaving the 
                <b>top 250 molecular descriptors</b> (mutual information ranked). 
                Finally, a trained <b>Random Forest Classifier</b> predicts the olfactive classification (e.g. floral, sweet, menthol).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Examples selection
    st.markdown("### 💡 Select a Common Molecule to Test")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    selected_smiles = "COC1=CC=C(C=C1)C=O" # Vanillin default
    
    if col_m1.button("🧁 Vanillin (Sweet / Vanilla)"):
        selected_smiles = "COC1=CC=C(C=C1)C=O"
    if col_m2.button("🍋 Limonene (Citrus / Lemon)"):
        selected_smiles = "C=C(C)CCC1=CC=C(C=C1)C"
    if col_m3.button("🌿 Menthol (Minty / Menthol)"):
        selected_smiles = "CC(C)C1CCC(C)CC1O"
    if col_m4.button("🌹 Phenylethyl alcohol (Rose Floral)"):
        selected_smiles = "C1=CC=C(C=C1)CCO"
    if col_m5.button("🌲 Alpha-pinene (Piney / Woody)"):
        selected_smiles = "CC1=CCC2CC1C2(C)C"
        
    input_smiles = st.text_input("Enter a custom molecular SMILES string:", selected_smiles)
    
    if input_smiles:
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            st.markdown("#### 🔬 Dynamic 2D Chemical Drawing")
            svg_text = render_molecule_svg(input_smiles)
            if svg_text:
                st.markdown(f"<div class='mol-container'>{svg_text}</div>", unsafe_allow_html=True)
            else:
                st.error("Invalid SMILES structure. Please provide a valid chemical SMILES string.")
                
        with col_pred2:
            st.markdown("#### 🧠 AI Scent Profile Prediction")
            
            # Run prediction pipeline
            try:
                with st.spinner("Featurizing molecule and running Random Forest prediction..."):
                    predicted_class = predict_smiles(input_smiles, models_dir="models")
                    
                # Show elegant prediction card
                st.markdown(
                    f"""
                    <div class="glass-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); margin-top:10px;">
                        <h4 style="margin: 0px 0px 10px 0px; color:#34d399;">SCENT PROFILE RESULT</h4>
                        <div style="font-size: 32px; font-weight: 700; color: #ffffff; text-transform: uppercase; margin-bottom:15px;">
                            ✨ {predicted_class}
                        </div>
                        <p style="margin:0px; font-size:13px; color:#cbd5e1; line-height:1.6;">
                            Our random forest classifier analyzed the 250 high-impact Mordred descriptors computed 
                            for this molecule. The predicted odor classification represents the strongest olfactory group 
                            associated with this specific chemical structure.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Fetch and show molecular details
                mol_obj = Chem.MolFromSmiles(input_smiles)
                if mol_obj:
                    st.markdown("##### 🧬 Molecular Statistics")
                    st.markdown(f"- **Chemical Formula:** `{Chem.rdMolDescriptors.CalcMolFormula(mol_obj)}`")
                    st.markdown(f"- **Molecular Weight:** `{Chem.rdMolDescriptors.CalcExactMolWt(mol_obj):.2f} g/mol`")
                    st.markdown(f"- **Heavy Atom Count:** `{mol_obj.GetNumHeavyAtoms()}`")
                    st.markdown(f"- **Rotatable Bonds:** `{Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_obj)}`")
                    
            except Exception as e:
                st.exception(e)

# Page 5: ML Dashboard & Features
elif menu == "📊 ML Dashboard & Features":
    st.markdown("<h1 style='margin-bottom: 0px;'>📊 ML Dashboard & Feature Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Analyze model performance, classification reports, and high-impact molecular descriptors</p>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        st.markdown("### 📈 Molecular Feature Ranking")
        st.markdown("This chart visualizes the top **15 molecular descriptors** ranked by **Mutual Information Score** in classifying fragrance odors.")
        
        # Load feature importance
        if os.path.exists("data/feature_importance.csv"):
            importance_df = pd.read_csv("data/feature_importance.csv")
            top_importance = importance_df.head(15)
            
            chart = alt.Chart(top_importance).mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
                color="#8b5cf6"
            ).encode(
                x=alt.X('mi_score:Q', title='Mutual Information Score'),
                y=alt.Y('feature:N', sort='-x', title='Mordred Descriptor'),
                tooltip=['feature', 'mi_score']
            ).properties(
                height=450
            ).configure_axis(
                labelColor='#cbd5e1',
                titleColor='#cbd5e1'
            ).configure_view(
                strokeOpacity=0
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No feature importance data found. Please run the training pipeline first.")
            
    with col_d2:
        st.markdown("### ⚙️ Pipeline Management")
        
        st.markdown(
            """
            <div class="glass-card" style="margin-bottom: 20px;">
                <h5 style="margin-top: 0px; color: #a78bfa;">Model Properties</h5>
                <p style="margin-bottom: 5px; font-size:13px;"><b>Algorithm:</b> Random Forest Classifier</p>
                <p style="margin-bottom: 5px; font-size:13px;"><b>Estimators:</b> 200 Decision Trees</p>
                <p style="margin-bottom: 5px; font-size:13px;"><b>Top Features Kept:</b> 250 Descriptors</p>
                <p style="margin-bottom: 0px; font-size:13px;"><b>Evaluation Accuracy:</b> 60.0%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### 🔄 Dynamic Retraining Engine")
        st.markdown("Trigger an end-to-end retraining cycle. This will re-featurize the chemical training data, run correlation pruning, select top descriptors, and fit a new Random Forest model.")
        
        if st.button("🚀 Trigger Model Retraining"):
            with st.spinner("Retraining model... Please wait (takes ~30-45 seconds)"):
                try:
                    # Run training pipeline
                    FINAL_DATASET_CSV = "data/final_dataset.csv"
                    train_model(FINAL_DATASET_CSV)
                    st.success("🎉 ML Pipeline Retrained and Saved Successfully!")
                    st.toast("Model saved to models/odor_model.pkl")
                except Exception as e:
                    st.error(f"Retraining failed: {e}")
                    
        # Feature Explanations
        st.markdown("### 📚 Mordred Feature Glossary")
        st.markdown(
            """
            - **`MW`**: Molecular weight of the molecule.
            - **`LogP`**: Octanol-water partition coefficient (represents lipophilicity).
            - **`nHBDon` / `nHBAcc`**: Number of Hydrogen Bond Donors and Acceptors.
            - **`RingCount`**: Number of carbon/hetero rings present.
            - **`nAcid` / `nBase`**: Acidic/basic atom counts.
            """
        )
