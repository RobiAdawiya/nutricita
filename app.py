import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np # Import numpy
import folium
from streamlit_folium import folium_static

# --- START: Perbaikan Model & Feature Engineering (Dipindahkan dari evaluate_model.py) ---

# Load data
df = pd.read_csv("nutrition2.csv")

# Tentukan kolom numerik untuk nutrisi
nutrisi_cols = ['kalori', 'protein', 'lemak', 'karbohidrat']

# --- A. Pembersihan dan Konversi Tipe Data Nutrisi ---
for col in nutrisi_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0) # Isi NaN dengan 0

# --- B. One-Hot Encode Kolom 'Kategori' ---
df['Kategori'] = df['Kategori'].fillna('Unknown') # Isi NaN di kategori jika ada
df_encoded = pd.get_dummies(df, columns=['Kategori'], prefix='Kategori')

# --- C. Tentukan Fitur untuk Similarity dan Lakukan Normalisasi ---
features_for_similarity = nutrisi_cols + [col for col in df_encoded.columns if 'Kategori_' in col]
features_for_similarity = [col for col in features_for_similarity if col in df_encoded.columns]

df_features = df_encoded[features_for_similarity].copy()

# Normalisasi fitur menggunakan Min-Max Scaling pada df_features
for col in features_for_similarity:
    current_column = df_features[col].astype(float) # Force to float
    
    min_val = current_column.min()
    max_val = current_column.max()
    
    col_range = max_val - min_val
    
    if not np.isclose(col_range, 0):
        df_features[col] = (current_column - min_val) / col_range
    else:
        df_features[col] = 0

# Final cleanup: Isi NaN yang mungkin tersisa dan ganti inf
df_features = df_features.fillna(0)
df_features = df_features.replace([np.inf, -np.inf], 0)

# Hitung similarity matrix
# similarity_matrix sekarang menggunakan df_features (nutrisi + kategori one-hot)
similarity_matrix = cosine_similarity(df_features)

# --- END: Perbaikan Model & Feature Engineering ---


# Fungsi rekomendasi content-based (tetap sama, karena sekarang menggunakan similarity_matrix yang baru)
def content_based_recommend(food_name, n=5):
    if food_name not in df['nama_makanan'].values:
        return pd.DataFrame()
    
    idx = df[df['nama_makanan'] == food_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    food_indices = [i[0] for i in sim_scores[1:n+1]]
    
    return df.iloc[food_indices]

# Fungsi untuk menerapkan filter kesehatan dan kategori (tetap sama)
def apply_filters(df_to_filter, rules, categories):
    filtered = df_to_filter.copy()

    if 'rendah_kalori' in rules:
        filtered = filtered[filtered['kalori'] < 300]
    if 'tinggi_protein' in rules:
        filtered = filtered[filtered['protein'] > 20]
    if 'rendah_lemak' in rules:
        filtered = filtered[filtered['lemak'] < 10]

    if categories:
        filtered = filtered[filtered['Kategori'].isin(categories)]

    return filtered

# Fungsi rekomendasi hybrid (tetap sama, dengan initial_n_fetch yang lebih agresif)
def hybrid_recommend(food_name=None, rules=None, categories=None, n=5):
    initial_n_fetch = n * 5 # Ambil lebih banyak untuk filtering
    
    if food_name:
        recommendations = content_based_recommend(food_name, initial_n_fetch)
        
        # Fallback jika rekomendasi content-based kosong atau tidak cukup
        if recommendations.empty or len(recommendations) < n:
            recommendations = df.copy() # Gunakan seluruh df untuk fallback
            recommendations = apply_filters(recommendations, rules or [], categories or []).head(n)
        else:
            recommendations = apply_filters(recommendations, rules or [], categories or []).head(n)
            
    else:
        recommendations = apply_filters(df, rules or [], categories or []).head(n)
        
    return recommendations

# --- Streamlit UI ---
st.title("🍽️ NutriCita")

# Input Pengguna
col1, col2 = st.columns(2)
with col1:
    food_name = st.selectbox("Pilih makanan:", df['nama_makanan'].unique())
with col2:
    rules = st.multiselect(
        "Filter Kesehatan:",
        ["rendah_kalori", "tinggi_protein", "rendah_lemak"]
    )

# Input Kategori
categories = st.multiselect(
    "Pilih Kategori Makanan:",
    df['Kategori'].unique()
)

if st.button("Rekomendasi!"):
    if not food_name and not rules and not categories:
        st.warning("Pilih makanan, filter kesehatan, atau kategori untuk mendapatkan rekomendasi.")
    else:
        recommendations = hybrid_recommend(food_name, rules, categories)

        st.markdown("## Rekomendasi Makanan Sehat 🍱")

        if recommendations.empty:
            st.info("Tidak ada rekomendasi yang cocok dengan kriteria Anda.")
        else:
            # Tampilan rekomendasi makanan
            for idx, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(row['gambar'], width=120)
                with col2:
                    st.markdown(f"### {row['nama_makanan']}")
                    st.markdown(f"""
                    - **Kalori:** {row['kalori']:.1f} kcal  
                    - **Protein:** {row['protein']:.1f} g  
                    - **Lemak:** {row['lemak']:.1f} g  
                    - **Karbohidrat:** {row['karbohidrat']:.1f} g
                    - **Kategori:** {row['Kategori']}
                    """)
                st.markdown("---")

            # Bagian PETA
            st.markdown("## Temukan di Sekitar Anda 🗺️")

            m = folium.Map(location=[-7.288119, 112.813501], zoom_start=13)
            sample_locations = [
                {"name": "Restoran A", "lat": -7.290903, "lon": 112.806606},
                {"name": "Warung B", "lat": -7.296172, "lon": 112.800875},
                {"name": "Rumah Makan C", "lat": -7.291508, "lon": 112.804964},
                {"name": "Sentra Wisata Kuliner D", "lat": -7.285000, "lon": 112.810000},
                {"name": "Cafe E", "lat": -7.280000, "lon": 112.815000}
            ]

            for loc in sample_locations:
                folium.Marker(
                    [loc["lat"], loc["lon"]],
                    popup=loc["name"],
                    tooltip=loc["name"]
                ).add_to(m)

            folium_static(m)