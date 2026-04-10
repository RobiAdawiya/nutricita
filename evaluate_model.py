import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Load Data & Feature Engineering ---
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
    # Ensure the column is explicitly numeric (float) to prevent boolean issues
    current_column = df_features[col].astype(float) # Force to float
    
    min_val = current_column.min()
    max_val = current_column.max()
    
    col_range = max_val - min_val
    
    # Use np.isclose for floating-point comparison to zero
    if not np.isclose(col_range, 0):
        df_features[col] = (current_column - min_val) / col_range
    else:
        # If column has constant values (range is zero), normalize to 0
        df_features[col] = 0

# Final cleanup: Fill any remaining NaNs and replace infs (less likely now)
df_features = df_features.fillna(0)
df_features = df_features.replace([np.inf, -np.inf], 0)

# Hitung similarity matrix
similarity_matrix = cosine_similarity(df_features)

# --- 2. Fungsi Rekomendasi ---

def content_based_recommend(food_name, n=5):
    if food_name not in df['nama_makanan'].values:
        return pd.DataFrame()
    
    idx = df[df['nama_makanan'] == food_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    food_indices = [i[0] for i in sim_scores[1:n+1]]
    
    return df.iloc[food_indices]

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

def hybrid_recommend(food_name=None, rules=None, categories=None, n=5):
    initial_n_fetch = n * 5 
    
    if food_name:
        recommendations = content_based_recommend(food_name, initial_n_fetch)
        
        if recommendations.empty or len(recommendations) < n:
            recommendations = df.copy() 
            recommendations = apply_filters(recommendations, rules or [], categories or []).head(n)
        else:
            recommendations = apply_filters(recommendations, rules or [], categories or []).head(n)
            
    else:
        recommendations = apply_filters(df, rules or [], categories or []).head(n)
        
    return recommendations

# --- 3. Fungsi Evaluasi Offline ---

def calculate_diversity(recommended_df):
    if recommended_df.shape[0] < 2:
        return 0.0

    recommended_indices = recommended_df.index
    
    # Ensure this subset of features is also explicitly float
    recommended_features_subset = df_features.loc[recommended_indices, features_for_similarity].astype(float).copy()

    sim_matrix = cosine_similarity(recommended_features_subset)
    
    upper_triangular_sum = 0
    count = 0
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            upper_triangular_sum += sim_matrix[i, j]
            count += 1
    
    avg_similarity = upper_triangular_sum / count if count > 0 else 0.0
    
    return 1 - avg_similarity

def run_evaluation(scenarios, n_recommendations=5):
    print("--- Memulai Evaluasi Model Offline ---")
    
    all_recommended_items = set() 
    
    for i, scenario in enumerate(scenarios):
        food_name = scenario.get('food_name')
        rules = scenario.get('rules', [])
        categories = scenario.get('categories', [])
        
        print(f"\n--- Skenario Evaluasi {i+1} ---")
        print(f"  Makanan Input: {food_name if food_name else 'Tidak Ada'}")
        print(f"  Filter Kesehatan: {', '.join(rules) if rules else 'Tidak Ada'}")
        print(f"  Kategori: {', '.join(categories) if categories else 'Tidak Ada'}")

        recommendations = hybrid_recommend(food_name, rules, categories, n_recommendations)
        
        if recommendations.empty:
            print("  Tidak ada rekomendasi untuk skenario ini.")
            continue

        print("  Rekomendasi Dihasilkan:")
        for idx, row in recommendations.iterrows():
            print(f"  - {row['nama_makanan']} (Kalori: {row['kalori']:.1f} kcal, Protein: {row['protein']:.1f} g, Lemak: {row['lemak']:.1f} g, Kategori: {row['Kategori']})")
            all_recommended_items.add(row['nama_makanan'])

        # --- Hitung Metrik Skenario ---
        diversity = calculate_diversity(recommendations)
        
        # Cek Kepatuhan Aturan
        adherence_issues = []
        for _, row in recommendations.iterrows():
            if 'rendah_kalori' in rules and row['kalori'] >= 300:
                adherence_issues.append(f"{row['nama_makanan']} (Kalori: {row['kalori']:.1f}) tidak rendah kalori.")
            if 'tinggi_protein' in rules and row['protein'] <= 20:
                adherence_issues.append(f"{row['nama_makanan']} (Protein: {row['protein']:.1f}) tidak tinggi protein.")
            if 'rendah_lemak' in rules and row['lemak'] >= 10:
                adherence_issues.append(f"{row['nama_makanan']} (Lemak: {row['lemak']:.1f}) tidak rendah lemak.")
            if categories and row['Kategori'] not in categories:
                adherence_issues.append(f"{row['nama_makanan']} (Kategori: {row['Kategori']}) tidak cocok dengan kategori yang dipilih.")
        
        rule_adherence_status = "Semua aturan dipatuhi." if not adherence_issues else "Ada masalah: " + "; ".join(adherence_issues[:3]) + ("..." if len(adherence_issues) > 3 else "")

        print(f"\n  Metrik Skenario:")
        print(f"  - Jumlah Rekomendasi: {len(recommendations)}")
        print(f"  - Keberagaman (1 - Avg Cosine Similarity): {diversity:.3f}")
        print(f"  - Kepatuhan Aturan: {rule_adherence_status}")
    
    # --- Ringkasan Evaluasi Keseluruhan ---
    total_unique_items_in_catalog = len(df['nama_makanan'].unique())
    catalog_coverage_percentage = (len(all_recommended_items) / total_unique_items_in_catalog) * 100 if total_unique_items_in_catalog > 0 else 0
    
    print("\n--- Ringkasan Evaluasi Keseluruhan ---")
    print(f"Total Item Unik di Katalog: {total_unique_items_in_catalog}")
    print(f"Total Item Unik Direkomendasikan di Semua Skenario: {len(all_recommended_items)}")
    print("\n--- Evaluasi Selesai ---")

# --- 4. Skenario Pengujian (Bisa Disesuaikan) ---
evaluation_scenarios = [
    {
        'food_name': None,
        'rules': ['tinggi_protein', 'rendah_lemak'],
        'categories': ['Daging Mentah', 'Makanan Utama']
    },
    {
        'food_name': 'Ikan Bandeng',
        'rules': ['rendah_kalori'],
        'categories': ['Makanan Laut']
    },
    # {
    #     'food_name': 'Abon haruwan',
    #     'rules': ['rendah_kalori', 'rendah_lemak'],
    #     'categories': ['Makanan Utama']
    # },
    {
        'food_name': None,
        'rules': ['rendah_kalori'],
        'categories': ['Makanan Laut']
    },
    {
        'food_name': 'Agar-agar',
        'rules': [],
        'categories': []
    },
    {
        'food_name': 'Apel',
        'rules': ['rendah_kalori'],
        'categories': ['Buah-buahan']
    },
    {
        'food_name': 'Sawi',
        'rules': ['rendah_kalori', 'rendah_lemak'],
        'categories': ['Sayur-sayuran']
    }
]

if __name__ == "__main__":
    run_evaluation(evaluation_scenarios)