import pandas as pd

# 1. Verileri oku
df_m = pd.read_csv('movies.csv')
df_g = pd.read_csv('genres.csv')
df_p = pd.read_csv('posters.csv')

print(f"movies.csv:  {len(df_m):>9,} satır | {df_m['id'].nunique():>9,} benzersiz film")
print(f"genres.csv:  {len(df_g):>9,} satır | {df_g['id'].nunique():>9,} benzersiz film")
print(f"posters.csv: {len(df_p):>9,} satır | {df_p['id'].nunique():>9,} benzersiz film")
print(f"\ngenres.csv'deki tüm türler ({df_g['genre'].nunique()} adet):")
print(sorted(df_g['genre'].dropna().unique()))

# 2. genres.csv'yi pivot et: long → wide
#    Bu adımda 1M satır 676K satıra düşer, her tür ayrı kolon olur.
df_genres_wide = pd.crosstab(df_g['id'], df_g['genre']).reset_index()

# crosstab 0/1 dışı bir değer üretmez ama duplikasyona karşı emniyet:
tur_kolonlari = [c for c in df_genres_wide.columns if c != 'id']
df_genres_wide[tur_kolonlari] = (df_genres_wide[tur_kolonlari] > 0).astype(int)

# 3. main.py ile uyum için 'Tur_' öneki ekle
df_genres_wide = df_genres_wide.rename(columns={c: f'Tur_{c}' for c in tur_kolonlari})

# 4. Üç tabloyu birleştir (inner join — üçünde de bulunan filmler)
df = df_m.merge(df_genres_wide, on='id').merge(df_p, on='id')

# 5. DOĞRULAMA — bu çıktıları benimle paylaş
print("\n" + "="*55)
print("MASTER DATASET DOĞRULAMA")
print("="*55)
print(f"Toplam satır:       {len(df):>9,}")
print(f"Benzersiz film_id:  {df['id'].nunique():>9,}")
print(f"Eşit mi?            {'✓ EVET' if len(df) == df['id'].nunique() else '✗ HAYIR — BUG VAR'}")

tur_kols = [c for c in df.columns if c.startswith('Tur_')]
print(f"\nTür kolonu sayısı:  {len(tur_kols)}")
print(f"Film başına ort. tür sayısı: {df[tur_kols].sum(axis=1).mean():.2f}")

print(f"\nTür dağılımı (büyükten küçüğe):")
print(df[tur_kols].sum().sort_values(ascending=False).to_string())

# 6. Kaydet
df.to_csv('master_dataset.csv', index=False)
print(f"\n✓ master_dataset.csv kaydedildi ({len(df):,} satır)")