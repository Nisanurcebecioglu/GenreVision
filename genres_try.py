import pandas as pd
 
df_g = pd.read_csv('genres.csv')
 
# 1. Toplam satır vs benzersiz film sayısı
print("Toplam satır:", len(df_g))
print("Benzersiz film_id sayısı:", df_g['id'].nunique())
 
# 2. Bir filmin kaç satırı var?
ornek_id = df_g['id'].iloc[0]
print("\nÖrnek film için satırlar:")
print(df_g[df_g['id'] == ornek_id])
 
# 3. Tür dağılımı
print("\nFilm başına ortalama tür sayısı:",
      len(df_g) / df_g['id'].nunique())
