import pandas as pd
import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

df = pd.read_csv('master_dataset.csv')
tur_kolonlari = sorted([c for c in df.columns if c.startswith('Tur_')])
print(f"Master dataset: {len(df):,} film, {len(tur_kolonlari)} tür")

# ============================================================
# 1. NADİR TÜRLERİ FULL KORU
# ============================================================
NADIR_ESIK = 12000
HEDEF_BOYUT = 50000

tur_sayilari = df[tur_kolonlari].sum()
nadir_turler = tur_sayilari[tur_sayilari < NADIR_ESIK].index.tolist()
print(f"\nNadir türler (< {NADIR_ESIK:,}): {[t.replace('Tur_','') for t in nadir_turler]}")

# Nadir türlerden en az biri olan filmleri ayır
mask_nadir = df[nadir_turler].sum(axis=1) > 0
df_nadir = df[mask_nadir]
df_kalan = df[~mask_nadir]
print(f"Nadir türlü filmler (full korunacak): {len(df_nadir):,}")
print(f"Stratified seçim havuzu:               {len(df_kalan):,}")

# ============================================================
# 2. KALAN HAVUZDAN STRATIFIED SEÇİM
# ============================================================
hedef_kalan = HEDEF_BOYUT - len(df_nadir)
oran = hedef_kalan / len(df_kalan)
print(f"\nKalan slot: {hedef_kalan:,} ({oran*100:.1f}% seçim oranı)")

splitter = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=oran, random_state=42
)
_, idx_secim = next(splitter.split(df_kalan, df_kalan[tur_kolonlari].values))
df_secim = df_kalan.iloc[idx_secim]

# ============================================================
# 3. BİRLEŞTİR + KARIŞTIR + KAYDET
# ============================================================
df_final = pd.concat([df_nadir, df_secim], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n{'='*55}")
print(f"FINAL EĞİTİM SETİ")
print(f"{'='*55}")
print(f"Toplam: {len(df_final):,} film")
print(f"\nTür dağılımı (büyükten küçüğe):")
yeni_dagilim = df_final[tur_kolonlari].sum().sort_values(ascending=False)
for tur, sayi in yeni_dagilim.items():
    eski = tur_sayilari[tur]
    oran_yeni = sayi / len(df_final) * 100
    oran_eski = eski / len(df) * 100
    isaret = "↑" if oran_yeni > oran_eski else ("↓" if oran_yeni < oran_eski else "=")
    print(f"  {tur.replace('Tur_',''):20s} {sayi:>6,}  ({oran_yeni:5.2f}%)  {isaret} eskiden %{oran_eski:.2f}")

df_final.to_csv('train_dataset_50k.csv', index=False)
print(f"\n✓ train_dataset_50k.csv kaydedildi")

# ============================================================
# 4. POS_WEIGHT HESAPLA + KAYDET
# ============================================================
print(f"\n{'='*55}")
print(f"POS_WEIGHT HESABI")
print(f"{'='*55}")

pozitifler = df_final[tur_kolonlari].sum().values
toplam = len(df_final)
negatifler = toplam - pozitifler

pos_weight_raw = negatifler / np.maximum(pozitifler, 1)
pos_weight = np.clip(pos_weight_raw, 1.0, 50.0)  # gradient explode koruması

torch.save({
    'pos_weight': torch.tensor(pos_weight, dtype=torch.float32),
    'tur_kolonlari': tur_kolonlari
}, 'pos_weight.pt')

print(f"{'Tür':<22} {'Pozitif':>8} {'Negatif':>8} {'pos_w (raw)':>12} {'pos_w (clipped)':>16}")
print("-" * 70)
for tur, p, n, w_raw, w in zip(tur_kolonlari, pozitifler, negatifler, pos_weight_raw, pos_weight):
    flag = " ← clip!" if w_raw > 50 else ""
    print(f"{tur.replace('Tur_',''):<22} {p:>8} {n:>8} {w_raw:>12.2f} {w:>16.2f}{flag}")

print(f"\n✓ pos_weight.pt kaydedildi")