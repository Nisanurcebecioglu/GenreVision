import pandas as pd
import numpy as np
import torch

# === FAILED UPLOADS ANALİZİ ===
failed = pd.read_csv('failed_uploads.csv')
print(f"Toplam hata: {len(failed):,}\n")

# Hata kategorilerinin altındaki nedenleri özetle
print("Hata sebebi dağılımı:")
print(failed['reason'].value_counts())
print()

# fail_exc'lerin altındaki gerçek sebepleri grupla (ilk 60 karakter)
if 'detail' in failed.columns:
    print("İstisna türleri (en yaygın 10):")
    failed['detail_short'] = failed['detail'].fillna('').str[:60]
    print(failed[failed['reason']=='fail_exc']['detail_short'].value_counts().head(10))

# === SINIF DAĞILIMI KARŞILAŞTIRMASI ===
df_orig = pd.read_csv('train_dataset_50k.csv')
df_clean = pd.read_csv('train_dataset_50k_clean.csv')
tur_kolonlari = sorted([c for c in df_orig.columns if c.startswith('Tur_')])

print(f"\n{'='*68}")
print(f"DAĞILIM KAYMASI ANALİZİ")
print(f"{'='*68}")
print(f"{'Tür':<22} {'Eski':>8} {'Yeni':>8} {'Kayıp':>8} {'Kayıp %':>10}")
print('-' * 68)

ciddi_kayip = []
for tur in tur_kolonlari:
    eski = int(df_orig[tur].sum())
    yeni = int(df_clean[tur].sum())
    kayip = eski - yeni
    kayip_yuzde = (kayip / eski * 100) if eski > 0 else 0
    isaret = " ⚠" if kayip_yuzde > 20 else ("" if kayip_yuzde < 15 else " •")
    print(f"{tur.replace('Tur_',''):<22} {eski:>8,} {yeni:>8,} {kayip:>8,} {kayip_yuzde:>9.1f}%{isaret}")
    if kayip_yuzde > 20:
        ciddi_kayip.append(tur.replace('Tur_',''))

if ciddi_kayip:
    print(f"\n⚠ %20+ kayıp olan türler: {ciddi_kayip}")
else:
    print(f"\n✓ Hiçbir türde %20'den fazla kayıp yok.")

# === POS_WEIGHT YENİDEN HESAPLAMA (clean dataset için) ===
print(f"\n{'='*68}")
print(f"POS_WEIGHT YENİDEN HESAPLAMA (clean dataset için)")
print(f"{'='*68}")

pozitifler = df_clean[tur_kolonlari].sum().values
toplam = len(df_clean)
negatifler = toplam - pozitifler

pos_weight_raw = negatifler / np.maximum(pozitifler, 1)
pos_weight = np.clip(pos_weight_raw, 1.0, 50.0)

torch.save({
    'pos_weight': torch.tensor(pos_weight, dtype=torch.float32),
    'tur_kolonlari': tur_kolonlari
}, 'pos_weight.pt')  # Mevcut dosyayı güncel hâlle değiştir

# Eski vs yeni karşılaştırma
print(f"{'Tür':<22} {'pos_w (yeni)':>14} {'değişim':>10}")
print('-' * 50)
eski_ckpt = None
try:
    # Eski değeri okumayı dene (yedek için)
    pass
except:
    pass

for tur, w in zip(tur_kolonlari, pos_weight):
    flag = " ← clip!" if pos_weight_raw[tur_kolonlari.index(tur)] > 50 else ""
    print(f"{tur.replace('Tur_',''):<22} {w:>14.2f}{flag}")

print(f"\n✓ pos_weight.pt güncellendi (clean dataset üzerinden).")