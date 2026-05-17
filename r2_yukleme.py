import pandas as pd
import boto3
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# --- 1. CLOUDFLARE R2 BİLGİLERİNİ BURAYA GİR ---
ACCESS_KEY = '26ff1aa1645b69cac212507ec4c079dd'
SECRET_KEY = '37872cda19b45e33509d5b84a3514234a5a06755481363e1561cb7a404f23a77'
ENDPOINT_URL = 'https://8a97a7792885ace66acb04fa79b9ded7.r2.cloudflarestorage.com' # Örn: https://<hesap-id>.r2.cloudflarestorage.com
BUCKET_NAME = 'genrevision' 


df = pd.read_csv('train_dataset_50k.csv')
print(f"Yüklenecek afiş sayısı: {len(df):,}")

def make_s3_client():
    """Her thread için ayrı client (boto3 thread-safe değildir)"""
    return boto3.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name='auto'
    )

def process_one(row):
    film_id = int(row['id'])
    url = row['link']
    key = f"{film_id}.jpg"
    s3 = make_s3_client()

    # 1. Zaten R2'de varsa atla (idempotent — script'i yeniden çalıştırabilirsin)
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return ('skip', film_id, None)
    except s3.exceptions.ClientError:
        pass

    # 2. İndir + boyutlandır + yükle (3 retry ile)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                if attempt == 2:
                    return ('fail_http', film_id, f'HTTP {r.status_code}')
                time.sleep(0.5)
                continue

            img = Image.open(BytesIO(r.content)).convert('RGB').resize((224, 224))
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            s3.put_object(
                Bucket=BUCKET_NAME, Key=key, Body=buf,
                ContentType='image/jpeg'
            )
            return ('ok', film_id, None)
        except Exception as e:
            if attempt == 2:
                return ('fail_exc', film_id, str(e)[:100])
            time.sleep(0.5)

    return ('fail_unknown', film_id, None)

# === ANA DÖNGÜ ===
sonuc = {'ok': 0, 'skip': 0, 'fail_http': 0, 'fail_exc': 0, 'fail_unknown': 0}
hatalilar = []

print("Yükleme başlıyor (16 paralel thread)...\n")
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = [ex.submit(process_one, row) for _, row in df.iterrows()]
    pbar = tqdm(as_completed(futures), total=len(futures), desc="Yükleniyor")
    for fut in pbar:
        durum, film_id, hata = fut.result()
        sonuc[durum] += 1
        if durum.startswith('fail'):
            hatalilar.append({'id': film_id, 'reason': durum, 'detail': hata})
        # Her 500 işlemde özet göster
        if (sonuc['ok'] + sonuc['skip']) % 500 == 0:
            pbar.set_postfix(ok=sonuc['ok'], skip=sonuc['skip'],
                             fail=sum(v for k,v in sonuc.items() if k.startswith('fail')))

# === ÖZET ===
print(f"\n{'='*55}")
print(f"YÜKLEME ÖZETİ")
print(f"{'='*55}")
for k, v in sonuc.items():
    print(f"  {k:<15} {v:>6,}")
print(f"  {'TOPLAM':<15} {sum(sonuc.values()):>6,}")

# Hatalıları kaydet
if hatalilar:
    pd.DataFrame(hatalilar).to_csv('failed_uploads.csv', index=False)
    print(f"\n⚠ {len(hatalilar):,} hata kaydedildi → failed_uploads.csv")

# Hataları eğitim setinden çıkar
if hatalilar:
    failed_ids = set(h['id'] for h in hatalilar)
    df_clean = df[~df['id'].isin(failed_ids)].reset_index(drop=True)
    df_clean.to_csv('train_dataset_50k_clean.csv', index=False)
    print(f"✓ Temiz set kaydedildi: train_dataset_50k_clean.csv ({len(df_clean):,} satır)")
else:
    df.to_csv('train_dataset_50k_clean.csv', index=False)
    print(f"✓ Hiç hata yok! train_dataset_50k_clean.csv = train_dataset_50k.csv")