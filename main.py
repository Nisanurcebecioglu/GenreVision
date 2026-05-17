import os
import numpy as np
import torch
import torch.nn as nn
import requests
import boto3
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image

# === KONFIGÜRASYON (Railway environment variables) ===
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY', '').strip().strip('"').strip("'")
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY', '').strip().strip('"').strip("'")
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL', '').strip().strip('"').strip("'").rstrip('/')
R2_BUCKET = os.getenv('R2_BUCKET', 'genrevision').strip().strip('"').strip("'")

# === MODEL KONFIGÜRASYONLARI ===
# Her model için R2 yolları ve yerel dosya adları tanımlanır.
# Model dosyalarını R2'ye yükledikten sonra bu yollar otomatik çalışır.
MODEL_CONFIGS = {
    'custom': {
        'display_name': 'GenreVision Custom (ResNet18)',
        'arch': 'resnet18',
        'r2_model': 'model/genre_vision_v3_best.pth',
        'r2_thresholds': 'model/best_thresholds_v3.npy',
        'local_model': 'genre_vision_v3_best.pth',
        'local_thresholds': 'best_thresholds_v3.npy',
    },
    'resnet50': {
        'display_name': 'GenreVision ResNet50',
        'arch': 'resnet50',
        'r2_model': 'model/genre_vision_v4_best.pth',
        'r2_thresholds': 'model/best_thresholds_v4.npy',
        'local_model': 'genre_vision_v4_best.pth',
        'local_thresholds': 'best_thresholds_v4.npy',
    },
    'efficientnet': {
        'display_name': 'GenreVision EfficientNet-B0',
        'arch': 'efficientnet_b0',
        'r2_model': 'model/genre_vision_efficientnet.pth',
        'r2_thresholds': 'model/best_thresholds_efficientnet.npy',
        'local_model': 'genre_vision_efficientnet.pth',
        'local_thresholds': 'best_thresholds_efficientnet.npy',
    },
}

DEFAULT_MODEL = 'custom'

app = FastAPI(title='GenreVision API v3')
app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

device = torch.device('cpu')

# Yüklü modeller: { model_key: { 'model': nn.Module, 'label_cols': list, 'thresholds': np.array } }
loaded_models: dict = {}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PredictionRequest(BaseModel):
    url: str


def diagnose_config():
    issues = []
    if not R2_ACCESS_KEY:
        issues.append('R2_ACCESS_KEY boş')
    elif len(R2_ACCESS_KEY) != 32:
        issues.append(f'R2_ACCESS_KEY uzunluğu {len(R2_ACCESS_KEY)}, 32 olmalı')
    if not R2_SECRET_KEY:
        issues.append('R2_SECRET_KEY boş')
    elif len(R2_SECRET_KEY) != 64:
        issues.append(f'R2_SECRET_KEY uzunluğu {len(R2_SECRET_KEY)}, 64 olmalı')
    if not R2_ENDPOINT_URL:
        issues.append('R2_ENDPOINT_URL boş')
    elif not R2_ENDPOINT_URL.startswith('https://'):
        issues.append('R2_ENDPOINT_URL https:// ile başlamalı')
    if not R2_BUCKET:
        issues.append('R2_BUCKET boş')
    return issues


def get_s3_client():
    return boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto')


def r2_file_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except Exception:
        return False


def download_from_r2(s3, key: str, local_path: str):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        print(f'  {local_path} zaten var, atlandı')
        return
    s3.download_file(R2_BUCKET, key, local_path)
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f'  ✓ {key} indirildi ({size_mb:.2f} MB)')


def build_model_arch(arch: str, num_classes: int) -> nn.Module:
    if arch == 'resnet18':
        base = models.resnet18(weights=None)
        base.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(base.fc.in_features, num_classes))
    elif arch == 'resnet50':
        base = models.resnet50(weights=None)
        base.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(base.fc.in_features, num_classes))
    elif arch == 'efficientnet_b0':
        base = models.efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f'Bilinmeyen mimari: {arch}')
    return base


def try_load_model(key: str, cfg: dict, s3) -> bool:
    """Tek bir modeli R2'den indirip yükler. Başarısızsa False döner."""
    print(f'\n  [{key}] {cfg["display_name"]} yükleniyor...')

    if not r2_file_exists(s3, cfg['r2_model']):
        print(f'  [{key}] R2\'de model dosyası bulunamadı ({cfg["r2_model"]}), atlandı')
        return False
    if not r2_file_exists(s3, cfg['r2_thresholds']):
        print(f'  [{key}] R2\'de eşik dosyası bulunamadı ({cfg["r2_thresholds"]}), atlandı')
        return False

    try:
        download_from_r2(s3, cfg['r2_model'], cfg['local_model'])
        download_from_r2(s3, cfg['r2_thresholds'], cfg['local_thresholds'])

        ckpt = torch.load(cfg['local_model'], map_location=device, weights_only=False)
        label_cols = ckpt['label_cols']

        net = build_model_arch(cfg['arch'], len(label_cols))
        net.load_state_dict(ckpt['model_state'])
        net.eval()

        thresholds = np.load(cfg['local_thresholds'])

        loaded_models[key] = {
            'model': net,
            'label_cols': label_cols,
            'thresholds': thresholds,
        }
        print(f'  [{key}] ✓ Yüklendi — {len(label_cols)} tür')
        return True

    except Exception as e:
        print(f'  [{key}] ✗ Yükleme hatası: {e}')
        # Yarım kalan dosyaları temizle
        for path in (cfg['local_model'], cfg['local_thresholds']):
            if os.path.exists(path) and os.path.getsize(path) < 1000:
                os.remove(path)
        return False


@app.on_event('startup')
def load_all_models():
    print('=' * 60)
    print('Startup: Konfigürasyon kontrol ediliyor...')

    issues = diagnose_config()
    if issues:
        for issue in issues:
            print(f'  ✗ {issue}')
        raise RuntimeError(f'Env eksik: {issues}')

    print(f'  R2_ENDPOINT_URL: {R2_ENDPOINT_URL}')
    print(f'  R2_BUCKET: {R2_BUCKET}')

    s3 = get_s3_client()
    try:
        s3.list_objects_v2(Bucket=R2_BUCKET, Prefix='model/', MaxKeys=1)
    except Exception as e:
        raise RuntimeError(f'R2 bucket erişimi yok: {e}')

    print('\nStartup: Modeller yükleniyor...')
    for key, cfg in MODEL_CONFIGS.items():
        try_load_model(key, cfg, s3)

    if not loaded_models:
        raise RuntimeError('Hiçbir model yüklenemedi. R2 bucket\'ı kontrol edin.')

    available = list(loaded_models.keys())
    print(f'\n✓ Yüklü modeller: {available}')
    print('=' * 60)


def get_model_or_raise(model_key: str) -> dict:
    if model_key not in MODEL_CONFIGS:
        raise HTTPException(status_code=400,
            detail=f'Geçersiz model: {model_key}. Geçerliler: {list(MODEL_CONFIGS.keys())}')
    if model_key not in loaded_models:
        raise HTTPException(status_code=503,
            detail=f'"{model_key}" modeli yüklü değil. Mevcut: {list(loaded_models.keys())}')
    return loaded_models[model_key]


def run_inference(img: Image.Image, model_key: str) -> dict:
    entry = get_model_or_raise(model_key)
    net = entry['model']
    label_cols = entry['label_cols']
    thresholds = entry['thresholds']

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(net(x)).squeeze().numpy()

    all_scores, confident_genres = {}, {}
    for i, tur in enumerate(label_cols):
        tur_adi = tur.replace('Tur_', '')
        yuzde = round(float(probs[i]) * 100, 2)
        all_scores[tur_adi] = yuzde
        if probs[i] > thresholds[i]:
            confident_genres[tur_adi] = yuzde

    sirali_all = dict(sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True))
    sirali_confident = dict(sorted(confident_genres.items(), key=lambda kv: kv[1], reverse=True))

    return {
        'model_used': model_key,
        'model_display': MODEL_CONFIGS[model_key]['display_name'],
        'turler': sirali_confident,
        'tum_skorlar': sirali_all,
        'top_5': dict(list(sirali_all.items())[:5]),
    }


@app.get('/')
def root():
    return {
        'status': 'ok',
        'available_models': list(loaded_models.keys()),
        'default_model': DEFAULT_MODEL,
        'endpoints': [
            'GET  /models',
            'POST /predict?model=custom  (JSON: {url})',
            'POST /predict_file?model=custom  (multipart file)',
        ],
    }


@app.get('/health')
def health():
    return {
        'status': 'ok' if loaded_models else 'loading',
        'loaded_models': list(loaded_models.keys()),
    }


@app.get('/models')
def list_models():
    """Frontend'e hangi modellerin mevcut olduğunu ve açıklamalarını döner."""
    result = {}
    for key, cfg in MODEL_CONFIGS.items():
        result[key] = {
            'display_name': cfg['display_name'],
            'arch': cfg['arch'],
            'available': key in loaded_models,
        }
    return {'models': result, 'default': DEFAULT_MODEL}


@app.post('/predict')
def predict_url(
    request: PredictionRequest,
    model: str = Query(default=DEFAULT_MODEL, description='Kullanılacak model anahtarı'),
):
    """URL'den afiş indir ve seçili modelle tahmin et."""
    try:
        r = requests.get(request.url, timeout=8)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return run_inference(img, model)
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f'Görsel indirilemedi: {e}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Tahmin hatası: {e}')


@app.post('/predict_file')
async def predict_file(
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL, description='Kullanılacak model anahtarı'),
):
    """Yüklenen dosyadan seçili modelle tahmin et."""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400,
            detail=f'Görsel dosyası bekleniyor, alınan: {file.content_type}')

    try:
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail='Dosya 20 MB sınırını aşıyor')
        img = Image.open(BytesIO(contents)).convert('RGB')
        return run_inference(img, model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Dosya işlenemedi: {e}')
