# Kurulum

**Sürüm 0.1.0** · Windows 10/11 · Python 3.10+ · PyTorch (CUDA)

---

## 1 · Gereksinimler

| Bileşen | En az | Önerilen | Not |
|---|---|---|---|
| Windows | 10 64-bit | 11 | Linux/macOS denenmedi |
| Python | 3.10 | **3.10.11** | 64-bit · "Add to PATH" işaretli |
| RAM | 8 GB | 16 GB+ | |
| Disk | 10 GB | 30 GB | checkpoint'ler büyük |
| GPU | — | NVIDIA 8 GB+ | uygulama GPU'suz da çalışır |
| NVIDIA sürücü | — | ≥ 550 | eğitim/çıkarım için |

### GPU profilleri

```
Pascal (GTX 1080 Ti, 11 GB) : fp16 çalışır · `--amp-yok` gerekebilir · bf16 emülasyon
Ampere (RTX 3060 Ti,  8 GB) : donanımsal bf16 · fp16 AMP kararlı · --batch 8
```

⚠ 1,5B parametreli model 8 GB VRAM'e **sığmaz** (40M model rahat çalışır).

---

## 2 · Hızlı kurulum (tek komut)

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
Set-ExecutionPolicy -Scope Process Bypass -Force
.\kur_istemci.ps1
```

Betiğin yaptıkları:

```
── 1b) SSH ANAHTARI
  • çalışan anahtar arar (NC_SSH_KEY → ~/.ssh/nc_<makine> → id_ed25519)
  • yoksa ed25519 ÜRETİR, izinleri kısıtlar
  • sunucuda dener; yetkisizse AÇIK anahtarı + yetkilendirme komutunu YAZDIRIR
  • NC_SSH_KEY'i kalıcı yazar (kullanıcı ortamı)
── 1) Python ≥ 3.10
── 2) .venv
── 3) PyTorch (CUDA 12.4)
── 4) requirements.txt
── 5) tools/kurulum_kontrol.py  → HAZIR / eksik listesi
```

### Seçenekler

| Parametre | İş |
|---|---|
| `-SadeceAnahtar` | yalnız anahtar adımı (kurulum yapmaz) |
| `-SadeceKontrol` | yalnız doğrulama |
| `-YetkiBekle` | yetkilendirme komutunu gösterir, Enter bekler, yeniden dener |
| `-AnahtarDizin`, `-AnahtarAdi` | anahtarın yeri/adı |
| `-AnahtarYenile` | yeni anahtar üret |
| `-Torch cu121/cu118` | farklı CUDA sürümü |
| `-VenvYok` | sanal ortam kullanma |
| `-Sunucu user@host` | havuz sunucusu (yoksa `NC_SERVER` / `peer.json`) |

---

## 3 · Elle kurulum

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\kurulum_kontrol.py
```

---

## 4 · Doğrulama

```powershell
.venv\Scripts\python.exe tools\kurulum_kontrol.py            # 34–36 kontrol
.venv\Scripts\python.exe tools\kurulum_kontrol.py --kaydet   # havuza kayıt da dene
```

Kontrol başlıkları:

```
1) PYTHON             sürüm, mimari
2) DEPO               neurocore/, requirements.txt, tools/, tests/
3) PAKETLER           torch, fastapi, uvicorn, psutil, numpy, …
4) GPU                CUDA, ad, VRAM, donanımsal bf16
5) RAM / DİSK         eşikler
6) SSH / HAVUZ        anahtar, bağlantı, sunucunun gördüğü IP, küme anahtarı
7) ARAÇLAR            çekirdek 28 · uzaktan izinli 16
8) UYGULAMA           FastAPI, peer ucu, __main__
```

Çıktı: `SONUÇ: n/m · HAZIR` veya eksik listesi (çıkış kodu 1).

---

## 5 · Paketleme (başka makineye taşıma)

```powershell
.\paketle.ps1
```

Ölçülen: **12,0 MB** zip · 8.656 dosya · 5 klasör.

```
Dahil  : neurocore · tools · tests · raporlar · knowledge_base
         requirements.txt · kur_istemci.ps1 · BASLAT-istemci.bat · paketle.ps1
Hariç  : models (23,6 GB) · data (6,5 GB) · .neurocore (2,3 GB)
         Varliklar · TestProjects · __pycache__ · .venv
```

⚠ `.neurocore` **bilinçli olarak dışarıda**: içinde düğüm kimliği
(`node_id`) var. Kopyalanırsa iki makine aynı düğüm görünür.

---

## 6 · Sorun giderme

| Belirti | Neden | Çözüm |
|---|---|---|
| `python` bulunamadı | PATH | Python 3.10 kurarken "Add to PATH" |
| PowerShell betiği çalışmıyor | yürütme ilkesi | `Set-ExecutionPolicy -Scope Process Bypass -Force` |
| `UNPROTECTED PRIVATE KEY FILE` | Windows anahtar izni | `icacls "<anahtar>" /inheritance:r /grant:r "%USERNAME%:R"` |
| `Sunucuya bağlantı ✗` | anahtar yetkisiz | `kur_istemci.ps1 -SadeceAnahtar` → çıkan komutu çalıştır |
| `torch` CUDA görmüyor | CPU tekerleği kurulmuş | `pip install torch --index-url .../cu124 --force-reinstall` |
| Doğrulamada `bf16` • | Pascal'da emülasyon | normal (bilgi amaçlı, engel değil) |
| Havuzda 1 düğüm | aynı `node_id` | `.neurocore\node_id` sil |
| `HAZIR` değil, paket `YOK` | pip hatası | `pip install -r requirements.txt` |

Günlük dosyası: `.neurocore/logs/neurocore.log`
