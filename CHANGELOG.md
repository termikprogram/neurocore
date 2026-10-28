# Sürüm Geçmişi

Tüm önemli değişiklikler bu dosyada tutulur.
Biçim: [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) ·
Sürümleme: [SemVer](https://semver.org/lang/tr/)

---

## [0.1.0] — 2026-10-28

İlk yayınlanan sürüm. Bu sürüm **ölçülmüş** davranışlara dayanır; her
iddianın yanında ölçüm değeri verilmiştir.

### Eklendi

**Ortak havuz — makineler arası iş paylaşımı**
- `neurocore/net/peer_pool.py` — SSH tabanlı keşif + iş paylaşımı
  - `register` / `peers` / `deregister` (TTL 90 sn)
  - `baslat` / `durdur` / heartbeat (30 sn) / **tünel gözetimi**
  - `komsu_dene` (yalnız komşu) · `dispatch` (yerel devralmalı)
  - ters tünel (`ssh -R`) + **sunucu üzerinden relay** (NAT aşımı)
- `POST /peer/job` · `GET /peer/info` — uzak düğümden iş alma
- **Küme anahtarı sunucudan dağıtılır** (`cluster.token`, izin 600);
  token SSH anahtarından **türetilmez**
- İzin listesi: uzaktan **yalnız okuma** (16 araç); `shell.run`,
  `code.run_command`, yazma araçları uzakta çalışmaz
- Ajan döngüsü entegrasyonu: izinli araç önce komşuda, yanıt yoksa
  **yerelde** (adım metninde `[uzak:<adres>]`)
- Ölçüm: tünel 6,3 sn · relay 1,7–2,7 sn · uçtan uca **6/6**

**Düşünme döngüsü ajana bağlandı**
- `AgentLoop.attach_thinker()` · `_ask_plan` → `(plan, meta)`
- Bulut **N farklı plan** önerir → **yerel model seçer** (ek LLM çağrısı yok)
- Düşünme izi adım kaydına yazılır · hata durumunda ilk adaya düşer
- Uzun görevde **otomatik plan** (`AUTO_PLAN_CHARS = 60`)

**Araştırma çekirdeğe alındı**
- `CORE_TOOLS`: 25 → **28** (`web.search`, `web.fetch`, `knowledge.ask`)
- Öncesinde yalnız anahtar sözcük eşleşince listeye giriyorlardı →
  ajan kendiliğinden araştırmıyordu

**Talimat verisi (soru → cevap)**
- `tools/talimat_verisi.py` — havuzdan 6.000 + elden 10 çift
  (DXGI/TDR/sürücü — havuzda bu konular yok)
- Eğitimde iki ayrı doğrulama: **belge** ve **talimat**
- `--seq` / `--batch` / `--kaydet-her` / `--amp-yok` parametreleri

**Kurulum ve dağıtım**
- `kur_istemci.ps1` — Python + `.venv` + PyTorch CUDA + bağımlılıklar;
  **SSH anahtarını kendisi üretir**, yetkisizse yapılacak komutu yazar
- `tools/kurulum_kontrol.py` — 36 kontrol → `HAZIR` / eksik listesi
- `paketle.ps1` — taşıma paketi (ölçüldü: 12,0 MB)
- `BASLAT-istemci.bat` — havuz açık başlatıcı

### Düzeltildi

- **Öğrenme oranı çökmesi**: ömür boyu token sayacı LR'yi 5e-06'ya
  sabitliyordu → tur hacmine göre 1,5e-04
- **Doğrulama kaybı hiç ölçülmüyordu**: `train()` artık başlangıç,
  periyodik ve final doğrulama üretiyor (2,6645 → 1,7704, −%33,6)
- **Kalite kapısı çöpü kabul ediyordu** (%16,7 gerçek kelime):
  zorunlu sözcük ölçütü eklendi → kapı doğruluğu %100
- **`/peer/job` anahtarsız açıktı**: küme anahtarı yoksa artık **403**
- **Test kirliliği**: sahte token gerçek `.neurocore`'u eziyordu →
  testler geçici dizin kullanıyor
- **Heartbeat aralığı eziliyordu** (`wait(max(1.0, aralık))`) → 0,05
- **`son_kayit_yas` 1,79 milyar sn** görünüyordu → damga yazılıyor

### Altyapı notları

- GPU sürücüsü yoğun eğitimde sıfırlanıyor (`nvlddmkm` Olay 153 = TDR);
  `TdrDelay` tanımsız (2 sn). Aşılma yolu: `--batch 8 --amp-yok`
  (300 adım kesintisiz doğrulandı)
- 8 GB VRAM sınıfında `--batch 8 --seq 128` önerilir
- Kodda **sır yok**: sunucu adresi ve anahtar yolu ortam değişkeni veya
  `.neurocore/peer.json` üzerinden okunur

### Ölçümler

```
Testler                 : 1001 OK (tam) · 52 (ortak havuz) · 10 (düşünme)
Kurulum doğrulaması     : 35/36 HAZIR
Çekirdek araç           : 28 · uzaktan izinli 16
Taşıma paketi           : 12,0 MB
Nöral model             : 15,9k adım · 107M token
```
