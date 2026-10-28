# NeuroCore AI

**Sürüm 0.1.0** · Windows · Python 3.10+ · PyTorch (CUDA) · [Sürüm geçmişi](CHANGELOG.md)

> ### Hızlı başlangıç
> ```powershell
> git clone https://github.com/termikprogram/neurocore.git
> cd neurocore
> Set-ExecutionPolicy -Scope Process Bypass -Force
> .\kur_istemci.ps1
> ```
> Ayrıntılar: [Kurulum](docs/KURULUM.md) · [Kullanım](docs/KULLANIM.md) ·
> [Ortak havuz](docs/ORTAK-HAVUZ.md) · [Güvenlik](SECURITY.md)

**Donanım-farkında, çok-modlu, kendi kendini büyütebilen yapay zeka sistemi.**
Aynı zamanda **kod yazabilen, komut çalıştırabilen ve uygulamaları (Godot,
Blender) kontrol edebilen bir masaüstü ajanı.**

NeuroCore tek bir "model" değil, **sekiz katmanlı bir bilişsel işletim sistemidir**:
donanımını bilir, kullanıcının bilgisayarını doyurmaz, bilgiyi Obsidian uyumlu bir
kasada tutar, o bilgiyi okuyup geliştirir ve gerektiğinde **kendi sinir ağını
büyütür**.

```
┌─────────────────────────────────────────────────────────────┐
│  L8 Evrim    deneyim · değerlendirme · sentez · merak        │
│  L7 Planlayıcı  görev grafiği · akıl yürütme · checkpoint    │
│  L6 Dil      algılama · çeviri · terim sözlüğü              │
│  L5 İşletim Sistemi  envanter · dosya · süreç · sandbox     │
│  L4 Çok-modlu  metin · görüntü · ses · tablo · füzyon       │
│  L3 Nöral    MoE transformer · bellek bankası · büyüme      │
│  L2 Bilgi    Obsidian kasası · SQLite indeks · bilgi grafı  │
│  L1 Donanım  izleme · kaynak tahsisi · profilleme           │
│  L0 Çekirdek  config · olay bus'ı · motor (orkestrasyon)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Masaüstü Uygulaması — Çift Tıkla Çalıştır

| Dosya | Ne yapar |
|---|---|
| **`Run.bat`** | **Ana başlatıcı.** Uygulamayı konsol penceresi olmadan açar. Çift tıklayın. |
| `Kurulum.bat` | Gerekli Python paketlerini kurar. İlk kullanımdan önce bir kez çalıştırın. |
| `Run-Debug.bat` | Konsolu görünür bırakır, tüm günlüğü ekrana yazar. **Uygulama açılmazsa bunu çalıştırın.** |
| `Run-Here.bat` | Bulunduğu klasörü çalışma alanı yapar. Bir proje klasörüne kısayolunu koyun. |
| `Run-Yolo.bat` | Serbest mod (onay sormaz). Önce uyarı gösterip onay ister. |
| `Sunuculari-Kapat.bat` | Çalışan tüm NeuroCore örneklerini temiz biçimde kapatır. |
| `Kisayol-Olustur.bat` | Masaüstü + Başlat menüsü kısayolu oluşturur; belirli bir proje için de kısayol ekleyebilir. |

`Run.bat` üç şeyi garanti eder:

1. **Konsol penceresi yok** — `pythonw.exe` kullanılır, arkada siyah pencere kalmaz.
2. **Bağımsız süreç** — başlatıcı penceresi kapansa bile uygulama çalışmaya devam eder.
3. **Tek örnek** — zaten çalışan bir örnek varsa yenisi açılmaz; yeni pencere mevcut
   örneğe bağlanır (aksi hâlde pencere eski sunucuya bağlanıp kafa karıştırıcı
   hatalar verirdi).

Günlük dosyası: `<proje>\.neurocore\logs\neurocore.log`

### Komut satırından

```bash
python -m neurocore.app                                  # mevcut dizin
python -m neurocore.app --workspace E:/PROJELER/oyunum   # belirli proje
python -m neurocore.app --autonomy readonly              # salt okunur başlat
python -m neurocore.app --no-window                      # yalnızca sunucu
```

Pencere, Edge'in `--app` moduyla **ayrı bir profilde** açılır: adres çubuğu ve
sekme yoktur, kullanıcının tarayıcı oturumuna dokunmaz. Windows 11'de WebView2
ve Edge zaten kurulu olduğu için **hiçbir ek kurulum gerekmez**.

---

## Hızlı Başlangıç (Bilgi Motoru)

```bash
# 1. Bağımlılıklar (torch zaten kuruluysa atlanabilir)
pip install -r requirements.txt

# 2. İlk kurulum: sistem notlarını tohumla + tokenizer eğit
python -m neurocore init

# 3. Bilgi öğret
python -m neurocore learn --file notlarim.md --tags donanim gpu

# 4. Soru sor
python -m neurocore ask "GPU sıcaklığı neden yükselir?"
python -m neurocore ask --deep "Ryzen 9 ile i7 karşılaştır"

# 5. Eğit ve büyüt
python -m neurocore train --steps 300
python -m neurocore evolve --cycles 2

# 6. Sistemin ne kadarını kullandığımızı gör
python -m neurocore resource
```

Python'dan:

```python
from neurocore import NeuroCoreEngine

with NeuroCoreEngine() as engine:          # donanımı algılar, katmanları kurar
    engine.bootstrap()                     # ilk kurulum (tohum + tokenizer)
    engine.learn("Termal macun iki yılda bir yenilenmelidir.",
                 title="Termal Macun", tags=["donanım"])

    result = engine.query("GPU sıcaklığı neden yükselir?")
    print(result.render())                 # yanıt + kaynaklar + güven

    print(engine.resource_report()["summary"])
    engine.train(steps=200)                # kaynak uygunsa eğitir
    engine.evolve(cycles=2)                # evrim + büyüme kararı
    print(engine.health_report().render())
```

---

## Kaynak Nezaketi — Bilgisayarınızı Kilitlemez

Sistem **boşta olan kaynağın %90'ından fazlasını kullanmaz**:

```
harcanabilir = toplam × (1 − mevcut_kullanım) × 0.90
```

| Durum | Ajanın bütçesi |
|---|---|
| CPU %50 kullanımda (video izliyorsunuz) | toplamın **%45'i** |
| Sistem %90 boşta | toplamın **%81'i** (üst sınır %85) |
| RAM %90 dolu | neredeyse hiç — ajan **duraklar** |

Uzun işler (eğitim, araştırma, indeksleme) her 2 adımda sistemi denetler:
kaynaklar dolarsa **duraklar**, gerekirse toplu işi küçültür, sürekli baskı
altında turu bitirir (ilerleme checkpoint'lenir).

```bash
python -m neurocore resource --factor 0.5     # daha çekingen ol
python -m neurocore resource --factor 1.0 --max-ram 0.95   # agresif mod
python -m neurocore resource --no-limit       # nezaketi kapat
```

---

## CLI Komutları

| Komut | Açıklama |
|---|---|
| `python -m neurocore.app` | **Masaüstü uygulaması** (sohbet + ajan) |
| `init` | İlk kurulum: sistem notları + tokenizer eğitimi |
| `ask` | Soru sor (`--deep` çok adımlı, `--json` makine okur) |
| `learn` | Dosya veya metinden bilgi öğret |
| `research` | Konuyu kasadan ve (varsa) çevrimiçi kaynaklardan araştır |
| `train` | Modeli eğit (kasa + deneyim verisi) |
| `evolve` | Evrim turu + büyüme kararı |
| `search` | Kasadaki bilgileri ara (yanıt üretmeden) |
| `graph` | Bilgi grafı raporu (`--export`, `--write`) |
| `health` | Sağlık raporu (`--deep`) |
| `stats` | Katman katman istatistikler |
| `status` | Tek satır durum |
| `resource` | **Kaynak nezaketi raporu ve politikası** |
| `os` | İşletim sistemi işlemleri (sandbox denetimli) |
| `shell` | Etkileşimli mod |

Her komut `--json` destekler (boru hattı ve otomasyon için).

---

## Dil Modeli

> ⚠ **Kod üretimi ölçümü (40 turluk çalışma, `raporlar/KAPANIS-RAPORU-40-Tur.md`)**
>
> Kendi modellerimizin **Godot 4** kod üretiminde derleme başarısı:
>
> | Model | Parametre | 50 soruluk benchmark | Godot derleme |
> |---|---|---|---|
> | `qwen_g4b` ⭐ | 0,5B | **26/50 (%52)** | **3/4** |
> | `qwen_coder_15b` | 1,5B | 23/50 (%46) | 1/4 |
>
> **Kritik bulgu:** 1,5B **temel** model, 0,5B **ince ayarlı** modelden
> **daha kötü** — çünkü benchmark **kendi alanımızı** ölçüyor ve alan
> verisi model boyutundan önemli.
>
> **Derleme döngüsü** (`tools/derleme_onarim.py`): ÜRET → DERLE → HATA →
> TEKRAR. 6 sözdizimsel normalleştirme ile **0/4 → 3/4** (ortalama tur 3,0).
> Ayrıntı: `raporlar/Tur36-3te4-Derlendi.md`.

Ajanın akıl yürütmesi için bir dil modeli gerekir. Sağlayıcı katmanı
**takılıp çıkarılabilir**:

| Sağlayıcı | Tür | Ortam değişkeni | Önerilen model |
|---|---|---|---|
| **Google Gemini** | bulut | `GEMINI_API_KEY` | `gemini-3.6-flash` |
| **DeepSeek** | bulut | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| **Groq** | bulut | `GROQ_API_KEY` | `openai/gpt-oss-20b` |
| **OpenAI** | bulut | `OPENAI_API_KEY` | `gpt-4o-mini` |
| **OpenRouter** | bulut | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat` |
| **Together AI** | bulut | `TOGETHER_API_KEY` | `deepseek-ai/DeepSeek-V3` |
| LM Studio · vLLM · llama.cpp · Ollama | yerel | gerekmez | herhangi |

Anahtar sırası: **Ayarlar** → ortam değişkeni → kayıtlı dosya. Anahtar
`.neurocore/llm_credentials.json` içine yazılır ve dosya izinleri kısıtlanır
(Windows'ta `icacls` ile ACL daraltılır).

### Model adları neden kendiliğinden güncelleniyor?

Sağlayıcılar modelleri sık kapatır. Google, `gemini-2.0-flash` için şu mesajı
döndürüyor: *"This model is no longer available. Please update your code to use
models/gemini-3.8-flash"*. Eski bir ad, anahtar geçerli olsa bile `404` verir ve
kullanıcı bunu "anahtar geçersiz" sanır. Bu yüzden üç katmanlı koruma vardır:

1. **Göç** — kullanımdan kaldırılmış adlar (`gemini-2.0-flash`,
   `llama-3.3-70b-versatile`, …) açılışta güncel karşılıklarıyla değiştirilir ve
   diske yazılır.
2. **Başlangıç doğrulaması** — aktif model hafif bir istekle sınanır; yeni bir
   model yoğunluk nedeniyle geçici `503` veriyorsa yedek zincirdeki çalışan
   modele geçilir ve seçim kaydedilir. Böylece görev **yarıda kesilmez**.
3. **Sağlayıcının önerisini yakalama** — hata mesajındaki `use models/<ad>`
   ifadesi okunur ve o ad denenir.

### İstek bütçesi (ücretsiz katmanlar)

Ücretsiz katmanların dakikalık token sınırı vardır. Ölçülen değer: **Groq
ücretsiz katmanı dakikada 8000 token**. 63 aracın tüm şemalarını göndermek bu
sınırı aşar (`HTTP 413` / `HTTP 429`) ve üstelik modelin doğru aracı seçmesini
zorlaştırır. Bu yüzden:

* **Göreve göre araç seçimi** — her istekte yalnızca ilgili araçlar gönderilir
  (Godot görevinde Godot araçları, ekran görevinde görsel doğrulama araçları).
* **Sağlayıcıya göre bütçe** — Groq için 12 araç / 2048 token, Gemini için
  20 araç / 8192 token.
* **Hız sınırında bekleme** — `429` alındığında 20 sn beklenip yeniden denenir
  (3 denemeye kadar); kullanıcının isteği tekrar yazması gerekmez.

**Hiçbir sağlayıcı yoksa** ajan **kural tabanlı modda** çalışır: serbest akıl
yürütemez ama proje haritası, dosya okuma, kod arama, test çalıştırma ve araç
sürümleri gibi somut istekleri yerine getirir. Sistem tamamen kullanılamaz
hâle gelmez.

---

## Mimari Özeti

| Katman | Ne yapar | Ana modüller |
|---|---|---|
| **L1 Donanım** | CPU/RAM/VRAM izleme, nezaketli kaynak bütçesi, profilleme | `monitor`, `allocator`, `profiler` |
| **L2 Bilgi** | Obsidian kasası, SQLite+FTS5 indeks, bilgi grafı, hibrit arama, sürümleme | `vault`, `note`, `index`, `graph`, `search`, `versioning` |
| **L3 Nöral** | Byte-level BPE, RoPE dikkat, MoE transformer, bellek bankası, gömme, eğitim | `tokenizer`, `attention`, `transformer`, `moe`, `memory`, `embedding`, `trainer` |
| **L4 Çok-modlu** | Metin/görüntü/ses/tablo kodlayıcıları + dikkat füzyonu | `router`, `text_encoder`, `image_encoder`, `audio_encoder`, `table_encoder`, `fusion` |
| **L5 OS** | Sistem envanteri, yetenek keşfi, dosya/süreç/kabuk (sandbox) | `inventory`, `capabilities`, `filesystem`, `process`, `shell`, `sandbox`, `audit` |
| **L6 Dil** | Dil algılama, terim sözlüğü, çeviri zinciri (yerel→API), özetleme | `detector`, `dictionary`, `translator`, `api_bridge` |
| **L7 Planlayıcı** | Görev DAG'ı, şablon ayrıştırıcı, yürütücü, checkpoint, yeniden planlama | `task`, `decomposer`, `executor`, `checkpoint` |
| **L8 Evrim** | Deneyim tamponu, öz değerlendirme, müfredat, sentez, merak, büyüme | `experience`, `evaluator`, `curriculum`, `synthesizer`, `curiosity`, `growth`, `integrity` |

Detaylı tasarım: **[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## Kendini Büyütme (Nötr Büyüme)

Model, yönlendirme entropisi yükseldiğinde (uzmanlar token'ları ayırt edemiyor)
**yeni uzman ekler**. Ekleme matematiksel olarak **nötrdür**:

* Yeni uzmanın yönlendirme bias'ı çok negatif başlar → top-k seçimine girmez
* Mevcut uzmanların kapı ağırlıkları birebir korunur
* Uzman gradyan akışıyla (dengeleme kaybı) kademeli devreye girer

Sonuç: büyüme anında model çıktısı **değişmez** (ölçülen delta ≤ 5e-7), mevcut
bilgi bozulmaz. Aynı mekanizma katman eklemede de geçerlidir (delta tam 0.0).

Buda/dondurma geri alınabilir: uzman **dondurulur** (silinmez), böylece yanlış
karar geri alınabilir.

---

## Testler

```bash
python -m unittest discover -s tests -v      # 412 test
python -m unittest tests.test_headroom -v    # kaynak nezaketi
python -m unittest tests.test_knowledge -v   # bilgi motoru
python -m unittest tests.test_learning -v    # öğrenme + sağlayıcı yedekleme
python -m unittest tests.test_study -v       # kendi kendine geliştirme + web
```

Test dosyaları: `test_core.py`, `test_knowledge.py`, `test_neural.py`,
`test_headroom.py`, `test_app.py`, `test_learning.py`, `test_study.py`.
Tümü **gerçek** bileşenleri geçici dizinlerde kullanır — geliştiricinin
kasasını kirletmez.

---

## Müfredat Eğitimi (Sıfırdan Öğretim)

Ajan deneyimleri **dar** bir veri kümesi üretir (aynı tür görevler, sınırlı
sözcük dağarcığı). Temel kavramları öğretmek için yapılandırılmış müfredat
gerekir: `neurocore/neural/curriculum.py`.

### Katmanlar

| Katman | İçerik | Ders |
|---|---|---|
| **İkili Sayı Sistemi** | bit/bayt, mantık kapıları, ASCII/UTF-8, hex, hata denetimi, örnekleme | 10 |
| **Donanım ve Açılış** | CPU döngüsü, bellek hiyerarşisi, veri yolu, POST→BIOS→UEFI→bootloader, GPU, sanal bellek | 8 |
| **İşletim Sistemi** | çekirdek/kullanıcı alanı, süreç/thread, zamanlama, deadlock, dosya sistemi, kabuk, IPC, sürücü | 10 |
| **Programlama Dilleri** | tipler, akış, fonksiyon, veri yapıları, hata yönetimi, OOP, Big-O, özyineleme, test | 10 |
| **Kütüphaneler** | pathlib, json, numpy, torch, asyncio, urllib, Pillow, psutil, FastAPI | 10 |
| **Blender** | bpy, sahne, ilkel şekiller, dönüşüm, malzeme, ışık/kamera, render, glTF/FBX, sık hatalar | 10 |

**464 ders** (58 temel ders × 8 soru kalıbı), **243.202 karakter**.

Soru kalıpları (`nedir`, `nasıl çalışır`, `neden önemli`, `nasıl uygulanır`,
`örnek göster`, `özetle`, `sık hatalar`, `karşılaştır`) yalnızca **soru
biçimini** değiştirir — hedef metin aynı kalır, yani **bilgi uydurulmaz**.

### Tokenizer yenileme (neden şart?)

Ölçüldü: eski sözlük (1.259) teknik metinde **1,73 karakter/token** üretiyordu
— neredeyse her karakter bir token. Müfredatla yeniden eğitildiğinde:

```
Sözlük    : 1.259 → 4.301
Verimlilik: 1,73 → 4,16 karakter/token   (2,4× kazanç)
```

Aynı donanımda **2,4 kat fazla içerik** öğrenmek demektir. Sözlük büyüyünce
gömme katmanı da büyütülür (`TransformerBackbone.resize_vocab`): eğitilmiş
satırlar korunur, yalnızca yeni satırlar başlatılır ve `tie_embeddings`
paylaşımı bozulmaz.

### Çalıştırma

```bash
python tools/mufredat_egit.py --tur 8 --adim 70 --dakika 6   # hazırlık + turlar
python tools/mufredat_egit.py --durum                        # ilerlemeyi yazdır
python tools/uretim_testi.py                                 # üretim kalitesini ölç
```

`--adim` **önemlidir**: otomatik plan küçük veri kümesinde
`max(10, örnek×epoch/(batch×accum))` formülüyle **10 adıma yuvarlanır**. 1051
örnekle bu 70 adıma çıkar.

İlerleme: `.neurocore/curriculum_progress.json`

### Ölçülen tuzaklar (hepsi düzeltildi)

Bu hat çalışır hâle gelene kadar altı gerçek hata bulundu. Hepsi
**ölçümle** doğrulandı:

| # | Belirti | Kök neden | Düzeltme |
|---|---|---|---|
| 1 | 8 turun 1'i tamamlandı | `engine.shutdown()` VRAM'i bırakmıyor (1450 MiB'de kalıyor) | `gc.collect()` + `torch.cuda.empty_cache()`; tur başına motor yeniden yükleme |
| 2 | 400 adım isteği 2 adımda bitti | Gözcü kendi ayırdığı VRAM'i (%94) "sistem dolu" sanıp 60 sn sonra iptal ediyor | `user_initiated` ile bekleme 600 sn, VRAM tavanı %98, baskı sınırı 200 |
| 3 | 6 olayda erken durma | Gözcü `max_pressure_events=6`, her 2 adımda ölçüm → **12 adımda** stop | Kullanıcı istekli eğitimde 200 |
| 4 | `unscale_() has already been called` | NaN gradyan dalı `step()` çağırmadan `continue` ediyor; `GradScaler` bayrağı sıfırlanmıyor | Daldan önce `_make_scaler()` ile yeni scaler |
| 5 | Her tur 7 adımda kesiliyor | Kasa veri kümesi 58 not ≈ 57 bin token = 1 epoch | Genişletilmiş müfredat (464 ders) → 1051 örnek, 91 bin token |
| 6 | 9. turdan sonra grup boyutu 32→1 | Turlar arası VRAM birikimi (10,6 GB) | Taze süreç + VRAM boşaltma doğrulaması |

### Ölçülen sonuç

```
Tokenizer    : 1.259 → 4.301 sözlük, 1,73 → 4,16 karakter/token (2,4×)
Gömme        : 1.259 → 4.301 satır (bağ korundu, eğitilmiş satırlar korundu)
Veri kümesi  : 1.051 örnek / 91.159 token
Eğitim       : 122 → 809 adım (%101 eşik) — üretim eşiği (800) AŞILDI
Kayıp        : 4,62 → 0,24 (en iyi tur)
Token görüldü: 3,4 milyon
Testler      : 530 OK
```

### Model ne yapabildi, ne yapamadı (dürüst ölçüm)

809 adımda yapılan ölçümler (`tools/yetki_olcumu.py`,
`tools/blender_kod_secimi.py`):

| Yetenek | Sonuç | Değerlendirme |
|---|---|---|
| **Ezber** (eğitim istemi → hedef) | **0/3** | ❌ Hedefi geri getiremiyor |
| **Anlam sıralaması** (anlamlı devam) | −7,15 vs saçma −8,87 | ✅ Doğru sıralıyor |
| **Blender kod seçimi** (4 aday) | **7/10 = %70** | ✅ Rastgele %25 → **2,8×** |
| **Katman sınıflama** (6 aday) | 6/12 = %50 | ✅ Rastgele %17 → 3× |
| **Akıcı metin üretimi** | Tutarsız | ❌ Dilbilgisi kurulamıyor |
| **bpy kodu üretimi** | Bozuk | ❌ Geçerli kod yazamıyor |

**Sonuç:** Model **yazamıyor ama tanıyor.** Bu bir kusur değil, ölçek
sınırıdır: 208M parametre (GPT-3.5'in ~%1'i) ve 3,4M token (ciddi bir model
için ~10.000× az). Anlamlı metin üretimi bu ölçekte mümkün değildir.

**Ama tanıma yeteneği işe yarar** — ve ajan mimarisinde kullanılıyor:
`local.rank_candidates` aracı, bulut modelinin ürettiği aday kodları yerel
modelin olasılığıyla süzer.

---

## Yerel Model ile Aday Sıralama

Bulut modeli birden fazla çözüm ürettiğinde **hangisi daha tutarlı?**
Bu soruyu yerel model yanıtlar: her adayı istemle birlikte kodlar ve
**yalnızca devam kısmının** ortalama log olasılığını hesaplar
(`NeuroCoreModel.sequence_logprob`).

```
Yetenekler                         Güvenlik   Ne döner
local.rank_candidates   (read)  →  sıralama, en iyi aday, güven düzeyi
local.model_status      (read)  →  eğitim adımı, üretim hazır mı, sayaçlar
```

Örnek (gerçek çağrı):

```
İSTEM: Blender'da sahnedeki tüm nesneleri silmek için hangi kod?
  1. bpy.ops.object.select_all(action='SELECT') + delete   ← seçildi (fark 1,13)
  2. bpy.ops.mesh.primitive_cube_add(size=2)
  3. bpy.ops.export_scene.gltf(filepath='m.glb')
```

**Dürüst sınırlar:**
* İsabet **%70**; sonuç *öneri*dir, kesin karar değil.
* `uncertain: true` ise skor farkı `min_margin` altındadır → **karar vermeyin**.
* Kayıtlı alan dışında (müfredat dışı konular) güvenilmez.
* Kelime torbası sezgisi **kullanılmaz**; skor modelin kendi olasılığıdır.

---

## Ölçek Büyütme ve Düşünme

Yerel modelin **düşünmesini** sağlamak üç parçalı bir iştir: ölçek, akıl
yürütme verisi ve seçme yeteneğini kullanma. Üçü de kuruldu.

### Donanım sınırı (ölçüldü)

```
GPU : GTX 1080 Ti, 11.264 MiB VRAM, ~10.200 MiB kullanılabilir
RAM : 31,9 GB (23,8 GB boş)

Mevcut model (208M, 8 katman, 4 uzman) → eğitim ≈ 1,8 GB
  +4 katman (316M)          → ≈ 2,7 GB
  +8 katman (422M)          → ≈ 3,5 GB
  16 katman + 12 uzman (738M) → ≈ 6,2 GB   ← üst sınır
```

Yani bu makinede **~740M parametreye kadar** büyütülebilir. Ama ölçüm
gösterdi ki önce **veri** gerekir (aşağıya bakın).

### 1. Akıl Yürütme Müfredatı

`neurocore/neural/reasoning.py` — beş düşünme biçimi, her biri farklı bir
soruyu yanıtlar:

| Biçim | Soru | Ders |
|---|---|---|
| **ANALYZE** | Bu problem gerçekte ne istiyor? | 4 |
| **COMPARE** | Hangi yaklaşım daha etkili, neden? | 4 |
| **DECIDE** | Hangi yolu seçiyorum ve neden? | 4 |
| **DIAGNOSE** | Bu hata neden oluyor? | 4 |
| **INTERPRET** | Bu bilgi ne anlama geliyor? | 4 |

İzler **adım adım**dır (`ADIM 1…KARAR…GEREKÇE`); model yalnızca cevabı
değil, cevaba giden yolu öğrenir. Örnek:

```
DÜŞÜN: Eğitim kaybı 0,2 ama üretim tutarsız. İlk ne yapmalı?
ADIM 1: Düşük kayıp eğitim verisinde iyi olduğunu gösterir.
ADIM 2: Üretim farklı dağılım → genelleme sorunu.
ADIM 3: 3,4M token / 208M parametre = 0,016 token/parametre.
        Sağlıklı oran ~20 → veri ~1.200× eksik.
KARAR: Sorun veri ölçeği, model değil.
GEREKÇE: Kayıp tek başına yeterli ölçü değildir; orana bakılır.
```

İzler **ölçülen gerçek hatalardan** türetildi: `unscale_` hatası, VRAM
sızıntısı, 403 bot koruması, `bpy.ops` bağlam tuzağı, adım sınırı.

### 2. Ölçekli Veri Üretimi

`neurocore/neural/scale.py` — her ders dört eksende çoğaltılır
(biçim × düşünme × kitle × kod):

```
Temel dersler      : 78  (58 bilgi + 20 düşünme)
Üretilen örnek     : 2.030
Karakter           : 1.219.040
Token              : 271.457     (önceki 91.159 → 3× artış)
Tokenizer sözlüğü  : 4.679
Verim              : 4,49 karakter/token
```

⚠ **Dürüstlük kuralı:** Şablonlar **bilgi uydurmaz**. Her iz dersin kendi
gövdesinden türetilir; yalnızca sunum biçimi değişir. Uydurma bilgi modele
zehir olurdu. Bu kural `test_reasoning.py` ile kilitli
(`test_target_derives_from_lesson`).

### 3. Düşünme Döngüsü

`neurocore/agent/thinking.py` — bulut modeli **üretir**, yerel model
**seçer**, belirsizlikte buluta düşülür:

```
1. ÜRET    → bulut modeli N yaklaşım önerir
2. SEÇ     → yerel model kendi olasılığıyla sıralar
3. DOĞRULA → fark eşiğin altındaysa yerel seçim KULLANILMAZ
4. KAYDET  → .neurocore/thinking_journal.jsonl (isabet kanıtlanabilir olsun)
```

### Ölçülen yetenekler (rastgele taban %25)

| Yetenek | Sonuç | Kazanç |
|---|---|---|
| **Blender kod seçimi** | **%70** | 2,8× |
| **Yol/yaklaşım seçimi** | %50 | 2,0× |
| **Teşhis** | %50 | 2,0× |
| Serbest üretim | Yok | — |

**Sonuç:** Bu ölçekte "düşünme" **yazmakla değil, seçmekle** olur. Yerel
model akıcı metin üretemiyor ama **doğru seçeneği tanıyor** — ve bu, ajan
mimarisinde gerçek bir iş bölümü sağlar.

### Eğitim sonucu

```
Adım    : 122 → 1.175  (üretim eşiği 800 aşıldı)
Token   : 3,4M → 6,0M
Kayıp   : 3,42 → 0,09  (tur 1);  0,28 → 0,09 (tur 2)
Tur süresi: ~13 dk / 150 adım (GPU %98, 11 GB)
```

**Tur arası VRAM temizliği şart:** ölçüldü — eğitim VRAM'i doldurunca
sonraki turun ön kontrolü `VRAM %99 ≥ %90` görüp bekliyor. Her turdan sonra
motor kapatılıp `gc.collect()` + `torch.cuda.empty_cache()` çağrılır
(10.610 MB → 5.174 MB).

---

## Parçalı İş Motoru (IDM Mantığı)

Büyük bir işi **parçalara böler**, kaynak bütçesine göre paralel çalıştırır,
sığmayanı **kuyruğa** alır ve her adımı **günlüğe** yazar.

`neurocore/core/governor.py` (bütçe) + `neurocore/core/work.py` (motor)

### Neden gerekli?

8.000.000 adımlı bir işi tek döngüyle çalıştırmak:

* kaynağı doyurur → **bilgisayar kilitlenir**,
* kesintiye uğrarsa **baştan başlar**,
* nerede kalındığı bilinmez,
* paralelleştirilemez (hepsi seri).

Motor dört sorunu birden çözer: **bölme**, **sıralama**,
**kesintiye dayanıklılık**, **izlenebilirlik**.

### İki kural

| Kural | Anlamı |
|---|---|
| **Bütçe** | Boştakinin %90'ı. CPU %50 dolu → %45; %10 dolu → %81. Sert tavan %85. |
| **Parça** | Toplam kapasitenin küçük dilimi (varsayılan %5). Küçük parça → **çok parça paralel** → kaynak doygunluğu yok. |

Parça boyu ile eşzamanlılık **tutarlıdır**: `paralel = 1 / chunk_ratio`.
Parçayı küçültmek paralelliği artırır.

### Sisteme otomatik uyum (ölçüldü)

```
Sistem                     Bütçe      Parça      Paralel   Kısıt
KÜÇÜK (4 GB, %10 dolu)     %81.0     205 MB        16      ram
ORTA  (32 GB, %50 dolu)    %45.0    1638 MB         9      ram
YOĞUN (4 GB, %85 dolu)     %13.5     205 MB         2      ram
```

**Kod değişmez** — aynı motor 4 GB'lık bir makinede de 32 GB'lıkta da çalışır.

### Paralel yürütme (ölçüldü)

| Adım | Süre | Seri olsaydı | Hızlanma |
|---|---|---|---|
| 8 | 0,21 sn | 1,6 sn | **7,6×** |
| 16 | 0,22 sn | 3,2 sn | **14,7×** |
| 24 | 0,42 sn | 4,8 sn | **11,4×** |
| 32 | 0,45 sn | 6,4 sn | **14,4×** |

### Kaynak yetmezse kuyruk

```
Bütçe   : 2 paralel, RAM 205 MB
Çalışan : 1 adım (hafif olan)
Kuyrukta: 1 parça (ağır olan)
  c00000 [bitti          ] bağımlılık katmanı 0
  c00001 [kaynak bekliyor] kaynağa sığmıyor
                           (ihtiyaç {'ram_mb': 2000.0})
```

### Günlük ve algoritma

Her parça ve adım `work_journal.jsonl` dosyasına yazılır (**JSONL**: kesinti
anında yarım satır atlanır, tüm günlük kaybolmaz).

```
Günlük kaydı : 98 satır     (adım 48, parça 48, başladı 1, bitti 1)
Devam durumu : 48 adım tamamlanmış   ← kesintiden sonra kaldığı yer
```

`engine.algorithm()` yapılan işi **adım adım, süreleriyle** yeniden üretir:

```
AMAÇ: 8M adımlı proje
ÖLÇEK: 48 adım, 48 parça

   1. [code] Modül 0 (166,666 alt adım)
      kaynak: cpu=2, ram_mb=120
      sonuç: başarılı (0.05 sn)
   2. [code] Modül 1 (166,666 alt adım) (bağımlı: m000)
      sonuç: başarılı (0.06 sn)
```

### Ölçülen tuzaklar

| Belirti | Kök neden |
|---|---|
| 40 adımın **hiçbiri** çalışmadı | `_deps_ok` parça **içi** bağımlılıkları da bekliyordu; parça zaten kendi içinde sıralı |
| Küçük sistemde tek adım çalışmadı | Parça boyu **adım sayısına** göre seçiliyordu; 5 adım × 400 MB = 2000 MB, bütçe 205 MB |
| Paralellik **1,0×** (hiç hızlanma yok) | "Küçük dilimleri birleştir" geçişi parçaları geri birleştirip 8 bağımsız adımı tek parçaya indiriyordu |
| Ağır iş yine de çalıştı | `fits` denetimi bütçe yerine ham ihtiyaca bakıyordu |

### Gösterim

```bash
python tools/parcali_motor_gosterim.py     # 8M adım senaryosu, üç sistem profili
```

**Benzetim sonucu:** 48 modül × 166.666 alt adım = **7.999.968 alt adım**,
0,47 saniyede, 0 hata, kaynak sınırı aşılmadan.

---

### Ajan Köprüsü — Görevleri Motora Bağlar

`neurocore/agent/bridge.py` — ajan işlerini `WorkStep`'e çevirir.

```
Ajan planı  →  AgentBridge.plan_steps()  →  WorkStep[]  →  parçalı motor
```

Kaynak ihtiyacı **araç türünden** tahmin edilir (ölçüme dayalı):

| Araç | CPU | RAM | Tahmini süre |
|---|---|---|---|
| `code.read_file` | %0,5 | 20 MB | 0,5 sn |
| `code.create_file` | %1 | 30 MB | 1,0 sn |
| `code.run_tests` | %6 | 300 MB | 30 sn |
| `blender.run_script` | %10 | 700 MB | 20 sn |
| `blender.render` | %40 | 1500 MB | 120 sn |

`needs_previous: true` olan adım bir öncekine zincirlenir; aksi hâlde
bağımsız sayılır ve **paralel** çalışır.

```
Girdi planı:
  [Dosyaları listele] [A yaz] [B yaz] [C yaz] [Test çalıştır ← zincirli]

Sonuç (4 GB sistem):
  c00000 [hafif]  c00001 [hafif]  c00002 [hafif]   ← paralel çalışır
  c00003 [ağır]   c00004 [ağır]                    ← kuyruğa alınır
```

**Araç çağrısı `registry.call()` ile yapılır** — `handler(**args)` doğrudan
çağrılmaz; çünkü `call()` özerklik politikasını, onay akışını ve denetim
günlüğünü uygular. Doğrudan çağrı bu güvenlik katmanlarını atlardı.

### ⚠ Onay Davranışı

Denetimli modda **yazma ve çalıştırma** araçları onay ister. Onay verilmezse
o adımlar `başarısız` olur:

```
[başarısız] A dosyası yaz
  RuntimeError: code.create_file: Denetimli modunda yazma işlemleri
                için onay gerekir
```

Bu bir hata değil, **güvenlik katmanıdır**. Onay gerektirmeyen toplu çalışma
için özerkliği `Güvenilir` yapın veya araçları önceden onaylayın.
`/api/work/plan` yanıtı hangi adımların onay isteyebileceğini bildirir
(`approval_may_be_needed`).

### API Uçları

| Uç | İş |
|---|---|
| `GET /api/work/status` | Durum: adım/parça sayısı, bütçe, kuyruk, son adımlar |
| `POST /api/work/plan` | Planı kur ve **arka planda** çalıştır |
| `POST /api/work/resume` | Günlükten **kalan** adımlarla devam et |
| `GET /api/work/algorithm` | Yapılan işi **algoritma** olarak döndür |

### Kuyruk Israrı

`run()` tek geçiştir; kaynak yetmezse parçalar bekler. `run_until_done()`
kaynak açılana kadar **kendiliğinden** dener:

```python
rapor = motor.run_until_done(max_wait_s=3600, poll_s=20)
print(rapor["waited_s"], rapor["still_blocked"])   # dürüst rapor
```

Süre aşılırsa **sessizce pes etmez**: `still_blocked` alanında hangi
parçanın, ne kadar kaynak beklediğini söyler.

`deferred_summary()` kuyruğun özetini verir:

```python
{"count": 1, "steps": 1,
 "reason": "kaynağa sığmıyor (ihtiyaç {'ram_mb': 2000.0}, bütçe RAM 205MB)"}
```

### Idempotentlik

`_run_step` zaten tamamlanmış adımı **tekrar çalıştırmaz**. Bu olmadan
`append_plan` sonrası yeniden çalıştırma bir dosyayı iki kez yazardı —
ölçüldü: bir adım iki kez koştu.

| Yöntem | İş |
|---|---|
| `append_plan(amaç, adımlar)` | Var olan plana adım ekler, tamamlananları korur |
| `resume()` | Günlükten kalanları bulur, yeni motorla devam eder |
| `pending_steps()` | Tamamlanmamış adımları listeler |

### Ölçülen tuzaklar (köprü)

| Belirti | Kök neden |
|---|---|
| Her adım "araç kayıt defteri yok" | Ajan döngüsü kayıt defterini `_registry` altında tutar; köprü `registry` arıyordu |
| Araç kayıt defteri hiç bağlanmadı | `initialize` "zaten kurulu" diyip erken dönüyordu, `loop` güncellenmiyordu |
| `AttributeError: _initialized` | Alan `__init__`'te tanımlanmamıştı |

---

### 🔴 En Kritik Bulgu: Aynı Süreçte Eğitim 13× Yavaşlıyor

Ölçüldü — eğitim VRAM'i **bırakmıyor**, sonraki tur kaynak dolu başlıyor:

| Tur | Yöntem | Adım/dk |
|---|---|---|
| 1 | aynı süreç (temiz) | **13,92** |
| 2 | aynı süreç (VRAM dolu) | 2,24 |
| 3 | aynı süreç | 10,26 |
| 4 | aynı süreç | 1,05 |
| 5 | aynı süreç | **0,87** |

**Çözüm:** her turu **ayrı Python sürecinde** çalıştırmak (CUDA bağlamı sıfırlanır):

```
python tools/taze_egitim.py --tur 12 --adim 150
```

Ölçülen sonuç — hız **sabit** kalıyor:

| Tur | Adım | Süre | Adım/dk |
|---|---|---|---|
| 1 | 50 | 295,5 sn | **10,15** |
| 2 | 50 | 295,0 sn | **10,17** |
| 3 | 50 | 294,8 sn | **10,18** |

**~12× iyileşme** ve tam öngörülebilirlik (hız aralığı 10,15–10,18).
Model her tur sonunda diske yazıldığı için **kayıp olmaz**.

`TrainingWorkflow.run_fresh_rounds(rounds, round_runner)` bu yöntemi
kütüphane olarak sunar; süreç yönetimini çağırana bırakır (bir kütüphane
kendi kendini yeniden başlatamaz).

### Ölçülen hız — özet tablo

| Ölçüm | Sonuç |
|---|---|
| **Paralellik (8 iş)** | 0,21 sn — **7,6×** hızlanma |
| **Paralellik (16 iş)** | 0,41 sn — **7,7×** |
| **Paralellik (24 iş)** | 0,43 sn — **11,1×** |
| **Eğitim (taze süreç)** | **10,17 adım/dk** (sabit) |
| **Eğitim (aynı süreç)** | 0,87 adım/dk (5. turda) |
| Günlük yazma (20 adım) | 0,028 sn / 5.158 bayt |
| Devam okuması | 0,0004 sn |
| Kuyruk koruması | ✅ ağır iş çalıştırılmadı |

### Bütçenin sisteme uyumu

```
Sistem                        Bütçe   Parça RAM  Paralel   Kısıt
KÜÇÜK  (4 GB, %10 dolu)     % 81.0      205 MB       16     ram
ORTA   (32 GB, %50 dolu)    % 45.0     1638 MB        9     ram
BÜYÜK  (32 GB, %10 dolu)    % 72.0     1638 MB       14     ram
YOĞUN  (4 GB, %85 dolu)     % 13.5      205 MB        2     ram
```

---

## "Bilmiyorsa Öğrenerek Yapar mı?" — Evet

Kullanıcının sorusu: *"bir iş verildiğinde bilmiyorsa öğrenerek yapacak mı?"*
Cevap **evet** — ve ölçüldü.

### Ölçülen boşluk

`SelfStudy` (öğretmen–öğrenci döngüsü) ve `LearningRecorder` **vardı** ama
ajana **bağlı değildi**. Ajanın 65 aracı arasında `knowledge.*` (arama)
vardı, `learn.*` **yoktu**. Yani ajan bilmediği bir konuda tahmin yürütüyor
ya da "bilmiyorum" diyordu.

### Çözüm: üç araç

| Araç | İş |
|---|---|
| `knowledge.ask` | Önce kasaya sor; yetersizse **öğren** |
| `learn.topic` | Sıfırdan öğren: soru → kaynak indir → doğrula → kasaya yaz |
| `learn.status` | Ne öğrenildi, hangi kaynakla |

Hepsi `read` güvenliğinde: kasa yazımı **araç içinde** olur ve doğrulanmış
kaynağa dayanır. `write` istemek, Denetimli modda her öğrenme için onay
gerektirirdi ve akış kullanılamaz hâle gelirdi.

### Uçtan uca kanıt (gerçek öğretmen, gerçek kaynak)

```
KONU: Rust ownership ve borrow checker kuralları

1) Kasa (önce)  : 0 alakalı sonuç  → BİLMİYOR
2) Öğreniliyor  : 63,8 sn
   soruyu üreten: teacher
   kaynaklar    : 3
     - https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
     - https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
   cevap        : "Rust'ın sahiplik tasarımının dayandığı ilk ilke, her
                   değerin bir sahibi olması ve aynı anda yalnızca bir
                   sahibinin bulunmasıdır [2]..."
3) Kalite kapısı: GEÇTİ
4) Kasa (sonra) : 1 sonuç → YAZILDI
5) Tekrar soruş : kaynak=vault (artık kasadan geliyor)
```

### İki kritik hata bulundu ve düzeltildi

**1. Alakalılık süzgeci yoktu** — ajan yanlışlıkla "biliyorum" diyordu:

```
"zzz qqq bilinmeyen konu"  →  "İşletim Sistemi — IPC" (skor 0,729)
"CharacterBody3D"          →  alakasız kasa notları
```

Skor tek başına yeterli değil; **terim örtüşmesi** de aranmalı. Üstelik
sorgunun **ayırt edici terimi** (en uzunu) sonuçta geçmeli — "konu" gibi
genel kelimeler yetmemeli. Düzeltmeden sonra:

```
CharacterBody3D              → 1 sonuç  → BİLMİYOR  ✓
Blender malzeme renklendirme → 5 sonuç  → BİLİYOR   ✓
zzz qqq bilinmeyen konu      → 0 sonuç  → BİLMİYOR  ✓
kuantum dolanıklık protokolü → 0 sonuç  → BİLMİYOR  ✓
```

**2. Kalite kapısı yoktu** — değersiz cevap kasaya yazılıyordu:

Öğretmen yanlış kaynak önerdiğinde (PyTorch sorusuna **NVIDIA CUDA
kılavuzu**) kendi cevabında şöyle diyordu:

> *"Verilen kaynak [1] ile bu soru **tam olarak cevaplanamaz**. Çünkü [1],
> NVIDIA CUDA Programming Guide'ın giriş metnidir."*

Bu cevap **başarılı** sayılıp kasaya yazılıyordu. Böyle bir not kasa için
**zehir**dir: model ondan öğrenir. Artık yetersizlik itirafı görülürse tur
**başarısız** sayılır ve **kasaya yazılmaz**.

### Dört dürüstlük kuralı

| Durum | Davranış |
|---|---|
| Kasa biliyor | Öğretmen **çağrılmaz** (API kotası harcanmaz) |
| Kasa bilmiyor | **Öğrenir** — öğretmen + gerçek kaynak indirme |
| Öğretmen yok | **Dürüstçe bildirir**: "öğretmen bağlı değil, öğrenme yapılamıyor" |
| Öğrenme başarısız | **Tahmin yürütmez**; "kaynak bulunamadı, kullanıcıya bildirilmeli" der |

Son kural önemli: ajan uydurmak yerine **bilmediğini söyler**. Araç
açıklaması da bu yönde: *"Bilmediğin bir konuda tahmin yürütmek yerine bunu
kullan."*

### Kullanım

```bash
python tools/ogrenme_testi.py --konu "Rust ownership kuralları"
```

---

## Ölçek Danışmanı — Veri mi, Model mi, Eğitim mi?

`tools/olcek_danismani.py` — **neden** modelin akıcı üretemediğini ölçer ve
ne yapılması gerektiğini söyler. Tahmin yok; oranlar ve donanım sınırı
ölçülür.

### Ölçülen durum

```
Parametre (toplam) : 211.255.328
Veri tokenu        : 271.726
ORAN               : 0,001286 token/parametre
Sağlıklı oran      : 20 token/parametre
EKSİK              : ~15.549× daha fazla veri

Eğitim VRAM        : 1.773 MB (bütçe 10.200 MB)
Donanım sınırı     : ~1,2 milyar parametre
```

### Danışmanın verdiği karar

```
DARBOĞAZ : DATA
ÖNERİ    : Model veriye göre çok büyük. Daha çok adım atmak aynı veriyi
           tekrarlar → ezberler; modeli büyütmek oranı daha da
           kötüleştirir. Doğru hamle: VERİ ÜRETMEK.
```

Bu önemli bir tuzak: "daha çok eğitelim" içgüdüsü **yanlış**. Ölçüm
gösteriyor ki sorun eğitim süresi değil, veri hacmi.

### Donanım sınırı (ölçüldü, GTX 1080 Ti / 11 GB)

```
Eğitim belleği ≈ parametre × 4 bayt × 2,2
~10 GB bütçede üst sınır : ~1,2 milyar parametre (teorik)
                            ~740 milyon (AMP + etkinleştirmelerle güvenli)
```

---

## Açık Konular — Düzeltmeler

### 1. Onay Akışı (toplu işi bloke ediyordu)

Denetimli modda yazma araçları onay ister. Onay bekleyen adım **"başarısız"
sayılıyordu** — ölçüldü: 3 adımlık planın 2'si çökmüş görünüyordu. Oysa iş
**bekliyor**, bozulmamış.

**Çözüm:** `WorkWaiting` istisnası ve `WAITING` ("onay bekliyor") durumu.

```python
# Motor artık beklemeyi başarısızlıktan ayırır
except WorkWaiting as exc:
    step.status = WAITING      # FAILED değil
    step.result = exc.reason
```

İki yeni uç:

| Uç | İş |
|---|---|
| `GET /api/work/approvals` | Onay bekleyen adımlar |
| `POST /api/work/approve` | Onayla (`allow: true`) veya reddet (`false`) |

Onay verilince adım **yeniden çalıştırılır**; iş durduğu yerden devam eder.
`/api/work/status` yanıtı `approvals` alanını da taşır.

**Bulunan hata:** Onay sonrası `_call_tool`'u tekrar çağırmak **sonsuz
döngü** kuruyordu (kayıt defteri onayı yeni tüketiyor, yine onay istiyor).
Süreç askıda kaldı. `_call_tool_raw` ile ham çağrıya geçildi.

### 2. Eğitim Hızı

Aynı süreçte turlar tekrarlandığında hız **13× düşüyordu** (13,92 → 0,87
adım/dk) çünkü eğitim VRAM'i bırakmıyor. **Taze süreç yöntemi** hızı sabit
tutuyor: 10,15 / 10,17 / 10,18 adım/dk.

### 3. Veri (asıl darboğaz)

İki yeni üretim yolu eklendi:

| Yol | Ne ekler |
|---|---|
| **Diyalog** (`dialogues`) | Netleştirme, derinleştirme, hata düzeltme turları |
| **Derinlik** (`depth`) | Ders başına 3→18 soru kalıbı (6 seviye) |

```
Şablon verisi : 271.726 → 418.720 token  (1,5×)
Kasa içeriği  : ~126.605 token (956 not)
TOPLAM        : ~545.325 token            (2,0×)
```

**Dürüst değerlendirme:** Bu 2× iyileşme **gerçek** ama **yetersiz**.
Sağlıklı oran için gereken ~4,2 milyar token'a ulaşmak şablon üretimiyle
mümkün değil. Gerçek ölçek için gereken:

* İnternetten toplu metin/kod toplama (lisans uyumlu)
* Öğretmen modelle **sentetik diyalog üretimi** (milyonlarca tur)
* Veya modeli veriyle uyumlu boyuta indirmek (~13,5 bin parametre —
  anlamsız)

Bu, tek oturumda kapanacak bir açık değildir; ölçüm ve yol haritası
`tools/olcek_danismani.py` ile her zaman güncel tutulur.

---

## Ağ (Network) Müfredatı

`neurocore/neural/network.py` — ağ yapısını **en temel bit seviyesinden**
dağıtık sunucu mimarilerine ve oyun ağına kadar öğretir.

### Altı katman, 57 ders

| Katman | İçerik | Ders |
|---|---|---|
| **Ağ Temelleri** | Katmanlı yapı, OSI/TCP-IP, IP/CIDR, MAC/ARP, yönlendirme, NAT/CGNAT, paket/MTU, gecikme/jitter, DNS/SRV, firewall | 10 |
| **Taşıma Katmanı** | TCP (üçlü el sıkışma, akış/tıkanıklık), UDP, TCP-UDP kararı, port haritası, soket programlama, QUIC/HTTP3, STUN/TURN/hole punching, WebSocket | 9 |
| **Uygulama Protokolleri** | HTTP ve durum kodları, HTTP/1.1-2-3, TLS/sertifika, REST/gRPC/GraphQL, önbellek/CDN, kimlik (JWT/OAuth), SMTP/IMAP/SPF, NTP | 8 |
| **Sunucu Mimarileri** | İstemci-sunucu/P2P, proxy, **ters proxy**, yük dengeleme, **mainserver↔proxy ilişkisi**, CAP/dağıtık, mesaj kuyruğu, Docker/Kubernetes, gözlemlenebilirlik, **listen/dedicated/authoritative**, tam akış | 11 |
| **Oyun Sunucuları** | Tick döngüsü, istemci tahmini, **lag compensation**, interpolation, yetkili sunucu/hile, ilgi yönetimi, bant bütçesi, sunucu yaşam döngüsü, eşleştirme/ELO, anti-cheat | 10 |
| **Godot ve Ağ** | MultiplayerAPI/ENet, **RPC** (`@rpc` modları), MultiplayerSynchronizer, MultiplayerSpawner, ENet/WebSocket/WebRTC seçimi, headless sunucu, **güvenlik**, HTML5+WebSocket, hata ayıklama | 9 |

**Toplam 33.449 karakter**, 18 kod örneği (3 geçerli Python + 15 kabuk/
diyagram bloğu).

### Kritik teknik gerçekler **testle kilitli**

Yanlış bilgi öğretmemek için gerçek protokol değerleri test edilir:

```
Portlar     : 22 SSH, 53 DNS, 80 HTTP, 443 HTTPS, 3306 MySQL, 5432 PostgreSQL
TCP         : SYN → SYN-ACK → ACK (üçlü el sıkışma)
UDP         : bağlantısız, onay yok, yeniden iletim yok
OSI         : 7 katman, isimleriyle
MTU         : Ethernet 1500 bayt
CAP         : tutarlılık / erişilebilirlik / bölünme toleransı
Godot API   : multiplayer.multiplayer_peer, create_server, create_client,
              get_remote_sender_id, is_server, set_multiplayer_authority
Godot peer  : ENetMultiplayerPeer, WebSocketMultiplayerPeer
RPC modları : any_peer, call_remote, reliable, unreliable_ordered
```

### Müfredata entegrasyon

Ağ katmanları birleşik müfredata bağlandı — tokenizer, kasa ve eğitim
hattı hepsini birlikte görür:

```
Önce : 58 ders  →  271.726 token
Sonra: 115 ders →  859.537 token   (2,05× artış, ağ dahil)
```

6 soru kalıbıyla genişletildiğinde **690 ders / 389.171 karakter**.

---

## Kendi Kendine Öğrenme ve İnternet Araştırması

Kullanıcının sorusu: *"modelimiz kendisi internetten araştırıp öğrenebiliyor mu?"*

**Cevap: Evet — ama kanal önemli.** İki ayrı yol var ve güvenilirlikleri
ölçüldü.

### Kanal 1 — Arama motoru (kırılgan)

`web.search_web()` üç motoru sırayla dener:

| Motor | Durum (ölçüldü) |
|---|---|
| DuckDuckGo Lite | Başta çalıştı (1-3 sn, alakalı). Yoğun kullanım sonrası **kalıcı engelledi** (15-19 sn, 0 sonuç — 90 sn beklemek geçirmedi) |
| Bing | Yanıt veriyor ama sorguyu daraltmıyor; site ana sayfalarını döndürüyor |
| Wikipedia API | Çalışıyor, bot koruması yok — ama kapsamı ansiklopedik (kod için zayıf) |
| GitHub API | Depo araması çalışıyor; **kod araması `HTTP 401`** (kimlik doğrulaması ister) |

**Sonuç:** arama motorları bot koruması uyguladığı için **plan A değildir**.

### Kanal 2 — Öğretmen önerisi + indirme doğrulaması (kanıtlı)

```
1. Öğretmen modelden adres önerisi al   (propose_sources)
2. Her adresi GERÇEKTEN indir           (fetch)
3. Yalnızca 200 dönen ve ≥300 karakter içeren kullan
4. Alakasız/boş sayfayı ele
```

**Ölçüm (3 soru, 10 adres):**

| Soru | Doğrulanan kaynak |
|---|---|
| Godot Spawner/Synchronizer farkı | 3/3 — `docs.godotengine.org/.../class_multiplayerspawner.html` vb. |
| SQLite WAL modu | 3/3 — `sqlite.org/wal.html`, `fileformat2.html#walformat` |
| Godot RPC kullanımı | 2/3 — biri 404, **elendi** |

**9/10 başarılı.** Öğretmen model resmî belgelerin adreslerini iyi bilir;
yanlış adres verdiğinde indirme adımı onu yakalar. Bu yüzden öğrenme akışı
`search_web` **başarısız olursa** öğretmen önerisine düşer — asla adres
uydurulmuş gibi kabul edilmez.

### Özerk konu keşfi (yeni)

**Ölçülen sorun:** özerk öğrenci (`SelfStudy`), sabit konu listesi boşken
kasada **zaten olan** başlıkları `"X konusunu derinleştir"` diye tekrar
soruyordu. Yani **kendi bildiğini** tekrar öğreniyor, **yeni konu
keşfetmiyordu**.

**Çözüm:** `_generate_topic()` öğretmen modele "henüz kapsanmamış ne var?"
diye sorar ve cevabı kasa içeriğine dayandırır:

```
1) Docker Compose ile çok kapsayıcılı uygulama tanımlama
2) Raft konsensüs algoritmasında lider seçimi
3) Git nesne modeli: blob, tree, commit ve tag
4) Docker imaj katmanlarında önbellek geçersizleştirme kuralları
5) PostgreSQL sorgu planı okuma
```

**5/5 yeni konu** üretildi. Derinleştirme artık **son çare**.

### Yeni ajan araçları

Uygulama **71 araç** ile çalışıyor (öncesi 69):

| Araç | İş |
|---|---|
| `web.search` | İnternette arama (motor zinciri) |
| `web.fetch` | Sayfayı indir, okunabilir metne çevir |

Her ikisi de `read` güvenliğinde. `web.fetch` **SSRF korumalı**: yerel/özel
ağ adresleri (`127.0.0.1`, `localhost`, `192.168.*`) reddedilir.

**Ölçülen tuzak:** araçlar kayıtlıydı ama ajan onları **göremiyordu** —
`TOOL_HINTS` içinde araştırma ipucu yoktu ve 24'lük araç bütçesi
`godot.`/`code.` araçlarıyla doluyordu. Araştırma ipucu **başa** alındı.

### Ajanın dürüstlüğü (gerçek çalışmadan)

Ajan `web.search` boş dönünce **uydurmadı**; durumu raporladı ve alternatif
yol buldu:

> `web_search` (5 farklı sorgu) — **Hepsi boş liste döndü** — arama servisi
> çalışmadı. Bu yüzden tahmin etmek yerine, dokümantasyonun **kendi
> kaynağını** (Godot deposundaki sınıf XML'leri) okudum; bunlar
> docs.godotengine.org sayfalarının birebir üretildiği dosyalardır.

---

## Çalışma Alanı Paneli (DSH modeli)

Paneller DeepSeek Harness'in mimarisine göre kuruldu:

| DSH paketi | Bizdeki karşılığı |
|---|---|
| `dsh-workspace` — proje kaydı, oturumlar klasöre göre gruplanır | `ChatStore` (sohbet ↔ klasör) |
| `dsh-client-ui-sidebar-files` — oturum köküne köklenen, seviye seviye okunan ağaç | **Akış → Dosyalar** sekmesi |
| `dsh-api-workspace-files` — `list()` / `read()`, **mutasyon yok** | `/api/workspace/tree`, `/api/workspace/file` |
| `dsh-client-ui-directory-picker-browse` — gezinme + **New folder** | **Klasör seç** dialogu + **+ Yeni klasör** |

### Dosya ağacı

**Akış → Dosyalar** sekmesi oturumun çalışma alanını ağaç olarak gösterir:

```
📁 WebTest          ▸   ← tıkla: genişlet
🐍 ds_test.py   151 b
🐍 selam.py      23 b   ← tıkla: önizleme açılır
📄 _tmp_probe.txt 109 b
```

- Kök **oturumun çalışma alanıdır**; kökün üstüne çıkılamaz (kum havuzu)
- Her seviye **istenildiğinde** okunur — tüm ağaç bir seferde değil
- Sıralama: önce dizinler, sonra harf sırası (büyük/küçük duyarsız)
- Dosyaya tıklayınca alt panelde önizleme (dil ve boyut etiketiyle)
- Ağaç **salt okunur**: oluşturma/silme/adlandırma yapmaz

### Yeni klasör nasıl oluşturulur?

DSH'de olduğu gibi **klasör oluşturma dizin seçicide** yapılır; dosya ağacı
yalnızca listeler:

1. Soldaki **Klasör seç** → gezgin açılır
2. İstediğiniz üst klasöre gidin
3. **+ Yeni klasör** → ad yazın → **Oluştur**
4. Yeni klasör **otomatik seçilir** (DSH davranışı), **Bu klasörü seç** ile
   sohbete bağlarsınız

Alternatif olarak ajandan isteyebilirsiniz: *"WebTest klasöründe assets adında
bir klasör oluştur"* — ajan `code.create_file` / `shell.run` araçlarıyla yapar.

---

## Sohbet Bazlı Çalışma Alanı

**Kural: sohbetin klasörü varsa çalışma alanı odur.** Klasör yoksa *genel
sohbet* modu çalışır (proje bağlamı olmadan). Ayarlar'daki genel çalışma
alanı yalnızca klasörsüz sohbetler için geçerlidir.

```
┌─ Sohbetler ──────┐
│ +                │
│ 📁 TestProjects  │  ← etkin sohbetin çalışma alanı
│ [Klasör seç] [Kaldır]
├──────────────────┤
│ sayfa yap        │  📁 site        4 mesaj
│ testleri düzelt  │  📁 oyun        9 mesaj
│ merhaba          │  💬 genel       2 mesaj
└──────────────────┘
```

- **+** → yeni sohbet (klasörsüz, genel mod)
- **Klasör seç** → gezgin açılır; seçilen klasör **o sohbete** bağlanır ve
  çalışma alanı hemen oraya geçer
- **Kaldır** → sohbet genel moda döner
- Sohbete tıklandığında geçmiş yüklenir **ve** çalışma alanı o sohbetin
  klasörüne ayarlanır
- Sohbet başlığı ilk mesajdan otomatik üretilir; ✎ ile değiştirilir,
  ⟲ ile mesajlar temizlenir (klasör bağlı kalır), ✕ ile silinir

Sohbetler `.neurocore/chats.json` içinde saklanır — klasör bağlantıları
süreç yeniden başlasa da korunur.

**Ölçülen hata:** `TestProjects` çalışma alanındayken `path="."` gönderen
`code.list_directory`, uygulamanın başlatıldığı dizini listeliyordu; çünkü
`"."` **süreç CWD'sine** çözülüyordu. Artık `""`, `"."` ve `"./"` çalışma
alanı kökü sayılır (`tests/test_chats.py::TestPathResolution`).

---

## Çalışan İşleri Durdurma

Ajan uzun süren bir komut çalıştırdığında (sunucu, derleme, uzun betik) ne
olduğunu görmek ve durdurmak gerekir.

**Sağ panelde "⚙ Çalışıyor" göstergesi** hangi komutun çalıştığını, türünü
(`powershell` / `session`), geçen süreyi ve PID'sini gösterir. Her satırdaki
**✕** yalnızca o komutu, **Tümünü durdur** hepsini sonlandırır.

```
Durdur düğmesi  →  /api/cancel  →  loop.cancel()
                                      ├─ iptal bayrağı (ajan durur)
                                      └─ shell.on_cancel() → süreç ağacı öldürülür
```

**`taskkill /T /F`** kullanılır: `terminate()` yalnızca doğrudan çocuğu
öldürür; `powershell.exe` içinde başlatılan `python` torun süreç ayakta kalır
ve boruyu tutmaya devam eder.

Ölçülen sonuç: **120 saniyelik komut, Durdur'dan 0,13 saniye sonra öldü.**

Komutlar ayrıca **adım sayacı** ile izlenir (`adım 7 / 24`); sınır dolduğunda
ajan durur ve o ana kadar yaptıklarını özetler — sınır, her araç çağrısının
bir adım sayıldığı bütçedir.

---

## Kendi Kendine Geliştirme (Öğretmen–Öğrenci)

Yerel model, API modellerini **öğretmen** olarak kullanarak kendini geliştirir.

```
1. SORU → önce **yerel model** sorar ("neyi bilmiyorum?")
          üretemiyorsa öğretmen sorar (tur durmaz)
2. KAYNAK → öğretmen resmî adres önerir → İNDİRİLİR → doğrulanır
3. CEVAP → öğretmen, yalnızca indirilen kaynağa dayalı, [1] atıflı yanıt verir
4. ÖĞRENCİ → yerel model cevaplar **ve emin olmadığı noktayı yazar**
5. ELEŞTİRİ → "YANLIŞ/EKSİK: … | DOĞRUSU: …" biçiminde hedefli geri bildirim
6. KASA → "Öğrenildi: …" notu (eğitim verisine dönüşür)
```

**Soru sormayı öğrenme:** Yerel model kendi sorusunu üretir; bu soru
sağlık kontrolünden geçer (en az 20 karakter, soru işareti şart, tek
satır). Reddedilirse öğretmen sorar. Arayüzde **"Yerel modelin soruları:
3 / 12"** göstergesi bu ilerlemeyi ölçer.

**Sağlayıcı yedeklemesi:** Öğretmen yanıt vermezse (`429 kota`, `503
yoğunluk`, ağ hatası) **sıradaki sağlayıcı denenir** — DeepSeek → Gemini →
Groq. En fazla 4 deneme; hepsi başarısız olursa tur anlaşılır hata döner.
Böylece öğrenme tek bir sağlayıcının kotasına bağlı kalmaz.

**Neden kaynak indiriliyor?** Öğretmenin hafızasından yazdığı bilgi
doğrulanmamış olurdu. Adres modelden alınır, sayfa **kendimiz indiririz**;
böylece kasaya giren bilgi gerçek kaynağa dayanır. Kaynak bulunamazsa tur
**başarısız** sayılır ve not yazılmaz.

**Arama motoru notu:** DuckDuckGo (her iki uç nokta), Mojeek ve Searx bot
koruması uyguladığı için arama yerine **öğretmen önerisi + doğrulama** yolu
seçildi. Ölçüldü: model `docs.python.org`, `peps.python.org`,
`docs.godotengine.org` gibi resmî adresleri güvenilir biçimde veriyor.

**Zehirlenme koruması:** Kasaya yalnızca **öğretmen** cevabı yazılır; yerel
modelin kendi cevabı asla eğitim hedefi olmaz (yalnızca eleştiri için
kullanılır). Yerel modelin **sorusu** hedef değil **girdi** tarafıdır, bu
yüzden kullanılabilir. Bu kural `TestStudyNoPoisoning` ile kilitlidir.

**Konular:** Ayarlar → Kendi Kendine Geliştirme bölümünden satır satır
yazılır. Boş bırakılırsa sıra: kullanıcı konuları → **başarısız ajan
çalışmaları** ("bu görev neden başarısız oldu") → kasadaki başlıklar.
Başarısız bir konu **atlanmaz**, sonraki turda yeniden denenir.

**Arka plan:** 20 dakikada bir tur; kullanıcı "Şimdi öğren" ile elle de
başlatabilir.

**Gerçekçi beklenti:** Yerel model üretim eşiğine (800 adım) ulaşana kadar
kendi sorusunu üretemez ve öğretmen sorar. Boru hattı hazırdır — model
olgunlaştıkça "Yerel modelin soruları" sayacı artmaya başlar.

---

## Öğrenme Döngüsü

Ajan yaptığı her işi **öğrenmeye** çevirir. Bu bağ olmadan ajan saatlerce
çalışsa bile yerel model tek adım ilerlemezdi.

```
Ajan görevi çalışır
        │
        ├─ 1. DENEYİM → .neurocore/learning/experiences.jsonl
        ├─ 2. KASA    → knowledge_base/  (başarılı çözüm not olarak)
        └─ 3. EĞİTİM  → models/neurocore/  (kaynak uygunsa arka planda)
```

### Ödül sinyali

| Durum | Ödül |
|---|---|
| Kendi hatasını düzeltip başardı | **+0.85** (en değerli örnek) |
| Başarılı + dosya değişti | +0.80 |
| Başarılı + araç kullandı | +0.70 |
| Başarılı sohbet (araçsız) | +0.30 |
| İptal / onay reddi | −0.30 |
| Başarısız | −0.60 |
| **Kullanıcı puanı** (👍/👎) | verilen puan (baskın) |

**Öğretici** sayılma koşulu: başarılı **ve** ödül ≥ 0.6 **ve** en az bir araç
veya dosya değişikliği. Tek satırlık sohbetler veri kümesini seyrelttiği için
dışarıda tutulur.

### Ne zaman eğitilir?

Arka plan işçisi 3 dakikada bir denetler. Eğitim **üç kapı** açıkken başlar:

1. Yeterli yeni deneyim birikmiş (varsayılan: 5)
2. İki eğitim arası bekleme geçmiş (varsayılan: 15 dakika)
3. `resources_available()` — kaynak nezaketi kapısı. Bilgisayarı boşta olan
   kaynağın %90'ından fazlasını kullanmaz ve siz çalışırken duraklar.

Ayrıca **Ayarlar → Öğrenme → Şimdi eğit** ile elle başlatılabilir; bu yol
kaynak kapısını atlar (açık istek sayılır) ama yine arka planda çalışır.

### Zehirlenme koruması

**Ajanın kararları bulut modelinden gelir; yerel modelin ürettiği metin asla
eğitim hedefi olmaz.** Aksi hâlde model kendi çıktısıyla eğitilir (model
collapse) ve kalite hızla düşer. Bu kural `tests/test_learning.py`
içindeki `TestNoPoisoning` sınıfıyla kilitlenmiştir.

### Gerçek ölçüm

```
Eğitim: 10 adım, 18.2 sn, kayıp 1.844 → 0.819, 28.160 token
Model : 80 → 90 adım (üretim eşiği 800)
```

Yerel model **üretim yapabilir** hâle gelmesi için 800 adıma ulaşmalıdır; o
noktaya kadar çıkarımsal yanıt verir. Gerçekçi beklenti: bu döngü modele
**projeye özel kalıpları** (dosya yapınız, isimlendirme alışkanlıklarınız,
tekrarlayan görevler) öğretir — bulut modelinin genel yeteneğine ulaşması
beklenmemelidir.

---

## Dizin Yapısı

```
myAi/
├── neurocore/              # ana paket (8 katman + CLI)
├── knowledge_base/         # Obsidian kasası (insan tarafından okunur)
│   ├── 10_concepts/  20_entities/  30_sources/
│   ├── 40_projects/  50_sessions/  60_skills/  70_journal/
│   └── _index/            # SQLite indeks, graf, sürümler
├── data/                   # deneyim, bellek, checkpoint, günlükler
├── models/                 # tokenizer + model ağırlıkları
├── tests/                  # test paketi
├── ARCHITECTURE.md         # tam mimari tasarımı
└── requirements.txt
```

Kasa **doğrudan Obsidian ile açılabilir** — her not normal bir markdown
dosyasıdır, frontmatter alanları NeuroCore tarafından yönetilir.

---

## Bilinen Sınırlar (dürüst liste)

| Sınır | Neden | Sonuç |
|---|---|---|
| Model **eğitilmeden üretim yapmaz** | Rastgele ağırlıklı transformer anlamsız metin üretir | Yeterli eğitim adımına (≥800) kadar çıkarımsal yanıt kullanılır |
| Çevrimiçi araştırma varsayılan kapalı | Gizlilik ve öngörülebilirlik | API anahtarı yoksa yalnızca kasa kullanılır |
| `soundfile`/`openpyxl` yoksa | Bu makinede kurulu değil | WAV çalışır, mp3/flac/xlsx net hata verir |
| Yönetici yetkisi yok | Sandbox güvenlik politikası | Bazı sistem işlemleri kısıtlı |
| Eğitim tek GPU'da | Tasarım gereği (kişisel bilgisayar hedefi) | Dağıtık eğitim yol haritasında (v0.4) |
| **Godot 4 API'si ağırlık düzeyinde öğretilemedi** | Temel model 18T token görmüş; bizim katkımız **1:24.000.000** — ölçüldü: `Seed` p=0,14 vs `frequency` p=0,00003 | **Retrieval** ile çözüldü: bağlam verilince **uydurma API 3 → 0** (`doc.godot_api_baglam`) |
| **1,5B ince ayar yapılamıyor** | GTX 1080 Ti (Pascal, 2016) **bf16 desteklemiyor**; fp16 1,5B'de NaN, fp32'de 0 MiB boş VRAM | LoRA mekanizması çalışıyor (311K parametre, 4 GB boş) ama kalite artmıyor |
| **Godot 4 derleme 3/4** | Onarım döngüsü **eksik kod ekleyemez** (`Identifier "_on_timeout" not declared`) | 6 sözdizimsel normalleştirme ile 0/4 → 3/4 |

---

## Lisans

Bu proje kişisel kullanım için geliştirilmiştir.

---

## 🎮 Godot/Blender Kod Üretimi (tur 21-38)

### Kod üretim modeli

| Model | Parametre | Veri | Kod tamamlama (40 soru) |
|---|---|---|---|
| myAi (sıfırdan) | 211M | 400k token | 1/14 (%7) |
| **SmolLM2-v5** (ince ayar) | **135M** | **1,93M token** | **25/40 (%62)** |

**Yaklaşım:** `HuggingFaceTB/SmolLM2-135M` (2T token ön-eğitim) +
kendi verimizle ince ayar (10.000 adım, 28 dk).

### Veri havuzu (5.660 not · 1,93M token)

| Kaynak | Boyut | Nitelik |
|---|---|---|
| Öğretmen üretimi | 2.291 not | ⚠️ Uydurma olabilir |
| **Blender resmî doküman** | 1.375 not | ✅ Doğrulanmış API |
| **TitoWar (gerçek proje)** | 4.920 fonksiyon | ✅ Godot 4.8'de çalışan |
| Doküman kod örnekleri | 135 not | ✅ Resmî |

### 11 adımlı kalite hattı

```
1. Model çıktısı (SmolLM2-v5)
2. kod_disi_kes     → prose kes
3. tekrar_kes       → sonsuz döngü kes
4. yazim_duzelt     → bpy_data → bpy.data
5. uydurma_filtre   → b3.b3_... reddet
6. prose_reddet     → 'The first step...' reddet
7. yapisal_denetim  → parantez, ':' , tire (Godot)
8. blender_yapisal  → zorunlu param, salt-okunur, bpy alanı
9. GODOT DERLEME    → --check-only
10. BLENDER DERLEME → --background --python
11. Kalite kapısı   → geçmezse ŞABLON
```

### Ölçülen sonuçlar

| Test | Şablon | Model |
|---|---|---|
| **Godot derleme** (12 parça) | **12/12 (%100)** | **%83-100** |
| **Blender derleme** (12 parça) | **12/12 (%100)** | — |
| Kod tamamlama | — | 25/40 (%62) |
| Nihai doğrulama | **7/7 bileşen** | — |

### Araçlar

| Araç | İşlev |
|---|---|
| `tools/godot_oyun_uret.py` | 13 dosya Godot projesi üretir |
| `tools/godot_karakter_uret.py` | 18 kemik, 6 animasyon |
| `tools/godot_ag_uret.py` | master + proxy + sunucu + istemci |
| `tools/kod_uretici.py` | SmolLM2 + 6 kalite katmanı |
| `tools/kod_son_isleme.py` | Yazım + uydurma + yapısal denetim |
| `tools/kod_kisitlari.py` | API sözlüğü (1.667) + dil kilidi |
| `tools/blender_dokuman.py` | Resmî doküman madenciliği |
| `tools/titowar_kod.py` | Gerçek Godot kodu madenciliği |
| `tools/godot_derleme_testi.py` | Godot gerçek derleme |
| `tools/blender_derleme_testi.py` | Blender gerçek derleme |
| `tools/genis_derleme.py` | 24 parça kapsam testi |
| `tools/kod_benchmark_v2.py` | 40 soru · 4 zorluk |
| `tools/guvenlik_filtresi.py` | Absürt değer eleme |
| `tools/nihai_dogrulama.py` | 7/7 bileşen doğrulama |

### Kullanım

```bash
# Tam oyun üretimi (myAi kararları + güvenlik filtresi + Godot doğrulama)
python tools/godot_oyun_uret.py --ad "oyunum" --karar --filtre --godot

# Kod üretimi testi
python tools/kod_uretici.py --test

# Derleme doğrulaması (24 parça)
python tools/genis_derleme.py --hepsi

# Kod tamamlama benchmark'ı
python tools/kod_benchmark_v2.py --model models/smollm2_kod_v5

# Nihai doğrulama (7/7 bileşen)
python tools/nihai_dogrulama.py
```
