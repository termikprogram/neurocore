# NeuroCore AI — Mimari Tasarım Dokümanı

> Sürüm: 0.1.0 · Donanım-farkında · Çok-modlu · Obsidian-tarzı bilgi motoru · Kendi kendini büyüten

---

## 1. Tasarım Felsefesi

NeuroCore tek bir "model" değil, **katmanlı bir bilişsel işletim sistemidir**. Temel ilkeler:

| İlke | Anlamı |
|---|---|
| **Donanım-farkında** | Her karar (batch, katman sayısı, önbellek, örnekleme) mevcut boş RAM/VRAM/CPU'ya göre verilir. |
| **Zarif bozulma** | Kaynak azaldığında sistem çökmez; çözünürlüğünü düşürür (kalite kademesi `ultra→minimal`). |
| **Bilgi dışsallaştırma** | Ağırlıklar değil, **kasa (vault)** bilgi taşır. Model küçük kalır, bilgi Obsidian-tarzı markdown dosyalarında yaşar. |
| **Sürekli büyüme** | Her etkileşim bir deneyimdir; deneyim → bilgi → yeni uzman (expert) → daha iyi cevap. |
| **Gözlemlenebilirlik** | Her katman `EventBus`'a olay yayar, her katman `get_stats()` verir, her işlem profillenir. |
| **Gevşek bağlılık** | Katmanlar birbirini doğrudan import etmez; `NeuroCoreEngine` üzerinden ve olaylarla konuşur. |

---

## 2. Katman Haritası

```
┌──────────────────────────────────────────────────────────────────────────┐
│  L8  EVOLUTION      Deneyim tamponu · öz-büyüme · bilgi sentezi ·        │
│                     merak güdümlü keşif · kendini değerlendirme          │
├──────────────────────────────────────────────────────────────────────────┤
│  L7  PLANNER        Görev ağacı · akıl yürütme zinciri · checkpoint ·    │
│                     yeniden planlama · paralel yürütme                   │
├──────────────────────────────────────────────────────────────────────────┤
│  L6  LANGUAGE       Yerel çeviri · dil algılama · terim sözlüğü ·        │
│                     API köprüsü (fallback)                               │
├──────────────────────────────────────────────────────────────────────────┤
│  L5  OS             Sistem envanteri · dosya/süreç/registry · sandbox ·  │
│                     yetenek keşfi · komut çalıştırma                     │
├──────────────────────────────────────────────────────────────────────────┤
│  L4  MULTIMODAL     Metin/Görüntü/Ses/Tablo kodlayıcıları · füzyon ·     │
│                     modalite yönlendirici                                │
├──────────────────────────────────────────────────────────────────────────┤
│  L3  NEURAL         MoE Transformer · bellek bankası · adaptif büyüme ·  │
│                     kendi eğitim hattı · niceleme                         │
├──────────────────────────────────────────────────────────────────────────┤
│  L2  KNOWLEDGE      Obsidian kasası · not modeli · hibrit arama ·        │
│                     bilgi grafı · sürümleme · SQLite indeks              │
├──────────────────────────────────────────────────────────────────────────┤
│  L1  HARDWARE       İzleme · kaynak tahsisi · profilleme · termal        │
├──────────────────────────────────────────────────────────────────────────┤
│  L0  CORE           SystemConfig · EventBus · NeuroCoreEngine (orkestra) │
└──────────────────────────────────────────────────────────────────────────┘
```

### Veri akışı (örnek: "GPU sıcaklığım neden yüksek?")

```
kullanıcı sorgusu
   └─> L0 Engine.query()
        ├─> L4 Multimodal: modalite tespiti (metin)
        ├─> L2 Knowledge: hibrit arama → ilgili notlar
        ├─> L1 Hardware: canlı GPU/termal anlık görüntü
        ├─> L5 OS: sürücü/süreç envanteri
        ├─> L3 Neural: gömme + MoE yönlendirme + bellek geri çağırma
        ├─> L6 Language: sorgu dili → iç dil (tr)
        ├─> L7 Planner: çok adımlıysa görev ağacı kur
        ├─> L3 Neural: bağlam füzyonu → yanıt üretimi
        └─> L2 Knowledge: yanıtı not olarak yaz (yeni bilgi)
             └─> L8 Evolution: deneyimi tampona al, kaliteyi değerlendir
```

---

## 3. Ortak Katman Sözleşmesi (Layer Contract)

Her katman modülü **aynı iskeleti** uygular. Bu, motorun katmanları tek tip sürmesini sağlar.

```python
class SomeLayer:
    name: str = "knowledge"          # katman kimliği (olay kaynağı öneki)

    def __init__(self) -> None: ...  # yalnızca durum kurulumu, ağır iş yok

    def initialize(self, config=None, event_bus=None) -> None:
        """Bağımlılıkları bağla. Ağır kaynakları burada aç (DB, thread)."""

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotent olmalı."""

    def get_stats(self) -> dict:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""

    # Katmana özgü iş metodları...
```

**Kurallar**

1. `__init__` asla I/O yapmaz; `initialize()` yapar.
2. Ağır bağımlılıklar (`torch`, `sqlite3`, `watchdog`, …) **fonksiyon içinde** import edilir → sistem eksik paketle bile ayağa kalkar.
3. Her katman `neurocore.<layer>.<module>` adlı logger kullanır (`NeuroCoreLogger.get_logger`).
4. Olay tipleri `<katman>.<olay>` biçimindedir (`knowledge.note_created`).
5. `initialize()` idempotent olmalı: iki kez çağrılırsa ikinci çağrı no-op.

---

## 4. Bilgi Kasası (Vault) Biçimi — Obsidian Uyumlu

Kasa kökü: `knowledge_base/` (config ile değiştirilebilir).

```
knowledge_base/
├── 00_meta/            # sistem notları, şema, istatistik
├── 10_concepts/        # kavramlar
├── 20_entities/        # kişi, kurum, ürün
├── 30_sources/         # araştırma kaynakları, alıntılar
├── 40_projects/        # görev/proje bağlamları
├── 50_sessions/        # konuşma özetleri
├── 60_skills/          # öğrenilmiş yordamlar (procedural memory)
├── 70_journal/         # sistem günlüğü, evrim kaydı
├── 90_archive/         # arşivlenmiş notlar
├── _index/
│   ├── neurocore.db    # SQLite: notlar + FTS5 + vektörler + kenarlar
│   └── graph.json      # dışa aktarılmış graf anlık görüntüsü
└── _attachments/       # görüntü, ses, dosya ekleri
```

### Not biçimi (frontmatter + markdown)

```markdown
---
id: kn_7f3a91c2
title: "GPU Termal Yönetimi"
kind: concept
tags: [donanim, gpu, termal]
category: learned
confidence: 0.78
importance: 0.6
created: 2026-10-05T23:40:12+03:00
updated: 2026-10-05T23:52:03+03:00
version: 3
sources: [os://nvidia-smi, file://notlar/termal.md]
links: [kn_a1b2c3d4, kn_9911ffee]
embedding_model: neurocore-hash-512
---

## Özet
...

## Detay
...

## İlişkiler
- [[kn_a1b2c3d4]] çünkü ...
```

- **Gövde metni** her zaman markdown; insan ve Obsidian tarafından doğrudan okunur.
- **Frontmatter** YAML; `python-frontmatter` yoksa kendi ayrıştırıcımız kullanılır.
- **`[[...]]` bağlantıları** iki yönlü graf kenarı üretir.
- Her yazma **sürüm** artırır ve `_index/versions/<id>/v<N>.md` altına anlık görüntü bırakır.

---

## 5. Kaynak Tahsisi Modeli

### 5.1 Nezaket Politikası — "boştakinin %90'ı" (ZORUNLU KURAL)

Sistem, kullanıcının bilgisayarını **asla doyurmaz**. Temel denklem:

```
harcanabilir = toplam × (1 − mevcut_kullanım) × nezaket_çarpanı
```

`nezaket_çarpanı = 0.90` (varsayılan). Örnekler:

| Durum | Ajanın bütçesi |
|---|---|
| CPU %0 kullanımda (tamamen boş) | toplamın %85'i (üst sınır) |
| CPU %50 kullanımda | toplamın **%45'i** |
| CPU %90 kullanımda | toplamın **%9'u** |
| RAM %50 dolu | boştakinin %90'ı, en fazla toplamın %85'i |
| VRAM %90 dolu | boştakinin %90'ı, en fazla VRAM'in %85'i |

**Üst sınırlar** (nezaket çarpanından bağımsız tavanlar):
RAM %85 · VRAM %85 · CPU %85.

**Duraklama eşikleri** — bu seviyelerin üstünde ajan **yeni ağır iş başlatmaz**:
RAM/CPU %85 · VRAM %90.

**Histerezis**: bütçe küçülmesi anında olur; büyümesi `cooldown_seconds`
(20 sn) boyunca bekletilir ve adım adım yapılır (tek adımda en fazla 1.5× RAM,
2× batch). Böylece kaynaklar bir an boşaldı diye ajan tam güce çıkıp kullanıcı
geri döndüğünde sistemi kilitlemez.

### 5.2 Zorlama noktaları

Bu politika **üç katmanda** uygulanır:

1. **`ResourceAllocator`** — her `allocate()` çağrısında bütçeyi hesaplar,
   `should_stall` bayrağını ve `pressure_level` (`NORMAL`/`ELEVATED`/`HIGH`/
   `CRITICAL`) değerini üretir.
2. **`NeuroCoreEngine.resources_available()`** — ağır iş kapısı. `train()`,
   `evolve()` ve toplu indeksleme buradan geçer; kaynaklar doluysa iş
   **başlatılmaz** ve gerekçe döner.
3. **`Trainer` içindeki `ResourceGuard`** — eğitim döngüsü her 2 adımda
   sistemi denetler:
   * `pause` → kaynaklar boşalana kadar bekler (en fazla 60 sn)
   * `reduce` → toplu iş / dizi uzunluğu küçültülür
   * `stop` → 6 baskı olayından sonra tur sonlandırılır (ilerleme checkpoint'lenir)

### 5.3 Boru hattı

```
SystemSnapshot (CPU%, RAM boş, VRAM boş, sıcaklık, disk)
        │
        ├─ Nezaket hesabı        : (1 − kullanım) × 0.90, üst sınırla kırpılır
        ├─ device seçimi         : VRAM ≥ 2GB → cuda, değilse cpu
        ├─ embedding_dim         : ajanın VRAM bütçesine göre 128…1024
        ├─ max_batch_size        : bellek bütçesi ÷ 32MB (RAM) / 48MB (VRAM)
        ├─ cache_size_mb         : bütçenin %35'i (config tavanıyla sınırlı)
        ├─ num_workers           : boş CPU çekirdeklerinin nezaketli payı
        ├─ mixed_precision       : yalnızca GPU + Volta≥
        ├─ gradient_checkpointing: baskı ≥ WARNING
        ├─ MemoryPressure        : NORMAL | WARNING | CRITICAL
        └─ should_stall          : yeni iş başlatılabilir mi?
                 │
                 └─> EventBus: allocator.memory_pressure / headroom_stall
                          │
                          └─> Trainer / Engine ağır iş kapısı
```

**Kalite kademesi (`profile`)**: `minimal → low → medium → high → ultra`.
Sistem yük altında kademe düşürür; yük geçince **kademeli olarak** geri yükselir.

### 5.4 Kullanıcı denetimi

```bash
python -m neurocore resource                 # anlık durum raporu
python -m neurocore resource --factor 0.5    # daha çekingen (boştakinin %50'si)
python -m neurocore resource --factor 1.0 --max-ram 0.95   # agresif mod
python -m neurocore resource --no-limit      # nezaketi kapat (VARSAYILAN DEĞİL)
```

Programatik:

```python
engine.config.hardware.headroom.cpu = 0.5    # CPU için boştakinin %50'si
engine.config.hardware.enforce_headroom = True
available, reason = engine.resources_available(for_heavy_work=True)
report = engine.resource_report()            # tam şeffaflık
```

---

## 6. Sinir Ağı Tasarımı (L3)

Küçük ama **genişleyebilir** bir omurga:

```
token id ──> Embedding ──┐
                         ├─> [Transformer Blok] × N ──> MoE Router ──> çıkış
modalite gömmeleri ──────┘         │
                                   ├─ Multi-Head Self Attention (+RoPE)
                                   ├─ SwiGLU FFN
                                   ├─ RMSNorm
                                   └─ Bellek Bankası (retrieval-augmented)
```

- **MoE**: `num_initial_experts` ile başlar, `max_experts`'e kadar **büyür**.
  Yönlendirme: top-k gating + yük dengeleme kaybı. Yeni uzman, **artık azaltma
  (residual) sıfırında** eklenir → mevcut bilgi bozulmaz.
- **Bellek Bankası**: anahtar-değer çiftleri; anahtar omurgadan, değer ham bağlam.
  Geri çağırma kosinüs benzerliği + LRU karışımı.
- **Kademeli niceleme**: `fp32 → fp16 → int8` (CPU'da int8 dinamik).
- **Büyüme ölçütü**: bir uzmanın yönlendirme entropisi `auto_expand_threshold`
  üzerinde kalırsa ve tamponda yeterli örnek varsa yeni uzman doğar.
- **Budama**: yönlendirme ağırlığı `prune_threshold` altına düşen uzman dondurulur
  (silinmez — `_archive` altına taşınır, geri alınabilir).

---

## 7. Çok-Modlu Hat (L4)

```
metin   → TextEncoder      → d_text
görüntü → ImageEncoder     → d_img     ──> FusionLayer ──> ortak gömme
ses     → AudioEncoder     → d_aud         (cross-attention
tablo   → TableEncoder     → d_tab          veya late-fusion)
```

- Ağır modeller (CLIP/Whisper) **isteğe bağlı**: yoksa hafif, deterministik
  özellik çıkarıcılar devreye girer (renk histogramı + gradyan, mel-enerji,
  tablo istatistikleri). Sistem her koşulda çalışır.
- **Füzyon stratejisi** konfigüre edilebilir: `attention | early | late | concat`.

---

## 8. İşletim Sistemi Katmanı (L5)

- **Envanter**: OS sürümü, çekirdek, CPU/GPU/disk/ağ, kurulu çalıştırılabilirler,
  ortam değişkenleri, yetki seviyesi.
- **Yetenek keşfi**: hangi araç var (`nvidia-smi`, `git`, `ffmpeg`, `docker`,
  `winget`, PowerShell sürümü) ve hangi işlemler izinli.
- **Eylemler**: dosya oku/yaz/listele/ara, süreç başlat/durdur, komut çalıştır,
  registry oku (Windows), ortam değişkeni yönet.
- **Sandbox**: `allowed_paths` + `denied_patterns` + `allow_shell` bayrakları.
  Yıkıcı komutlar (`format`, `rd /s`, `Remove-Item -Recurse`, `shutdown`) engellenir.
  Kuru çalıştırma (`dry_run`) desteği.
- Her eylem **denetim kaydı** (`70_journal/os_audit.md`) bırakır.

---

## 9. Dil ve Çeviri (L6)

- **Algılama**: karakter/durak kelime profili + isteğe bağlı yerel model.
- **Yerel çeviri**: `BaseTranslator` arayüzü; uygulamalar:
  - `IdentityTranslator` (tr↔tr, her zaman var)
  - `DictionaryTranslator` (kasanın `60_skills/` + `_index/terminology.json` sözlüğü)
  - `NeuralTranslator` (yerel seq2seq varsa)
  - `ApiTranslator` (OpenAI/Anthropic/DeepL anahtarı varsa — **çevrimdışıysa atlanır**)
- **Zincir**: `cache → terminology → neural → dictionary → api`, hepsi başarısızsa
  kaynak metni işaretleyerek döndürür (`[çevrilmedi]` etiketiyle).

---

## 10. Planlayıcı (L7)

```
hedef ──> Decomposer ──> TaskGraph (DAG)
                            │
                            ├─ her düğüm: {id, ad, araç, argümanlar, bağımlılıklar,
                            │              durum, deneme, sonuç, güven}
                            ├─ CheckpointManager: her N adımda diske yaz
                            ├─ ReasoningTrace: düşünce adımları, gerekçe
                            └─ Executor: topolojik sıra, paralel kanatlar
                                  └─ başarısızlıkta Replanner (geri izleme)
```

- `max_task_depth` aşılırsa görev **alt hedeflere** bölünür, özyineleme sınırlanır.
- Her adım `EventBus`'a `planner.step_started/finished` yayar → UI canlı izleyebilir.

---

## 11. Evrim Motoru (L8)

| Alt sistem | Görev |
|---|---|
| `ExperienceBuffer` | (girdi, bağlam, yanıt, geri bildirim, donanım) kaydı; öncelikli örnekleme |
| `SelfEvaluator` | Yanıt kalitesini iç tutarlılık, kaynak kapsamı, belirsizlik ile puanlar |
| `CurriculumBuilder` | Zayıf alanlardan artan zorlukta eğitim seti üretir |
| `KnowledgeSynthesizer` | Birden çok notu birleştirip yeni, daha yüksek seviyeli not üretir |
| `CuriosityEngine` | Bilgi grafındaki boşlukları/çelişkileri bulur, araştırma görevi açar |
| `GrowthController` | Uzman ekle/dondur, embedding genişlet, bütçeye göre eğitim planla |
| `IntegrityGuard` | Bilgi tutarlılığı denetimi, çelişki tespiti, geri alma |

**Döngü**: `gözlemle → değerlendir → sentezle → eğit → doğrula → birleştir (veya geri al)`.

---

## 12. Motor (L0) — Yaşam Döngüsü

```python
engine = NeuroCoreEngine()          # durum kurulumu
engine.initialize()                 # donanım algıla → profil → katmanları başlat
engine.start()                      # izleme döngüsü + arka plan bakım

answer = engine.query("...")                    # tekil sorgu
answer = engine.query("...", deep=True)         # planlayıcı + araştırma
engine.learn(text, title="...", tags=[...])     # bilgi ekle
engine.research("konu", depth=2)                # çok adımlı araştırma
engine.evolve(cycles=3)                         # evrim turu
report = engine.health_report()                 # tanı
engine.shutdown()                               # temiz kapanış
```

Fazlar: `CREATED → INITIALIZING → READY → RUNNING → DEGRADED → STOPPING → STOPPED`.

---

## 13. Hata ve Dayanıklılık Stratejisi

| Durum | Davranış |
|---|---|
| GPU yok / CUDA hatası | CPU'ya düş, `device=cpu`, kullanıcıya bildir |
| RAM kritik | batch/k cache küçült, gradient checkpointing aç, uzman dondur |
| Bağımlılık eksik | o yetenek `unavailable` işaretlenir, sistem çalışmaya devam eder |
| Kasa bozuk not | not karantinaya alınır (`90_archive/_corrupt/`), indeks atlanır |
| Plan adımı başarısız | 2 deneme + geri izleme + alternatif araç |
| Çelişkili bilgi | iki not `contradicts` kenarıyla bağlanır, güven düşürülür |
| Çökme | checkpoint'ten devam (`data/checkpoints/`) |

---

## 14. Dosya Sorumluluk Haritası

| Yol | Sorumluluk |
|---|---|
| `neurocore/core/config.py` | Tüm konfigürasyon + profil şablonları |
| `neurocore/core/event_bus.py` | Yayın/abone ol altyapısı |
| `neurocore/core/registry.py` | Katman kayıt defteri + bağımlılık çözümü |
| `neurocore/core/engine.py` | Orkestrasyon, yaşam döngüsü, `query/learn/research` |
| `neurocore/hardware/monitor.py` | CPU/RAM/GPU/disk anlık görüntüsü |
| `neurocore/hardware/allocator.py` | Kaynak bütçesi + bellek baskısı |
| `neurocore/hardware/profiler.py` | İşlem profilleme, darboğaz tespiti |
| `neurocore/knowledge/vault.py` | Kasa dizin düzeni, dosya G/Ç, bütünlük |
| `neurocore/knowledge/note.py` | Not modeli, frontmatter, bağlantılar |
| `neurocore/knowledge/index.py` | SQLite + FTS5 + vektör + graf kenarları |
| `neurocore/knowledge/search.py` | Hibrit arama + yeniden sıralama |
| `neurocore/knowledge/graph.py` | Bilgi grafı, PageRank, topluluklar |
| `neurocore/knowledge/versioning.py` | Sürüm anlık görüntüleri, geri alma |
| `neurocore/knowledge/synthesizer.py` | Not birleştirme, şablon üretimi |
| `neurocore/neural/tokenizer.py` | Byte-level BPE + eğitim |
| `neurocore/neural/attention.py` | MHA + RoPE + KV önbellek |
| `neurocore/neural/transformer.py` | Blok, SwiGLU, RMSNorm |
| `neurocore/neural/moe.py` | Uzman karışımı + yönlendirici |
| `neurocore/neural/memory.py` | Bellek bankası (retrieval-augmented) |
| `neurocore/neural/embedding.py` | Metin gömmeleri (hash + öğrenilmiş) |
| `neurocore/neural/model.py` | Omurga montajı + kaydet/yükle + niceleme |
| `neurocore/neural/growth.py` | Uzman ekle/dondur, katman genişletme |
| `neurocore/neural/trainer.py` | Eğitim döngüsü, LR planı, checkpoint |
| `neurocore/neural/engine.py` | Nöral katman cephesi |
| `neurocore/multimodal/*` | Görüntü/ses/metin/tablo kodlayıcıları + füzyon |
| `neurocore/os/*` | Envanter, dosya, süreç, kabuk, sandbox, denetim |
| `neurocore/language/*` | Algılama, çeviri, sözlük, API köprüsü |
| `neurocore/planner/*` | Görev modeli, ayrıştırıcı, yürütücü, checkpoint |
| `neurocore/evolution/*` | Deneyim, değerlendirme, sentez, merak, büyüme |
| `neurocore/cli.py` | Komut satırı arayüzü |
| `neurocore/__main__.py` | `python -m neurocore` giriş noktası |
| `tests/helpers.py` | Test altyapısı (izole config, örnek kasa) |
| `tests/test_core.py` | Konfigürasyon, olay bus'ı, motor uçtan uca |
| `tests/test_knowledge.py` | Not modeli, kasa, indeks, graf, arama, sürümleme |
| `tests/test_neural.py` | Tokenizer, gömme, bellek, MoE, dikkat, eğitim |
| `tests/test_headroom.py` | Kaynak nezaketi politikası (matematik + kapılar) |

---

## 15. Doğrulama Durumu

| Alan | Durum |
|---|---|
| Test paketi | **175 test, tamamı geçiyor** (`python -m unittest discover -s tests`) |
| Donanım profili | GTX 1080 Ti 11GB + Ryzen 9 5900X (24 iş parçacığı) + 32GB → `high`/`cuda` |
| Uçtan uca akış | init → bootstrap → learn → query → deep → research → train → grow → evolve → maintain |
| Eğitim | Gerçek kayıp düşüşü doğrulandı (7.26 → 1.03, 40 adım, 81 920 token, CUDA) |
| Nötr büyüme | Uzman ekleme çıktıyı değiştirmiyor (delta ≤ 5e-7); katman ekleme tam 0.0 |
| Kaynak nezaketi | %50 dolu → %42 bütçe · %90 dolu → %8 bütçe · duraklama eşikleri doğrulandı |
| Bilinen sınırlar | Model yeterince eğitilmeden (≥800 adım) üretim yapmaz; çıkarımsal yanıta düşer |

---

## 16. Yol Haritası

- **v0.1** (şimdi): 8 katman, çalışan sorgu/öğren/araştır/evrim döngüsü, CLI, 175 test.
- **v0.2**: Gerçek yerel LLM bağlama (llama.cpp / transformers), Whisper/CLIP opsiyonel.
- **v0.3**: Web arayüzü, canlı graf görselleştirme, çok-kullanıcılı kasa.
- **v0.4**: Dağıtık çalışma, uzman havuzu paylaşımı, artımlı öğrenme.

---

## 17. Ajan ve Uygulama Katmanları (v0.2)

Çekirdek (L0–L8) **bilgi ve öğrenme** ile ilgilenir. Kullanıcının bilgisayarında
**iş yapmak** için iki katman eklendi.

### 17.1 Ajan katmanı (`neurocore/agent/`)

| Modül | Sorumluluk |
|---|---|
| `tools.py` | Araç protokolü, **özerklik politikası**, denetim kaydı, kuru çalıştırma |
| `llm.py` | Dil modeli sağlayıcıları (bulut API / yerel sunucu / kendi model), kimlik yönetimi |
| `coding.py` | Kod tabanı: harita, sembol analizi, arama, cerrahi düzenleme, test, teşhis, git |
| `shell.py` | PowerShell/CMD: **kalıcı oturum**, komut sınıflandırma, teşhis, sınırlı çalıştırma |
| `godot.py` | Godot 4.x: proje, başsız GDScript, sahne düzenleme, export |
| `blender.py` | Blender 5.x: başsız Python, sahne kurma, render, export |
| `loop.py` | Planla → uygula → gözlemle döngüsü; kural tabanlı yedek mod |

**Araç protokolü.** Her eylem bildirilmiş bir araca indirgenir: ad, açıklama,
parametre şeması, güvenlik sınıfı ve denetim kaydı vardır. Ajan, modelin
şema tabanlı `tool_calls` üretmesiyle ya da (model desteklemiyorsa) yanıttan
JSON bloğu ayrıştırılarak araç çağırır.

**Özerklik politikası (kullanıcının ana denetimi).**

| Seviye | Otomatik izinli | Onay ister |
|---|---|---|
| `readonly` | okuma | yazma, çalıştırma, yıkıcı, sistem |
| `supervised` *(varsayılan)* | okuma | yazma, çalıştırma, yıkıcı, sistem |
| `trusted` | okuma, yazma, çalıştırma | yıkıcı, sistem |
| `yolo` | hepsi | — |

Onay **tek kullanımlıktır**: bir kez izin vermek sonraki çağrıları serbest
bırakmaz (bu, sessiz aşırı yetki verirdi). "Hatırla" seçeneği aracı oturum
boyunca serbest bırakır. Onay yanıtsız kalırsa **reddedilir**.

### 17.2 Uygulama katmanı (`neurocore/app/`)

| Modül | Sorumluluk |
|---|---|
| `state.py` | Paylaşılan durum, onay broker'ı (`threading.Event` köprüsü), olay akışı |
| `server.py` | FastAPI HTTP + WebSocket uç noktaları (yalnızca 127.0.0.1) |
| `launcher.py` | Katman kurulumu, sunucu, **Edge `--app` penceresi** |
| `web/` | Arayüz (HTML + CSS + çerçevesiz JS; derleme adımı yok) |

**Neden Edge `--app`?** Bu makinede Node.js yok, dolayısıyla Electron kabuğu
kurulamaz. Windows 11'de WebView2 ve Edge zaten kuruludur ve `--app=<url>`
bayrağı adres çubuğu olmayan gerçek bir uygulama penceresi açar — **sıfır ek
kurulum**. Pencere ayrı bir profilde çalışır; kullanıcının tarayıcı oturumuna
dokunmaz.

**Onay akışı (bloklamayan).** Ajan iş parçacığında çalışır ve onay gerektiğinde
`ApprovalBroker` üzerinden bir olay yayınlar; HTTP isteği **bekletilmez**.
Kullanıcı karar verdiğinde sonuç `tool.executed` olayıyla geri gönderilir.
Onay beklerken istek zaman aşımına uğramaz.

### 17.3 Kritik Windows tuzağı: sınırsız asılı kalma

`subprocess.run(timeout=N)` Windows'ta **N'de dönmeyebilir**. Alt süreç bir
torun doğurduğunda (`python → powershell → blender.exe` / `→ Godot.exe`) ve o
torun ayakta kalırsa, **stdout borusu açık kalır**; `communicate()` boru
kapanana kadar bekler. Ölçüm: 8 saniyelik sınır **59.6 saniyede** tetiklendi;
gerçekten takılan bir araçta çağrı hiç dönmedi.

Çözüm (`ShellIntegration._run_bounded`) üç katmanlıdır:

1. Çıktı **ayrı iş parçacıklarında** okunur (boru tıkanması olmaz).
2. Süre dolduğunda **süreç ağacı** `taskkill /T /F` ile öldürülür
   (`kill()` torunları sağ bırakır).
3. Borular **kapatılır**; okuyucular sınırlı süre beklenir.

Blender modülü ayrıca üretilen her betiğe bir **bekçi iş parçacığı** gömer
(`os._exit(1)`), böylece Blender kendi kendini bütçe içinde sonlandırır.
Godot tarafında `--quit-after` ek güvenlik ağıdır.

Doğrulama: Godot'ta sonsuz döngü içeren bir betik **12.1 saniyede** (12 s
bütçe) döndü; yetim süreç kalmadı.

### 17.4 Dil modeli yokken ne olur?

Ajan **kural tabanlı moda** düşer: yaygın istekleri (proje haritası, dosya
okuma, kod arama, test çalıştırma, araç sürümleri) doğrudan araçlara eşler.
Serbest akıl yürütme yapamaz ama sistem **tamamen kullanılamaz hâle gelmez**.
Arayüz bu durumu açıkça bildirir ve Ayarlar'dan anahtar girmeyi önerir.

### 17.5 Doğrulama durumu (v0.2)

| Alan | Durum |
|---|---|
| Test paketi | **309 test, tamamı geçiyor** (132'si ajan/uygulama katmanı) |
| Araç sayısı | 63 (kod 17, Godot 19, Blender 13, kabuk 7, görsel 7) |
| Godot | Gerçek `.exe` export doğrulandı (111 MB); 99/99 kontrol |
| Blender | Gerçek render (Cycles CPU 320×240/16 örnek: 0.46 s) + glTF/FBX export |
| Masaüstü penceresi | Edge `--app` ile açıldı, görsel doğrulama katmanıyla teyit edildi |
| Canlı akış | WebSocket olayları uçtan uca doğrulandı (plan → araç → yanıt) |
| Onay akışı | Bloklamayan: istek 15 ms'de döner, karar sonrası sonuç yayınlanır |
| Zaman sınırı | Torun süreçte bile kesin (6.1 s / 6 s bütçe) |
