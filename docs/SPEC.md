# NeuroCore — Technical Specification (AI-Readable)

```yaml
document: NeuroCore Technical Specification
version: 0.1.0
edition: 2
date: 2026-10-28
audience: [AI agents, LLM tooling, maintainers]
language: tr (technical terms may be English)
status: implemented-and-measured
repository: https://github.com/termikprogram/neurocore
companion: docs/NeuroCore-Kitapcik.pdf (human booklet)
```

> **Bu dosya, başka bir AI modelinin sistemi hatasız anlayabilmesi için yazıldı.**
> Her algoritma sözde kod, her kayıt JSON şeması, her kural RFC‑2119
> anahtar sözcüğüyle (MUST / MUST NOT / SHOULD / MAY) verilir.
> Buradaki her sayı **ölçülmüştür**; ölçülmemiş bir değer görürseniz
> `unmeasured` olarak işaretlenmiştir. **Uydurma değer yoktur.**

---

## 0. Bu belge nasıl okunur

### 0.1 Gösterim

| Gösterim | Anlam |
|---|---|
| `MUST` / `MUST NOT` | İhlal edilirse sistem bozulur; değiştirmeyin |
| `SHOULD` | Önerilen; sapma gerekçelendirilmeli |
| `MAY` | İsteğe bağlı |
| `→` | "sonuç olarak" / "ardından" |
| `∨` `∧` `¬` | mantıksal veya / ve / değil |
| `←` | atama |
| `fenced block` `text` | sözde kod; değişken adları Türkçe olabilir |
| `[kaynak]` | koddaki dosya (satır) veya ölçüm aracı |

### 0.2 Ölçüm disiplini (bu projenin değişmez kuralı)

```
1. ÖLÇ      : davranışı sayıyla yaz (kayıp, süre, oran, token)
2. KÖK NEDEN: hipotezleri tek tek ölçerek ele
3. DÜZELT   : en küçük değişiklik; hatayı sessizce yutma
4. TEST     : ağsız birim testi + gerçek uçtan uca ölçüm
5. RAPOR    : önce/sonra sayıları + "kalıcı dersler"
6. YAYIN    : commit + push; her iddia ölçüme bağlı
```

Bir AI ajanı bu depoda değişiklik yaparken **MUST** şu sırayı izlemeli;
"iyileştirme" ancak önce/sonra ölçümüyle kabul edilir.

---

## 1. Sistem özeti

NeuroCore üç ayrı zekâ katmanı kullanır:

| Katman | Bileşen | Ne yapar |
|---|---|---|
| **Bulut LLM** | `agent/llm.py` (DeepSeek/Gemini/Groq) | plan, araç seçimi, cevap metni — **ajanın beyni** |
| **Yerel nöral model** | `neural/model.py` (211M) | aday **sıralama/seçim**, eğitim, bazı çeviri işleri |
| **Kural tabanlı** | `agent/rules.py` (RuleRouter) | bulut yokken basit görevler |

Temel tasarım kısıtı: **ajan ikinci sıradadır** (bkz. §6).

---

## 2. Katman haritası ve açılış sırası

```text
L0 core      config · event_bus · engine (orkestrasyon) · work (parçalı iş)
L1 kaynak    governor.py  — CPU/RAM/GPU bütçesi
L2 bilgi     knowledge/   — vault, SQLite indeks, bilgi grafı
L3 nöral     neural/      — model, trainer, tokenizer, MoE, reasoning
L4 çok-modlu vision/      — ekran, görüntü, ses
L5 işletim   os/          — dosya, süreç, sandbox
L6 dil       language/    — algılama, çeviri, terim
L7 plan      planner/     — görev grafiği, executor, checkpoint
ajan         agent/       — loop, tools, thinking, local_rank, learning, bridge
ağ           net/         — peer_pool (keşif + iş paylaşımı + tünel)
uygulama     app/         — launcher, server (FastAPI), state
```

**Katman sözleşmesi** (her katman MUST uygular):

```python
initialize(config, event_bus) -> None   # idempotent
shutdown() -> None
get_stats() -> dict
```

### 2.1 Açılış sırası (`app/launcher.py:setup`)

```text
1  çalışma alanı + <workspace>/.neurocore
2  çekirdek (config, event bus)
3  kaynak gözcüsü          → anlık ölçüm
4  kasa                    → SQLite indeks
5  araç kayıt defteri      → ToolRegistry + coding/shell/vision/Blender/Godot
6  yerel sıralama          → LocalRankController (model tembel)
7  düşünme döngüsü         → ThinkingLoop(ranker=local_rank)
8  ajan köprüsü            → AgentBridge(loop, thinking)
9  dil modeli              → LLMRouter
10 ajan döngüsü            → AgentLoop.initialize(registry, router)
   └ attach_thinker(thinking)      # planı yerel model seçer
11 sunucu                  → create_app(state) → uvicorn
   └ _havuz_baslat()               # ters tünel + kayıt + heartbeat
```

`start()` içinde `self.port` kesinleştikten **sonra** havuz başlar; tünel bu
porta yönlendirilir. `NC_PEER=0` havuzu kapatır.

---

## 3. Ajan döngüsü (`agent/loop.py`)

### 3.1 Parametreler (koddan; MUST bu değerlerle uyumlu kalın)

| Sabit | Değer | Anlam |
|---|---|---|
| `max_steps` | 24 | bir görevde en fazla adım |
| `max_seconds` | 600 | duvarsaati sınırı |
| `REPEAT_LIMIT` | 3 | aynı (araç+argüman) tekrarı → döngü kırılır |
| `MAX_EMPTY_RESPONSES` | 3 | boş yanıt sınırı |
| `MAX_RATE_LIMIT_RETRIES` | 3 | HTTP 429 yeniden deneme |
| `RATE_LIMIT_WAIT_SECONDS` | 20.0 | bekleme tabanı (katları) |
| `PLAN_CANDIDATES` | 3 | düşünmede aday plan sayısı |
| `AUTO_PLAN_CHARS` | 60 | otomatik plan eşiği (karakter) |

### 3.2 Ana algoritma

```text
GİRDİ: görev, mode ∈ {model, rules}, plan_first=False
ÇIKTI: AgentRun{ok, steps[], tools_used[], süre, sağlayıcı, cevap}

1  messages ← [system, user(görev)]
2  plan_first ← plan_first ∨ (thinker bağlı ∧ |görev| ≥ AUTO_PLAN_CHARS)
3  EĞER plan_first:
4      adaylar ← router.chat("N farklı plan üret")        # TEK çağrı
5      (plan, meta) ← thinker.think(görev, adaylar)        # §5
6      messages += [assistant(plan), user("Planı uygula")]
7      adım kaydı: kind="thinking"                          # denetim izi
8  deadline ← now + max_seconds
9  DÖNGÜ adım ≤ max_steps ∧ now < deadline ∧ ¬iptal:
10     şemalar ← select_tools(görev, registry)             # §4.1
11     yanıt ← router.chat(messages, tools=şemalar)
12     EĞER ¬yanıt.ok:
13         EĞER "413" ∈ hata ∧ |şemalar| > 4:
14             şemalar ← şemalar[:max(4, |şemalar|/2)]       # istek küçült
15             plan adımı yaz; devam
16         EĞER "429" ∈ hata ∧ deneme < 3:
17             bekle(20 × deneme); devam
18         DÖNGÜDEN ÇIK
19     EĞER yanıt.tool_calls = ∅:
20         cevap ← yanıt.text; DÖNGÜDEN ÇIK
21     HER çağrı için:
22         ad ← düzelt_ad(ad)        # "shell_run" → "shell.run"
23         EĞER (ad,args) son 3 adımda tekrarlandı → döngüyü kır
24         sonuç ← execute(ad, args)                     # §3.3
25         messages += [assistant(tool_call), tool(sonuç)]
26     boş_sayısı ← 0
27 ÖZET ← router.chat("yapılanları özetle") veya son metin
```

### 3.3 `execute()` — komşu önce, yerel devralma

```text
execute(ad, args):
1  EĞER ad ∈ IZINLI_ARACLAR ∧ peer havuzu bağlı:
2      yanıt ← peer.komsu_dene({tool:ad, args})
3      EĞER yanıt.ok: DÖNDÜR (uzak ToolResult, kaynak="relay:<port>")
4  sonuç ← registry.call(ad, args)            # politika + onay dahil
5  EĞER sonuç.needs_approval:
6      EĞER ¬kullanıcı_onayı → None (döngü durur)
7      registry.approve(ad, consume=True); sonuç ← registry.call(ad, args)
8  adım.text ← "[uzak:<kaynak>]" veya "[yerel]"
9  DÖNDÜR sonuç
```

**Değişmez:** onay izni **tek kullanımlıktır** (`consume=True`); sonraki
çağrıda yeniden sorulur. "Hatırla" seçilirse oturum boyunca geçerlidir.

### 3.4 Adım türleri (`AgentStep.kind`)

`thinking` · `plan` · `tool` · `approval` · `answer` · `error`

---

## 3.5 Bilişsel katman (`agent/cognition.py`)

Ajanın **tutarlılığını** artırır (ham zekâyı değil). Dört bileşen:

| Bileşen | İş | Sınır |
|---|---|---|
| Hata taksonomisi | 9 tür → türe **özel** kurtarma talimatı | her hata için 1 talimat |
| Hedef durumu | hedef · olgular (≤12) · denemeler (≤16) · açık sorular | özet ≤1200 karakter |
| Öz-yansıma | `devam \| yeniden_planla \| cevapla \| bildir` | en fazla **2** çağrı |
| Kanıt doğrulama | yanıt araç çıktısıyla destekleniyor mu | 1 doğrulama + 1 düzeltme |

```text
araç sonucu:
  ok    → hedef.olgu_ekle(tek satır kanıt)
  hata  → tur = siniflandir(hata); mesaj += KURTARMA(tur)      # farklı yol
her reflect_every (8) araç adımında, max_reflect (2):
  yansıma = yansit(...)  → yeniden_planla | cevapla | bildir | devam
nihai yanıt:
  kanit = araç çıktıları (≤2500 kr)
  EĞER kanit boş → doğrulama ATLANIR
  d = dogrula(...); desteksiz ∧ bütçe → yanit = duzelt(...)     # tek çağrı
```

**Değişmezler (MUST):**

```text
MUST 21  Bilişsel katman bounded: ≤2 yansıma, ≤1 düzeltme
MUST 22  Kanıt yoksa doğrulama çağrısı YAPILMAZ
MUST 23  Router hatasında katman sessizce devre dışı kalır (akış bozulmaz)
MUST 24  NC_COG=0 → eski davranış birebir korunur
```

Adım türleri: `recovery` · `critique` · `verify`.
Test: `tests/test_cognition.py` — **34 test**, ağ çağrısı yok.

## 4. Araç sistemi (`agent/tools.py`)

### 4.1 `select_tools()` — sıra önemlidir

```text
1  seçili ← []
2  HER (anahtar_sözcükler, önekler) ∈ TOOL_HINTS:      # göreve ÖZEL, önce
       EĞER herhangi anahtar_sözcük ∈ görev: seçili += eşleşen araçlar
3  seçili += CORE_TOOLS                                 # 28 araç (her görevde)
4  seçili ← aile içi sırala(seçili, yürütme araçları öne)
5  DÖNDÜR seçili[:max_tools]                            # varsayılan 22
```

**Neden sınır:** 83 aracın tamamı gönderilirse sağlayıcı `HTTP 413` döner ve
modelin doğru aracı seçme olasılığı düşer.

`CORE_TOOLS` (28) = kod (11) + kabuk (3) + doküman (4) + Godot onarım (1) +
Blender/varlık (7) + **araştırma (3: `web.search`, `web.fetch`, `knowledge.ask`)**.

### 4.2 Güvenlik matrisi

| Seviye | Örnek | readonly | supervised | trusted | yolo |
|---|---|---|---|---|---|
| READ | `code.read_file`, `web.search` | ✓ | ✓ | ✓ | ✓ |
| WRITE | `code.edit_file`, `asset.uret` | ✗ | onay | ✓ | ✓ |
| EXECUTE | `shell.run`, `code.run_command` | ✗ | onay | onay | ✓ |
| DANGEROUS | silme, süreç öldürme | ✗ | onay | onay | onay |

### 4.3 Uzaktan çalıştırılabilir araçlar — `IZINLI_ARACLAR` (16, yalnız okuma)

```text
web.search · web.fetch · knowledge.ask · learn.status
code.workspace_map · code.read_file · code.list_directory
code.search_text · code.find_files
doc.kutuphane_ara · doc.api_imzasi_al · doc.surum_karsilastir · doc.godot_api_baglam
asset.katalog · asset.onizleme · asset.anim_onizleme
```

**MUST NOT:** `shell.run`, `shell.session_run`, `code.run_command`,
`code.edit_file`, `code.create_file` ve onay gerektiren tüm araçlar uzakta
çalıştırılamaz. Liste **tek doğru kaynaktır** (`net/peer_pool.py`); sunucu
(`/peer/job`) ve ajan döngüsü aynı sabiti import eder.

### 4.4 `ToolResult` şeması

```json
{
  "ok": true, "tool": "web.search", "data": {},
  "error": "", "duration_ms": 12.4, "safety": "read",
  "blocked": false, "needs_approval": false, "reason": "",
  "truncated": false
}
```

---

## 5. Düşünme döngüsü (`agent/thinking.py`)

Amaç: bulutun ürettiği adaylar arasından **yerel** modelin seçmesi.
Ölçüm: yerel model akıcı metin **üretemiyor** ama **seçebiliyor**
(Blender kodunda %70, yaklaşımda %50; rastgele %25).

```text
think(problem, propose, context):
1  adaylar ← propose(candidates = PLAN_CANDIDATES)     # bulut, TEK çağrı
2  EĞER |adaylar| < 2:
3      DÖNDÜR ThinkingResult(chosen=adaylar[0], source="cloud", reason="tek_aday")
4  EĞER ranker yok ∨ model yüklenemedi:
5      DÖNDÜR (adaylar[0], source="cloud")             # güvenli geri düşüş
6  sıralama ← rank.rank_candidates(problem, adaylar)
       # her aday için skor = model.sequence_logprob(prompt + aday)
7  marj ← skor[0] − skor[1]
8  EĞER marj < min_margin (0.15):
9      DÖNDÜR (adaylar[0], source="cloud", güven="düşük")   # belirsiz
10 DÖNDÜR (adaylar[sıralama[0]], source="local", güven=marj)
```

Ölçülen örnek: `"Düşünme: 3 aday, kaynak=local, güven=orta, 18307ms"` —
sürenin çoğu modelin **tembel yüklenmesidir**; yüklüyken sıralama **488 ms**.

---

## 6. Kaynak gözcüsü (`core/governor.py`, `core/config.py`)

### 6.1 Formül (MUST)

```text
harcanabilir = toplam × (1 − mevcut_kullanım) × 0.90
sert tavan   : RAM %85 · VRAM %85 · CPU %85
tabanlar     : CPU 1 çekirdek · RAM 128 MB · GPU 256 MB
rezerve      : işletim sistemi için ek pay
histerezis   : cooldown_seconds boyunca bütçe BÜYÜMEZ
durma/başlama: stall_*_percent üstünde yeni iş başlatılmaz
```

| Sistem durumu | Ajanın payı |
|---|---|
| CPU %50 kullanımda | toplamın **%45**'i |
| Sistem %90 boşta | toplamın **%81**'i (tavan %85) |
| RAM %85 kullanımda | toplamın **%13,5**'i |
| GPU başka süreçte | eğitim **başlamaz** (`kaynaklar_dolu`) |

### 6.2 Plan küçültme (`plan_training`)

```text
1  bütçe ← governor.budget()      # device, max_batch, max_ram_mb, max_gpu_mb
2  batch ← clamp(max_batch, 1, 32)
3  accum ← etkin boyutu koru (küçük batch → birikim artar)
4  adım  ← steps ∨ clamp(örnek × epoch / (batch·accum), 10, 2000)
              epoch = 3.0 (kalite < 0.5) ∨ 1.5
5  lr    ← base × {3.0 (kalite < 0.3) | 1.5 (< 0.5) | 0.6 (aksi)}
6  görülme ← (adım · batch · accum · seq) / veri_token        # BU TUR
7  lr    ← base × {1.0 | 0.5 | 0.2 | 0.05}   göre görülme {<2 | ≥2 | ≥5 | ≥20}
8  amp   ← cihaz cuda
9  DÖNDÜR TrainingPlan(batch, accum, seq, lr, adım, eval_every, save_every)
```

**Ölçülmüş hata (düzeltildi):** doygunluk payında **ömür boyu** token sayacı
kullanılıyordu → `görülme = 50,5` → `lr = base × 0.05 = 5e-06` (30× düşük).
Pay artık **bu turun hacmi / bu veri kümesidir**.

---

## 7. Nöral model (`neural/model.py`)

| Özellik | Değer |
|---|---|
| Parametre | **211.255.328** (`/api/learning` + 806 MB fp32) |
| Katman | 8 · gömme 768 · kafa 12 · başlık boyutu 64 |
| Sözlük | 4.679 token (BPE) |
| MoE | 5 uzman (temel 4 + **1 büyüme**), top‑2 |
| Dizi | `max_seq_len` 2048 (eğitimde 128–256) |
| Bağlılık | `tie_embeddings=true`, RoPE taban 10.000 |
| Hassasiyet | fp32 (806 MB) · Adam ile **2.418 MB** |
| Nesil | 1 (uzman büyümesi 1, katman büyümesi 0) |
| Üretim eşiği | `can_generate = training_steps ≥ 800` (sağlanıyor) |

### 7.1 Dosyalar

```text
models/neurocore/model.pt        806 MB   ağırlıklar (fp32)
models/neurocore/config.json             mimari + büyüme kopyası
models/neurocore/growth.json             adım/token/uzman kaydı
models/neurocore/tokenizer.json          BPE sözlüğü (4.679)
data/checkpoints/neural/latest.pt      2.418 MB  ağırlık + Adam durumu
data/checkpoints/neural/step_*.pt      806 MB    her 500–750 adımda
```

### 7.2 Yetenek sınırı (ölçüldü)

```
tanıma/seçim : %50–70 doğru (rastgele %25)   → işe yarar
üretim       : kelime isabeti %23,1 · anahtar terim 0/8 → ZAYIF
sebep        : 211M parametreye karşı 107,46M token = 0,51 token/parametre
               (Chinchilla ölçeğinin %2,5'i → ciddi under-trained)
```

---

## 8. Eğitim algoritması (`neural/trainer.py`)

```text
train(model, dataset, plan, validation):
1  başlangıç ← evaluate(validation)                  # ilk kayıp (ASIL ÖLÇÜ)
2  DÖNGÜ adım 1..plan.steps:
3      batch ← dataset.örnekle(plan.batch_size)
4      kayıp ← model(batch).loss + aux_loss         # MoE denge kaybı
5      EĞER amp: kayıp ← scaler.scale(kayıp)
6      kayıp.backward()
7      EĞER amp: scaler.unscale_(opt); clip_grad(1.0); scaler.step; scaler.update
8      EĞER adım % eval_every = 0:
9          doğrulama ← evaluate(validation)
10         EĞER doğrulama < en_iyi: en_iyi ← doğrulama; ağırlıkları sakla
11         checkpoint yaz (save_every)
12     EĞER adım % 20 = 0: en_iyi_ağırlık_deepcopy()   # her adımda DEĞİL
13     EĞER bütçe_doldu ∨ süre_geçti → dur
14 final ← evaluate(validation)
15 DÖNDÜR TrainingResult(initial, final, best, val_history, tokens, stop_reason)
```

**MUST:** doğrulama kaybı raporlanmadan "öğrendi" denemez. Eğitim kaybı tek
başına ezber göstergesidir.

### 8.1 Ölçülen eğitim sonuçları

| Koşu | Doğrulama kaybı | Değişim |
|---|---|---|
| v2 havuzu, 1.566 adım | 2,6645 → 1,7704 | **−%33,6** |
| Oturum toplamı | 6,44 → 1,77 | −%72,5 |
| Talimat karışımı, 300 adım | belge +%18 / talimat −%18 | takas (ölçüldü) |
| Kasa koşusu, 33 adım | 1,9663 → 1,4087 | −%28,4 |

### 8.2 GPU kararlılığı (donanım gerçeği)

```
belirti : RuntimeError: CUDA error: unknown error (backward sırasında)
kök     : nvlddmkm Olay 153 → sürücü GPU'yu sıfırladı (TDR)
ayar    : TdrDelay TANIMSIZ = 2 sn varsayılan
çözüm   : --batch 8 --seq 128 --amp-yok  → 300/300 adım kesintisiz
          (batch 32'de ilk adımlarda çöküyor)
kalıcı  : TdrDelay artırmak yönetici + yeniden başlatma gerektirir
```

---

## 9. Öğrenme döngüsü (`agent/learning.py`)

### 9.1 Ödül algoritması (`compute_reward`)

```text
EĞER user_feedback verilmiş → kırp(user_feedback, −1, +1)
EĞER iptal ∨ onay reddi        → −0.3
EĞER başarılı:
    kendi hatasını düzelttiyse → +0.85    # en değerli (hata→çözüm çifti)
    dosya değiştirdiyse        → +0.80
    araç kullandıysa           → +0.70
    yalnız sohbet              → +0.30
EĞER başarısız ∧ adım > 1      → +0.10    # kısmi ilerleme
DEĞİLSE                        → −0.60
```

**Öğretici mi?** `ok ∧ ödül ≥ 0.6 ∧ (|tools_used| ≥ 1 ∨ |files_changed| ≥ 1)`

### 9.2 Dört kalite kuralı (MUST — bu sürümde eklendi)

| # | Kural | Eşik | Ölçülen gerekçe |
|---|---|---|---|
| 1 | **Not süzgeci** | görev < `NOT_MIN_UZUNLUK` (20) **veya** onay sözcüğü oranı ≥ 0.60 → kasa notu **YAZILMAZ** | 22 notun 11'i "Çözüm: onay veriyorum" · güven 0.925–0.990 |
| 2 | **Başlık arındırma** | `SORU:` / `Sen bir …` / `You are …` istemi ayıklanır | not başlığı sistem isteminin kopyasıydı |
| 3 | **Güven kalibrasyonu** | tek adım → ≤ **0.60**; `taban × adım × araç başarısı`, üst sınır 0.90 | tek adımlık onaya 0.925 veriliyordu |
| 4 | **Eğitim eşikleri** | deneyim 5 → **50** · adım 2 → **en az 20** · kasa: `kind ∈ {reference,qa}` ∧ güven ≥ **0.6** | 2 adım = toplam eğitimin %0,015'i; kasanın **%48,5'i** `skill` |

Onay sözcükleri (küçük harf, noktalama atılmış):
`onay, onayliyorum, onaylıyorum, onayla, evet, tamam, olur, peki, ekle, devam,
aynen, dogru, doğru, ok, okay, yes, gonder, gönder, kaydet, yap, basla, başla,
ilerle, surdur, sürdür, 1, 2, 3, 0, veriyorum, veriyoruz, ediyorum, edelim,
olsun, lutfen, lütfen, yapalim, yapalım, ekleyelim, onaylarim, onaylarım`

### 9.3 Akış

```text
görev biter
 → deneyim kaydı (experiences.jsonl)  → ödül hesaplanır
 → EĞER öğretici: KURAL 1/2/3 → kasa notu (kind=skill)
 → evrim katmanına gözlem
 → EĞER bekleyen ≥ 50 deneyim ∧ kaynak uygun → eğitim (en az 20 adım)
```

### 9.4 Deneyim kaydı şeması

```json
{
  "at": 1793215288.12, "task": "…", "answer": "…", "ok": true,
  "reward": 0.7, "mode": "tools", "provider": "DeepSeek",
  "model": "deepseek-chat", "tools_used": ["code.create_file"],
  "files_changed": ["sayi.py"], "commands_run": ["python sayi.py"],
  "steps": 3, "duration_ms": 1200.0, "reason": "tamamlandı",
  "error": "", "workspace": "E:\\PROJELER\\myAi",
  "tool_trace": [{"tool": "code.create_file", "ok": true, "args": "path=sayi.py"}],
  "used_in_training": false, "trained_at": 0.0, "observed": false
}
```

---

## 10. Kasa (`knowledge/`)

```text
knowledge_base/
  10_concepts/   kavram notları
  20_howto/      nasıl yapılır
  30_reference/  başvuru / API
  70_journal/    günlük
  _index/        SQLite arama indeksi + sürüm geçmişi (üretilir)
```

### 10.1 Not frontmatter şeması

```yaml
---
id: kn_6f8116ac
title: 'Çözüm: …'
kind: concept | reference | qa | skill | meta
category: learned | api_referans | tercih | kod_uretim | kod_ogrenme | …
tags: [ogrenilen, ajan]
confidence: 0.0 – 1.0
importance: 0.0 – 1.0
created: '2026-10-28T22:21:36+03:00'
updated: '2026-10-28T22:21:36+03:00'
version: 1
sources: ['agent://run/1793215288']
embedding_model: neurocore-learned-768
---
```

### 10.2 Eğitim filtresi (KURAL 4)

```text
EGITIM_NOTE_TURLERI = ("reference", "qa")        # varsayılan
EGITIM_NOTE_MIN_GUVEN = 0.6
ortam ile genişletme: NC_EGITIM_NOTE_TURLERI="reference,qa,concept"
```

Ölçülen etki: **6.061 not → 1.803 not** eğitime uygun (%29,7). Atlananlar
loglanır: `Kasa süzgeci: tür dışı N, düşük güven M atlandı`.

---

## 11. Ortak havuz (`net/peer_pool.py`)

### 11.1 Protokol

```text
açılış:
  token ← sunucudan (yoksa İLK düğüm oluşturur, chmod 600)   # cluster.token
  tünel ← ssh -N -R <20.000 + (node_id[:4] base16 % 20.000)>:127.0.0.1:<port>
  kayıt ← ~/neurocore/peers.jsonl  (JSON satırı)
  heartbeat thread (30 sn): tünel ölü mü? → yeniden aç → kayıt tazele

iş:
  komsu_dene({tool,args}) → relay: ssh ile sunucuda urllib POST
                            http://127.0.0.1:<relay_port>/peer/job
  yanıt yok → YEREL çalıştır (iş kaybolmaz)

kapanış: kayıt silinir, tünel kapatılır
```

### 11.2 Sabitler

| Öğe | Değer |
|---|---|
| Kayıt TTL | **90 sn** |
| Heartbeat | **30 sn** |
| Relay portu | `20000 + (node_id[:4] taban16 % 20000)` |
| SSH zaman aşımı | 25 sn |
| Token yeniden deneme aralığı | 60 sn |
| Uzak istek zaman aşımı | 6–8 sn |

### 11.3 Kayıt şeması (`peers.jsonl` satırı)

```json
{
  "node_id": "a1b2c3d4e5f60718",
  "ad": "ornek-makine",
  "ip": "203.0.113.45",
  "port": 8791,
  "relay_port": 25949,
  "caps": ["code", "blender", "arastirma", "egitim"],
  "ts": 1793216000.0,
  "surum": "Windows-10-…"
}
```

`ip` alanı **sunucu tarafında** doldurulur (`$SSH_CLIENT`), istemci
hesaplamaz — böylece heartbeat başına **1 ssh** çağrısı yeter (önce 2'ydi).

### 11.4 HTTP uçları

| Uç | Girdi | Çıktı |
|---|---|---|
| `POST /peer/job` | `{"token","tool","args"}` | `{"ok","sonuc"|"hata","ms","arac"}` |
| `GET /peer/info` | — | `{"ok","surum","arac_sayisi","izinli_araclar"}` |

Durum kodları: yanlış/eksik token → **403**; izin listesi dışı araç → `200`
ama `ok=false` (`araç uzaktan çalıştırılamaz`).

### 11.5 Güvenlik (MUST)

```
1. Kimlik: paylaşılan küme anahtarı; hmac.compare_digest
2. Anahtar YOKSA istek REDDEDİLİR (uç asla açık kalmaz)
3. Araç erişimi: IZINLI_ARACLAR (yalnız okuma)
4. IP ile sınırlama YAPILMAZ: ters tünel 127.0.0.1 dinler → IP güvenilmez
5. Özel SSH anahtarı düğümler arasında PAYLAŞILMAZ; her makine kendi anahtarı
6. sunucudaki cluster.token: 600, istemciye kopyalanmaz
```

### 11.6 Ölçülen değerler

```
tünel açılışı        : 6,3 sn
relay iş gecikmesi   : 1,7 – 2,7 sn
yerel devralma       : 2,7 sn (komşu yanıtsız)
uçtan uca havuz testi: 6/6 · ajan iş paylaşımı: 5/5
gözetim              : "Ters tünel yeniden açıldı" (gerçek çalışmada görüldü)
```

---

## 12. İşi parçalara bölme (`agent/bridge.py`, `core/work.py`)

### 12.1 Zincir

```text
görev → plan_steps() → WorkStep × N → parçala() → WorkChunk × M → yürüt → günlük
```

### 12.2 `WorkStep` şeması

```json
{
  "id": "t000", "title": "Analiz", "kind": "tool|think|generic",
  "needs": {"cpu": 5.0, "ram_mb": 200, "vram_mb": 0},
  "depends_on": ["t000"], "payload": {}, "estimate_s": 1.0,
  "status": "bekliyor|çalışıyor|bitti|atlandı|hata",
  "chunk_id": "", "result": "", "error": ""
}
```

### 12.3 Algoritmalar

```text
plan_steps(objective, plan):
1  HER ham adım için:
2      tur ← ham.kind ∨ (candidates varsa "think" yoksa "tool")
3      ihtiyaç ← ham.needs ∨ resources_for_tool(araç)
4      süre ← ham.estimate_s ∨ estimate_for_tool(araç)
5      EĞER ham.needs_previous → depends_on ← [önceki_adım]     # ZİNCİR
6      aksi hâlde bağımsız → PARALEL çalışabilir
7      DÖNDÜR WorkStep(...)

parçala(bütçe):
1  tavan ← bütçe × chunk_ratio
2  hedef_parça ← round(1 / chunk_ratio)
3  DÖNGÜ adımlar (bağımlılık sırası korunarak):
4      EĞER adım tavanı aşıyor → yeni parça
5      EĞER chunk_steps (8) üst sınırına ulaşıldı → yeni parça
6      ata; chunk_id yaz
7  DÖNDÜR WorkChunk[]

yürüt(snapshot, max_chunks, max_wait_s, poll_s=15):
1  EĞER max_wait_s > 0 → run_until_done(poll_s)
2  DEĞİLSE            → run(max_chunks)
3  sığmayan parça → ertelenir (chunks_deferred); sonra yeniden denenir
4  HER adım sonucu JSONL günlüğe yazılır
```

### 12.4 Kesintiye dayanıklılık (MUST)

**JSONL** (satır başına bir kayıt, eklemeli) kullanılır. Tek JSON dosyası
yazılırken kesilirse **tümü** kaybolur; JSONL'de yarım kalan son satır atlanır.
`resume_state()` kaldığı yeri, `pending_steps()` kalan adımları bildirir.

---

## 13. Uygulama API'si (`app/server.py`)

| Uç | İş |
|---|---|
| `GET /api/state` | özerklik, model, araç sayısı, son çalışma |
| `POST /api/chat` | görev çalıştır (asenkron) |
| `POST /api/cancel` | iptal |
| `GET /api/runs` | son çalışmalar |
| `GET /api/tools` · `POST /api/tool` | araç listesi / tek çağrı |
| `GET/POST /api/autonomy` | özerklik seviyesi |
| `POST /api/approve` · `GET /api/approvals` | onay akışı |
| `GET /api/settings` · `POST /api/settings/llm` | sağlayıcı ayarları |
| `GET /api/learning` | **model istatistikleri** (parametre/adım/token) |
| `GET /api/health` | sağlık raporu |
| `POST /peer/job` · `GET /peer/info` | ortak havuz (§11.4) |

`GET /api/learning` model bölümü (canlı, ölçülmüş):

```json
{"parameters": 211255328, "training_steps": 15904,
 "tokens_seen": 107456675, "experts": 5, "generation": 1,
 "can_generate": true, "min_generation_steps": 800}
```

---

## 14. Yapılandırma

| Yol / değişken | İş |
|---|---|
| `.neurocore/peer.json` | `{"server":"user@host","ssh_key":"…"}` — git dışı |
| `NC_SERVER`, `NC_SSH_KEY` | aynı bilgiler, ortam değişkeni (dosyayı ezer) |
| `NC_PEER_TOKEN` | küme anahtarını elle sabitler |
| `NC_PEER=0` | ortak havuzu kapatır |
| `NC_EGITIM_NOTE_TURLERI` | eğitime girecek not türleri (varsayılan `reference,qa`) |
| `NC_DEBUG_TRAIN=1` | eğitim hatasını yutmaz, `raise` eder |
| `.neurocore/llm_credentials.json` | sağlayıcı API anahtarları |
| `.neurocore/logs/neurocore.log` | günlük (eğitim özetleri dahil) |
| `data/checkpoints/neural/` | checkpoint'ler |

**MUST NOT:** depoya sır koymak. Sunucu adresi ve anahtar yolu **kodda
tutulmaz**; `.gitignore` `*.key`, `peer.json`, `peer_token`, `cluster.token`,
`models/`, `data/`, `.neurocore/` dışlar.

### 14.1 CLI

```text
python -m neurocore.app
  --workspace <dizin> --autonomy readonly|supervised|trusted|yolo
  --port 8000 --host 127.0.0.1 --no-window --browser --log-level info|debug
```

---

## 15. Değişmez kurallar (AI ajanı için özet)

```text
MUST  1  Değişiklikten önce ölç, sonra ölç; iki sayı olmadan "iyileşti" deme
MUST  2  Doğrulama kaybı ölçülmeden eğitim başarılı sayılmaz
MUST  3  Kaynak sınırı: harcanabilir = toplam × (1−mevcut) × 0.90, tavan %85
MUST  4  Onay politikası: WRITE/EXECUTE supervised'da onay ister; izin tek kullanımlık
MUST  5  Uzaktan yalnız IZINLI_ARACLAR (16, okuma); shell/yazma ASLA
MUST  6  Küme anahtarı yoksa /peer/job isteği REDDEDİLİR
MUST  7  Özel SSH anahtarı düğümler arasında paylaşılmaz
MUST  8  Depoya sır (IP, anahtar yolu, token) yazılmaz
MUST  9  Kasa notu üretmeden önce 4 kalite kuralı uygulanır
MUST 10  Eğitimde kasa filtresi: kind ∈ {reference,qa} ∧ güven ≥ 0.6
MUST 11  İş günlüğü JSONL (eklemeli) — kesinti güvenli
MUST 12  Doğrulama paydası BU TURUN hacmi; ömür boyu sayaç kullanılmaz
MUST 13  Başarısız SSH/eğitim hatası sessizce yutulmaz (sayaç + son_hata)
MUST 14  Testler ağ çağrısı yapmaz (SSH/LLM/HTTP sahtelenir)
MUST 15  Türkçe docstring + `# ─── bölüm ───` başlıkları + `from __future__ import annotations`
MUST 16  `server.py` ve Blender `run_script` yüklerinde `__future__` KULLANILMAZ
MUST 17  Boyut/uzunluk iddiası dosyadan yeniden okunarak doğrulanır
MUST 18  Türkçe dosya yazıldıktan sonra BOM kontrolü yapılır
MUST 19  8 GB VRAM sınıfında eğitim: --batch 8 --seq 128 (TDR)
MUST 20  Her tur sonunda rapor: önce/sonra sayıları + kalıcı dersler
MUST 21  Bilişsel katman bounded: ≤2 yansıma, ≤1 düzeltme
MUST 22  Kanıt yoksa doğrulama çağrısı YAPILMAZ
MUST 23  Router hatasında bilişsel katman sessizce devre dışı kalır
MUST 24  NC_COG=0 → eski davranış birebir korunur
```

---

## 15.5 Playbook · hiyerarşi · bağlam sıkıştırma

| Bileşen | Dosya | İş | Sınır |
|---|---|---|---|
| Playbook | `agent/playbooks.py` | başarılı akış → prosedür (JSONL) · benzer görevde ipucu | ödül ≥0.6, ≥2 adım, ≤8 adım, Jaccard ≥0.35 (önek duyarlı), 200 kayıt |
| Hiyerarşik plan | `agent/hierarchy.py` | hedef → ≤4 alt hedef → ≤3 adım; araç listesi dışı atılır | tek çağrı, 900 token; hata → boş plan (düz plana düşer) |
| Bağlam sıkıştırma | `loop._baglami_sikistir` | 24 mesajdan sonra eski geçmiş hedef özetine indirilir | sistem + son 10 mesaj korunur |
| Kabiliyet seti | `tools/kabiliyet_olc.py` | 10 deterministik kontrol (ağ yok) | <1 sn · skor JSON |

```text
playbook: başarılı çalışma → {goal, adimlar[], odul} → .neurocore/playbooks.jsonl
          yeni görev → benzerlik ≥ 0.35 → "İPUCU (önceki çözüm)" mesajı
hiyerarşi: |görev| ≥ 80 kr → planla() → "HİYERARŞİK PLAN" mesajı
sıkıştırma: |messages| > 24 → [system, ÖZET, son 10]
```

⚠ **Türkçe ek dersi (ölçüldü):** düz küme kesişimi "ağaç" ↔ "ağaçları"
benzerliğini **0.20** veriyordu (eşik altı). Önek eşleşmesi (≥4 harf) ile
**0.50**. Playbook bulmanın çalışması buna bağlıdır.

**Değişmezler (MUST):**

```text
MUST 25  Playbook yalnız başarılı (ödül ≥0.6) ve ≥2 adımlı akışlardan üretilir
MUST 26  Hiyerarşi planındaki araç adları verilen listeden olmalı; uydurma atılır
MUST 27  Bağlam sıkıştırması sistem istemini ve son 10 mesajı KORUR
```

## 16. Arıza biçimleri ve çareleri

| Belirti | Kök neden | Çare |
|---|---|---|
| `CUDA error: unknown error` | TDR / sürücü sıfırlaması (`nvlddmkm` 153) | `--batch 8 --amp-yok`; kalıcı: TdrDelay↑ |
| `kaynaklar_dolu` | GPU/RAM dolu | bekle; toplu işi küçült |
| Doğrulama kaybı artıyor | dağılım kayması / ezber | talimat ağırlığını düşür, veriyi çeşitlendir |
| LR çok küçük (5e-06) | ömür boyu token sayacı | payda = bu turun hacmi |
| Havuzda 1 düğüm | aynı `node_id` | `.neurocore/node_id` sil |
| `403 yetkisiz` | küme anahtarı yok/farklı | `cluster.token` + `peer_token` eşleşmeli |
| ssh penceresi kırpışıyor | `pythonw` altında alt süreç | `CREATE_NO_WINDOW` + `STARTF_USESHOWWINDOW` |
| Not yazılmıyor | kalite kuralı 1/4 | **beklenen davranış** |
| Kalite kapısı çöpü geçiriyor | yalnız uzunluk/heuristic | sözcük isabeti ≥ %55 zorunlu |
| Tablo satırı bölünüyor (DOCX) | `cantSplit` yok | `w:cantSplit` + `w:tblHeader` |

---

## 17. Ölçüm protokolü (tekrar üretilebilir)

```powershell
# ortam doğrulama (36 kontrol)
python tools\kurulum_kontrol.py --kaydet

# model payı: yerel vs bulut (4 eksen)
python tools\model_payi.py --dakika 90

# ortak havuz uçtan uca (tünel+kayıt+relay+heartbeat+temizlik)
python tools\havuz_uctan_uca.py

# ajan iş paylaşımı (komşu önce, yerel devralma)
python tools\ajan_is_paylasimi.py

# testler
python -m unittest discover -s tests -q
```

Canlı sayaçlar (uygulama açıkken):

```powershell
Invoke-RestMethod http://127.0.0.1:8791/api/state     | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8791/api/learning  | ConvertTo-Json -Depth 4
```

### 17.1 Ölçüm değerleri (2026-10-28)

| Konu | Değer |
|---|---|
| Test paketi | **1001 OK** (skipped=4) · öğrenme: 106 · havuz: 52 |
| Kurulum doğrulaması | **35/36 HAZIR** (bf16 bilgi amaçlı) |
| Taşıma paketi | 12,0 MB · 8.656 dosya |
| Model | 211.255.328 parametre · 15.904 adım · 107.456.675 token |
| Eğitim hızı | ~2,3 sn/adım (batch 32, seq 256) · 0,5–2,9k token/sn |
| Üretim kalitesi | %23,1 kelime isabeti · 0/8 anahtar terim |
| Yerel model payı | çağrı **%0,6** · token **%11,5** · araç yürütme **%100** |
| Havuz | tünel 6,3 sn · relay 1,7–2,7 sn |
| Bulut çağrısı | 153 çağrı · 20 hata (Gemini 429 → DeepSeek failover) |

---

## 18. İnşa geçmişi (8 aşama)

| Aşama | Ne yapıldı | Ölçülen sonuç / ders |
|---|---|---|
| 0 · Doğrulama | gerçekten öğreniyor mu? | doğrulama kaybı **hiç ölçülmüyordu**; LR 5e-06; eğitim 600 notla |
| 1 · Veri havuzu | tekilleştirme + başlık grubu bölme | 61.195 → 52.345 tekil · v2: 44.422/4.018 · **sızıntı 0** |
| 2 · Eğitim | doğrulamalı hat | 1.566 adım: **2,6645 → 1,7704 (−%33,6)** |
| 3 · Değerlendirme | kalite kapısı | kapı çöpü kabul ediyordu (%16,7) → **%100** |
| 4 · Düşünme + araştırma | ThinkingLoop + 3 araştırma aracı | `CORE_TOOLS` 25 → **28**; plan 3 adaydan yerel seçim |
| 5 · Talimat verisi | soru→cevap + iki doğrulama | 6.010 çift; talimat −%18 / belge +%18 (takas) |
| 6 · Ortak havuz | `/peer/job`, yaşam döngüsü, iş paylaşımı | **6/6** ve **5/5**; devralma 2,7 sn |
| 7 · Kararlılık | TDR, ssh penceresi, kasa gürültüsü | batch 8 ile 300 adım; **4 kalite kuralı** |

Ayrı hat: **Blender varlık üretimi** — 26 tur, 6 kategori, 12 animasyon tarifi,
24 geçiş çifti, doku/UV/önizleme, 15 katman/54 derslik müfredat.

---

## 19. Sözlük

| Terim | Tanım |
|---|---|
| **Ajan döngüsü** | plan → araç → gözlem döngüsüyle görevi yürüten yürütücü |
| **Araç (tool)** | JSON şemalı, politika denetimli fonksiyon |
| **Düşünme döngüsü** | bulutun adayları arasından yerel modelin seçimi |
| **Kasa (vault)** | markdown not deposu; uzun süreli bilgi |
| **WorkStep / parça** | bölünmüş işin en küçük birimi / kaynak grubu |
| **Checkpoint** | eğitim sırasında kaydedilen ağırlık anlık görüntüsü |
| **MoE** | her token'ın yalnız birkaç uzmana gitmesi (top‑2) |
| **TDR** | Windows'un yanıt vermeyen GPU sürücüsünü sıfırlaması |
| **Relay** | işin sunucu üzerinden komşu düğüme taşınması |
| **Devralma (failover)** | komşu yanıt vermezse işi yerelin yapması |
| **Doğrulama kaybı** | görülmemiş verideki kayıp; asıl başarı ölçüsü |
| **Öğretici kayıt** | `ok ∧ ödül ≥ 0.6 ∧ (araç ∨ dosya)` |

---

## Ek A. Dosya haritası

```text
neurocore/
  app/       launcher.py · server.py (FastAPI) · state.py · chats.py
  agent/     loop.py · tools.py · thinking.py · local_rank.py · learning.py
             knowledge_work.py · bridge.py · rules.py · llm.py
  core/      engine.py · governor.py · config.py · work.py · event_bus.py
  neural/    model.py · trainer.py · tokenizer.py · moe.py · reasoning.py
  knowledge/ vault · layer.py · graph.py · index
  net/       peer_pool.py     (keşif + iş paylaşımı + tünel + token)
  planner/   görev grafiği + executor
  language/  algılama + çeviri
  vision/    ekran/görüntü/ses
  models/    model tanımı
tools/       kurulum_kontrol · model_payi · havuz_uctan_uca ·
             ajan_is_paylasimi · havuz_egitimi · talimat_verisi ·
             kitapcik_uret · ag_havuzu · nat_testi
tests/       1001 test (ağ yok)
docs/        NeuroCore-Kitapcik.(pdf|docx) · KURULUM · KULLANIM · ORTAK-HAVUZ · SPEC.md
```

## Ek B. Sürüm

| Sürüm | Tarih | Not |
|---|---|---|
| 0.1.0 | 2026-10-28 | İlk yayın: ajan, araçlar, model, kasa, ortak havuz, 4 kalite kuralı |
