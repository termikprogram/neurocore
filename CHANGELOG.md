# Sürüm Geçmişi

Tüm önemli değişiklikler bu dosyada tutulur.
Biçim: [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) ·
Sürümleme: [SemVer](https://semver.org/lang/tr/)

---

## [0.2.4] — 2026-10-31

### Düzeltildi

**ZIP indirmesinde uygulama başlamıyordu: paket dosyaları repoda yoktu.**

- `.gitignore` içindeki `_*.py` deseni **`__init__.py` ve `__main__.py`** dosyalarını
  da yutuyordu → 18 paket dosyası hiç commit edilmemişti.
- Sonuç: temiz klonda/ZIP'te `python -m neurocore.app` →
  `No module named neurocore.app.__main__; 'neurocore.app' is a package and
  cannot be directly executed` ve `Run.bat` sessizce çıkıyordu.
- Düzeltme: desen `/_*.py` (yalnız kök dizin) · 18 dosya eklendi:
  `neurocore/__init__.py`, `neurocore/__main__.py`, 15 alt paket `__init__.py`,
  `tests/__init__.py`.

### Ölçüm

```
Önce : git ls-files → __init__.py = 0, __main__.py = 0
Sonra: __init__.py = 16, __main__.py = 2
Uygulama (temiz kopya, .venv): python -m neurocore.app → HTTP 200

### Düzeltildi (devamı)

**Türkçe Windows konsolunda Unicode çökmesi.**
- `neurocore/app/launcher.py` kutu çizim karakterlerini (`─`) doğrudan yazdırıyordu;
  konsol `cp1254` olduğunda `UnicodeEncodeError: 'charmap' codec can't encode`
  ile uygulama açılmadan ölüyordu.
- `_utf8_konsol()` eklendi: `sys.stdout/stderr` → UTF-8 + `errors="replace"`;
  `main()` başında çağrılır; konsol yoksa (`pythonw`) sessizce geçer.
- `Run.bat`, `Run-Yolo.bat`, `Run-Here.bat` artık `PYTHONUTF8=1` ve
  `PYTHONIOENCODING=utf-8` ayarlıyor (kemer + askı).
- Ölçüm: temiz kopyada `python -m neurocore.app --port 8792` → **HTTP 200 (6 sn)**.
```

---

## [0.2.3] — 2026-10-31

### Eklendi

- **`installation.bat`** — İngilizce adlı kurulum başlatıcısı. `Kurulum.bat` ile
  aynı işi yapar ve **tüm argümanları ona iletir**:
  `--check`, `--no-venv`, `--no-torch`, `--help`.
  `Kurulum.bat` yanında yoksa anlaşılır hata verir.
- İki ad da dokümanlarda (`README.md`, `docs/en/INSTALL.md`) belirtildi.

### Ölçüm

```
installation.bat --check → "Python 3.10.11 bulundu ... 64-bit: True" · çıkış 0
(Kurulum.bat'a iletilen argümanlar aynen çalışıyor)
```

---

## [0.2.2] — 2026-10-31

### Düzeltildi

**Kurulum penceresi anında kapanıyordu (ZIP indirmesinde).**

- **Kök neden: satır sonları.** `Kurulum.bat`, `Kisayol-Olustur.bat` ve
  `Sunuculari-Kapat.bat` **LF** ile kaydedilmişti; CMD, LF-only batch
  dosyalarında blok/`goto` ayrıştırmasını bozar → betik sessizce ölür.
  - `.gitattributes` eklendi: `*.bat -text` (bayt olarak korunur), `*.ps1 eol=crlf`
  - Tüm `.bat` dosyaları **CRLF**'e çevrildi (9 dosya)
- **Blok içinde kaçışsız parantez.** `echo ... (.venv)...` bir `if (...)` bloğu
  içinde CMD tarafından yeni blok sanılıyordu → ``> was unexpected`` hatası.
  `^(` / `^)` ile kaçışlandı (`Kurulum.bat`, `Run.bat`).
- **`--check` etiketi eksikti** → `goto :check_only` "label not found" veriyordu.
  Etiket eklendi; `--check` artık Python'u tespit edip kurulum yapmadan çıkar.

### Eklendi

- **`Kurulum.bat --check`** — yalnız Python tespiti (kurulum yapmaz);
  sürüm + 64-bit bilgisini yazar. Başka makinede ilk kontrol için.
- **`tools\py_bul.bat` yoksa yedek tespit** — eksik/bozuk indirmede bile
  `where python` + `py -3` ile devam eder.
- **Python yoksa net uyarı** — `winget install --id Python.Python.3.12 -e`,
  python.org bağlantısı ve `NC_PYTHON` örneği; pencere `pause` ile açık kalır.
- **`Kurulum.bat --no-torch`** — `torch/torchvision/torchaudio` (~2,5 GB)
  atlanır; hızlı başlangıç. Sonradan kurmak için komutu yazar.
  Ölçüm: filtrelenmiş kurulumda torch **yüklenmedi**, `fastapi` yüklendi,
  "Setup complete" · çıkış 0.

### Ölçüm

```
Önce : Kurulum.bat LF, 149 LF / 0 CRLF → pencere anında kapanıyor (çıkış 255)
Sonra: 9 .bat dosyası CRLF, --check → çıkış 0
Uçtan uca (temiz kopya, hafif gereksinimler):
  .venv oluştu · paketler kuruldu · "Setup complete" · çıkış 0
Python yok senaryosu (stub bulucu):
  "[ERROR] Python 3.10+ was not found" + winget önerisi + pause · çıkış 1
```

---

## [0.2.1] — 2026-10-30

### Düzeltildi

**Kurulum diğer makinelerde çalışmıyordu** — betiklerde bu makineye özel
`E:\AppsTools\PYTHON\Python3.10\python.exe` yolu vardı ve başka bir
bilgisayarda Python farklı yerde olduğu için kurulum/başlatma başarısız oluyordu.

- `tools/py_bul.bat` — **taşınabilir Python bulucu** (tek kaynak):
  `NC_PYTHON` → `.venv` → `py -3.13…3.10 / py -3` → `PATH` →
  `%LOCALAPPDATA%\Programs\Python\PythonXY` → `%ProgramFiles%`/`C:\PythonXY`
  → **registry** (`PythonCore\InstallPath`).
  Her aday **>= 3.10** doğrulanır; uygun olmayan atlanır.
- `Kurulum.bat` — bulucuyu kullanır; `.venv` oluşturur (varsayılan),
  `--no-venv` ile sistem Python'u, `NC_PYTHON` ile özel yol; Python yoksa
  `winget install --id Python.Python.3.12 -e` komutunu yazar.
- `Run.bat`, `Run-Yolo.bat`, `Run-Debug.bat`, `Run-Here.bat` — sabit yol
  kaldırıldı, hepsi `tools\py_bul.bat` kullanıyor.
- `kur_istemci.ps1` — PowerShell tarafında aynı sıra (py launcher + registry
  dahil) ve sürüm doğrulaması.

**Ölçüm:** bulucu `.venv\Scripts\python.exe` (3.10.11) seçiyor; `NC_PYTHON`
verildiğinde ilk onu kullanıyor; `Kurulum.bat --help` tespit sırasını yazıyor.

---

## [0.2.0] — 2026-10-30

Bu sürüm **büyük projelerde çalışma** üzerine kurulu; her başlıkta ölçüm var.

### Eklendi

**Büyük dosya okuma — `neurocore/agent/bigfile.py`**
- Seyrek satır indeksi (`<dosya>.ncidx.json`, her 2000. satırın bayt ofseti)
- Pencere okuma (`pencere`) · yapısal harita (`harita`) · akışlı arama (`ara`)
- Ölçüm (10.000.000 satır / 427 MB): indeks **8,2 sn · 0,6 MB bellek**,
  rastgele pencere **11 ms**, arama **6,7 sn**
- `search_text` 4 MB üstü dosyaları artık **atlamıyor** (akışlı tarama)

**Proje kod indeksi — `neurocore/agent/codeindex.py`**
- SQLite (`sembol`, `iliski`, `cagri`, `tip`, `kalitim`) + FTS5
- **Artımlı**: yalnız mtime/size değişen dosyalar taranır
- **AST tabanlı**: Python `ast` (docstring/yanlış eşleşme yok, nitelikli adlar,
  `ast.Call` ile çağrılar) · GDScript yapısal tarayıcı (`class_name/func/signal/const`,
  `preload`, fonksiyon içi çağrılar)
- `.gdignore` dizinleri taranmaz
- Ölçüm (2.000 dosya / 5,99 M satır): ilk kurulum **39,8 sn** · DB **13,4 MB** ·
  artımlı **0,15 sn** · sorgu **3-8 ms**
- Ölçüm (TitoWarWeb, 3,7 GB): **7,3 sn** · 497 dosya · **144.035 satır** · 6.865 sembol

**6 yeni kod aracı** (toplam 94): `code.index`, `code.map`, `code.find_symbol`,
`code.find_references`, `code.callers`, `code.related`, `code.hierarchy`,
`code.mro`, `code.types`, `code.impact_map`, `code.outline`

**Etkileşimli etki haritası (`code.impact_map`)**
- Çağrı grafiğinden tek dosya HTML+SVG: tıkla-seç, komşu vurgulama, dosya:satır paneli
- Ölçüm: `sembol_ara` → 29 düğüm · 29 kenar · 9,6 KB · dış kaynak yok

**Refactor parite kapısı — `neurocore/agent/refactor.py`**
- `git diff --numstat` ölçümü: silinen > 150 ve eklenen < silinen×0.60 → **kod kaybı**
- `parite_kontrol()`: GDScript · Python · JS/TS · Rust · Go fonksiyon adı farkı
- Döngü kapıları: `_refactor_denetle` (yanıt öncesi) +
  `_refactor_bitis_denetle` (her bitişte, süre sınırı dahil) +
  **otomatik `git checkout`** (oran < 0,5 ve `NC_OTOMATIK_GERI_AL=1`)
- Gerçek olay: **706 silinen / 35 eklenen** yakalandı, dosya geri alındı

### Düzeltildi

- **Araç çıktısı ajana ulaşmıyordu** — `ToolResult.to_dict()` 12.000 karakteri
  aşınca `data`yı JSON **string**'e çeviriyordu; ajan yalnız meta veri görüyordu.
  Artık yapı korunur (`_kirp_veri`).
- **`render()` yerine `_summarize_result`** — 20 KB'lik dosya ham JSON olarak
  6 KB'a kesiliyordu; artık yapılandırılmış biçim.
- **Bağlam sıkıştırması okunan dosyayı siliyordu** — `_okuma_kasasi` +
  `_okuma_ozeti` (en fazla 3 dosya sıkıştırma özetine taşınır); eşik 24→40.
- **`tool_call` kimlikleri** — sağlayıcı `id` vermezse kimlik uydurulur ama
  asistan mesajının `raw_message`'ı eski kimlikleri taşırdı; `tool` sonuçları
  yetim sayılıp silinirdi. Artık üç yerde senkron.
- **`search_text(path=<dosya>)`** — dosya yolu dizin sanılıyor, 0 sonuç
  dönüyordu; ajan "karşılığı yok" sanıp doğru düzenlemeyi reddediyordu. Düzeltildi.
- **İçerik sınırı 14.000 → 36.000 karakter** (`NC_ICERIK_SINIRI` ile ayarlanır);
  790 satırlık gövde artık tam geliyor.
- **`read_file` varsayılanı 400 → 800 satır**
- **Kapı hata yalıtımı** — bir kapı çökerse çalışma ölmez (`'NoneType' has no 'get'`
  ile 613 sn'lik koşu sıfır cevapla bitmişti)

### Değişti

- **Araç zorunlu kapısı** — görevde `code.*` adı geçiyor ve hiç araç
  çağrılmamışsa yanıt reddedilir. Ölçüm: istekte tool **0 → 3**.
- **Madde kapısı** — 2+ maddeli görevde madde başına kanıt istenir; kanıt
  **mesajlardan** sayılır (plan/puzzle adımları "araç" sayılıp sayı bozuluyordu).
- **Madde → araç eşlemesi** — "nerede tanımlı" → `find_symbol`,
  "kim çağırıyor" → `callers`, "ilişkili" → `related`, "en büyük" → `map`.
- **Kanıt hatırlatma** — araç çıktısında sembol varken "yok" denirse o satırlar
  aynen alıntılanıp düzeltme istenir.
- **`.gdignore` artık indekste de geçerli** — TitoWarWeb'de 140 backup dosyası
  638→487 dosyaya düştü (190.208→142.665 satır).

### Testler

- **98 test** (`test_bigfile` 16 · `test_codeindex` 18 · `test_codeast` 20 ·
  `test_refactor` 18 · `test_toolresult` 10 · `test_envcheck` 16)
- Yeni: bellek sınırı (<8 MB), `.gdignore`, dosya yolu araması, çok dilli
  parite, gerçek `git checkout` senaryosu

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
