# Kullanım

**Sürüm 0.1.0**

---

## 1 · Masaüstü uygulaması

```powershell
.\BASLAT-istemci.bat          # ortak havuz AÇIK
python -m neurocore.app        # eşdeğeri
```

Tarayıcı `http://127.0.0.1:<port>` adresinde açılır (port doluysa sonraki boş).

### Seçenekler

```powershell
python -m neurocore.app --workspace E:/PROJELER/oyunum   # çalışma alanı
python -m neurocore.app --autonomy readonly              # salt okuma
python -m neurocore.app --no-window --port 9000          # yalnız sunucu
python -m neurocore.app --browser                        # varsayılan tarayıcı
python -m neurocore.app --log-level debug                # ayrıntılı günlük
```

---

## 2 · Özerklik seviyeleri

| Seviye | Okuma | Yazma | Kabuk | Onay |
|---|---|---|---|---|
| `readonly` | ✓ | ✗ | ✗ | — |
| `supervised` | ✓ | ✓ | ✓ | **her işlemde** |
| `trusted` | ✓ | ✓ | ✓ | düşük riskte gerekmez |
| `yolo` | ✓ | ✓ | ✓ | hiç sormaz |

⚠ `yolo` üretim için önerilmez.

---

## 3 · Ajan nasıl çalışır?

```
görev ──▶ plan (düşünme: bulut N aday üretir → yerel seçer)
      ──▶ araç seçimi (28 çekirdek araç)
      ──▶ [ortak havuz] izinli okuma aracı önce komşuda denenir
      ──▶ gözlem → sonraki adım (en fazla 24 adım / 600 sn)
      ──▶ cevap + kullanılan araçlar + adım kaydı
```

- **Uzun görevlerde plan otomatik** devreye girer (≥ 60 karakter).
- Her adım arayüzde görünür: `[uzak:relay:NNNNN]` veya `[yerel]`.
- 3 kez aynı araç+argüman tekrarlanırsa döngü kırılır (REPEAT_LIMIT).

---

## 4 · Örnek görevler

```
"DXGI_ERROR_DEVICE_HUNG hatasını araştır ve çözüm öner"
"bu projedeki testleri çalıştır, kırılanı düzelt"
"kılıç modeli oluştur, animasyon ekle, 4 açıdan önizleme al"
"Godot 4 sinyal notlarını kasadan özetle"
"bu dosyayı oku, hangi fonksiyonlar var listele"
```

Araştırma gerektiren görevlerde ajan `web.search` / `web.fetch` /
`knowledge.ask` araçlarını **kendiliğinden** kullanır.

---

## 5 · Komut satırı araçları

```powershell
python tools\kurulum_kontrol.py              # ortam doğrulama
python tools\ag_havuzu.py --liste            # havuzdaki düğümler
python tools\ag_havuzu.py --kaydet --port 8791
python tools\havuz_uctan_uca.py              # tünel+kayıt+relay ölçümü
python tools\ajan_is_paylasimi.py            # ajan iş paylaşımı ölçümü
python tools\nat_testi.py                    # NAT/ters tünel testi
```

### Eğitim

```powershell
# veri hazırla (tekilleştir + belge içi istem)
python tools\havuz_hazirla.py

# talimat (soru→cevap) verisi
python tools\talimat_verisi.py --uret

# eğit (doğrulama kaybı raporlanır)
python tools\havuz_egitimi.py --adim 300 --eval 100 --ornek 20000 `
       --seq 128 --batch 8 --kaydet-her 100 --dakika 25
```

| Parametre | Anlam |
|---|---|
| `--seq` | dizi uzunluğu (128 → ~2× hız) |
| `--batch` | toplu iş (8 GB VRAM için 8) |
| `--kaydet-her` | checkpoint aralığı |
| `--amp-yok` | fp16 AMP kapat (Pascal'da CUDA hatası olursa) |
| `--talimat` | soru→cevap verisi karıştır |
| `--dakika` | süre sınırı (kaynak gözcüsü ayrıca kısar) |

> Doğrulama kaybı **asıl ölçüdür**; eğitim kaybı tek başına yanıltır.
> Rapor hem belge hem talimat doğrulamasını ayrı gösterir.

---

## 6 · Kaynak nezaketi

```
harcanabilir = toplam × (1 − mevcut) × 0.90          # sert tavan %85
```

| Durum | Kullanılabilir |
|---|---|
| Sistem %50 dolu | toplamın %45'i |
| Sistem %90 boş | toplamın %81 → tavan %85 |
| Başka işlem GPU'yu tutuyor | eğitim başlamaz (`kaynaklar_dolu`) |

Planlayıcı kaynak yetmezse **toplu işi/çözünürlüğü küçültür**; başlatmazsa
sebebini yazar. İzleme: arayüzdeki kaynak paneli veya `neurocore.core.engine`.

---

## 7 · Kasa (bilgi deposu)

```
knowledge_base/            Obsidian uyumlu markdown
.neurocore/index/          SQLite arama indeksi
```

- `knowledge.ask` → kasadan yanıt
- `learn.topic` → konu araştır, kasaya yaz
- `learn.status` → öğrenme durumu

---

## 8 · Ortak havuz

Ayrı makinedeki NeuroCore'larla iş paylaşımı: **[ORTAK-HAVUZ.md](ORTAK-HAVUZ.md)**
