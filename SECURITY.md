# Güvenlik

Bu dosya, NeuroCore deposunun güvenlik modelini ve **depoya girmemesi
gereken** bilgileri açıklar.

---

## 1 · Depoda sır yoktur

Aşağıdakiler **kod içinde tutulmaz**, ortam değişkeni veya yerel ayar
dosyasından okunur:

| Bilgi | Nereden okunur | Depoya girer mi |
|---|---|---|
| Küme sunucusu adresi | `NC_SERVER` · `.neurocore/peer.json` | ✗ |
| SSH anahtar yolu | `NC_SSH_KEY` · `.neurocore/peer.json` | ✗ |
| Küme anahtarı (token) | sunucudan → `.neurocore/peer_token` | ✗ |
| Sağlayıcı API anahtarları | `.neurocore/llm_credentials.json` | ✗ |
| Model ağırlıkları | `models/`, `data/checkpoints/` | ✗ |

`.gitignore` bunları açıkça dışlar: `*.key`, `*.pem`, `peer.json`,
`peer_token`, `cluster.token`, `.env`, `models/`, `data/`, `.neurocore/`.

### Sızıntı kontrolü

```powershell
git grep -nE "(158\.|ssh-key|-----BEGIN|api[_-]?key|sk-)" -- . ":!*.md"
```
Boş çıktı = temiz.

---

## 2 · Anahtar yönetimi

```
✓ her makine KENDİ anahtar çiftini üretir (ed25519)
✓ sunucuya yalnız AÇIK anahtar eklenir
✗ özel anahtar makineler arasında paylaşılmaz
```

Windows'ta anahtar izni kısıtlanmalı (aksi hâlde OpenSSH reddeder):

```powershell
icacls "<anahtar>" /inheritance:r /grant:r "%USERNAME%:R"
```

`kur_istemci.ps1` bunu otomatik yapar.

---

## 3 · Uzak düğüm yetkileri

`POST /peer/job` iki katmanla korunur:

1. **Küme anahtarı** — `hmac.compare_digest` ile karşılaştırılır.
   Anahtar yoksa istek **reddedilir** (uç açık kalmaz).
2. **Araç izin listesi** — yalnız okuma araçları (16):
   `web.search`, `web.fetch`, `knowledge.ask`, `code.read_file`, …

**Uzaktan asla çalışmayanlar:** `shell.run`, `shell.session_run`,
`code.run_command`, `code.edit_file`, `code.create_file`, onay
gerektiren tüm araçlar.

⚠ Kaynak IP ile sınırlama **yapılmaz**: ters tünel `127.0.0.1` dinler ve
relay isteği sunucudan gelir → IP güvenilir değildir. Bu yüzden anahtar
zorunludur.

---

## 4 · Küme anahtarı döndürme

```bash
# sunucuda
rm ~/neurocore/cluster.token
```
Sonra her düğümde `.neurocore/peer_token` silinip yeniden başlatılır →
ilk açılan düğüm yeni anahtarı üretir, diğerleri okur.

`NC_PEER_TOKEN` ile elle sabitlenmiş düğümler döndürmeden etkilenmez
(bilinçli: acil durum override'ı).

---

## 5 · Bilinen sınırlar

| Konu | Durum |
|---|---|
| Aynı sunucu kullanıcısı | sunucuya erişen herkes kümeye girer (tek kullanıcı varsayımı) |
| Token yerelde düz metin | `.neurocore/peer_token` (izin kısıtlı) |
| Relay şifrelemesi | SSH tüneli içinde; uçlar arası TLS yok |
| `/peer/info` | kimlik doğrulaması istemez (yalnız yetenek listesi döner) |
| Masaüstü uygulaması | `127.0.0.1` dinler (varsayılan) — ağa açmayın |

---

## 6 · Bildirim

Güvenlik açığı bildirimi için depo sahibine **özel** mesaj gönderin
(genel issue açmayın). Lütfen şunları ekleyin: etkilenen sürüm, yeniden
üretme adımları, beklenen/gözlenen davranış.
