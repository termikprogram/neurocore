# Ortak Havuz — Çok Makineli İş Paylaşımı

**Sürüm 0.1.0**

Farklı makinelerde (farklı ağlarda) çalışan NeuroCore'lar birbirini bulur,
iş paylaşır ve biri yanıt vermezse **işi yerel üstlenir**. İnternete port
açmak gerekmez.

---

## 1 · Neden aracı sunucu?

```
yerel makine : 192.168.1.x/24      (router arkasında)
genel IP     : <router çıkışı>     (NAT)
```

Dışarıdan `<genel IP>:port` **router'a** düşer → doğrudan P2P çalışmaz.
Bu yüzden her düğüm, erişebildiği bir **SSH sunucusuna ters tünel** açar:

```
düğüm A ──ssh -R──▶ sunucu ◀──ssh -R── düğüm B
                       │
        <sunucu>/neurocore/peers.jsonl     (kayıt defteri, TTL 90 sn)
        <sunucu>/neurocore/cluster.token   (küme anahtarı, izin 600)
```

Ölçüldü: tünel açılışı **6,3 sn**, relay iş gecikmesi **~1,7–2,7 sn**.
`GatewayPorts` kapalı olduğu için tünel `127.0.0.1` dinler → iş,
**sunucu üzerinden relay** ile taşınır.

---

## 2 · Yapılandırma (sır kodda tutulmaz)

Sıra: **ortam değişkeni** → **yerel ayar dosyası**.

### Yol A — ortam değişkenleri

```powershell
setx NC_SERVER  "ubuntu@SUNUCU-ADRESI"
setx NC_SSH_KEY "%USERPROFILE%\.ssh\nc_buMakine"
```

### Yol B — yerel ayar dosyası

`.neurocore\peer.json` (**git'e girmez**):

```json
{
  "server": "ubuntu@SUNUCU-ADRESI",
  "ssh_key": "C:\\Users\\siz\\.ssh\\nc_buMakine"
}
```

### Küme anahtarı

```
1. Düğüm kendi SSH anahtarıyla sunucuya bağlanır
2. <sunucu>/neurocore/cluster.token okunur (yoksa İLK düğüm oluşturur, 600)
3. Yerelde .neurocore/peer_token olarak saklanır
4. /peer/job doğrulaması bu dosyayla yapılır (SSH'sız, hızlı)
5. NC_PEER_TOKEN ile elle sabitlenebilir
```

⚠ Token **SSH anahtarından türetilmez**: her makine kendi anahtarını
kullanır, sır sunucuda tek yerde durur.

---

## 3 · Anahtarlar (paylaşılmaz)

```powershell
# Yeni makinede — kurulum betiği bunu kendisi yapar:
.\kur_istemci.ps1                  # anahtar yoksa üretir, komutu yazdırır

# Yetkilendirme (sunucuya erişimi OLAN bir makinede, tek satır):
type "<yeni-makine>\.pub" | ssh -i "<orada çalışan anahtar>" ubuntu@SUNUCU `
    "cat >> ~/.ssh/authorized_keys"

# Sonra doğrula:
.\kur_istemci.ps1 -SadeceAnahtar   # DURUM: çalışıyor
```

---

## 4 · Çalışma zamanı

```
açılış : ters tünel → kayıt → heartbeat (30 sn)
kapanış: kayıt silinir, tünel kapatılır
gözetim: tünel süreci ölürse yeniden açılır
```

Kapalı başlatmak için: `NC_PEER=0`.

### İş akışı

```
AgentLoop._execute(tool, args)
    │
    ├─ araç IZINLI_ARACLAR'da mı?  ── hayır ──▶ YEREL
    │
    └─ evet ─▶ PeerPool.komsu_dene()
                    ├─ yanıt var  ──▶ [uzak:<adres>]
                    └─ yanıt yok  ──▶ YEREL (iş kaybolmaz)
```

Ölçülen (gerçek sunucu):
```
web.search  → komşuda çalıştı · yerel kayıt defteri çağrısı: 0
shell.run   → yerelde kaldı (komşuya gitmedi)
komşu yok   → yerel devraldı (2,7 sn)
```

---

## 5 · İzinli araçlar (uzaktan, yalnız okuma — 16)

```
web.search · web.fetch · knowledge.ask · learn.status
code.workspace_map · code.read_file · code.list_directory
code.search_text · code.find_files
doc.kutuphane_ara · doc.api_imzasi_al · doc.surum_karsilastir
doc.godot_api_baglam
asset.katalog · asset.onizleme · asset.anim_onizleme
```

**Yasak:** `shell.run`, `shell.session_run`, `code.run_command`,
`code.edit_file`, `code.create_file` … Onay gerektiren araçlar da
uzakta çalışmaz.

---

## 6 · Uçlar

```
GET  /peer/info   → düğüm bilgisi + izinli araç listesi (token gerekmez)
POST /peer/job    → {"token","tool","args"} → sonuç
```

- Yanlış/eksik token → **403** `yetkisiz: …`
- İzin listesi dışı araç → `ok=false` `araç uzaktan çalıştırılamaz`
- Onay gerektiren araç → `ok=false` `onay gerektiriyor`

---

## 7 · Doğrulama komutları

```powershell
python tools\ag_havuzu.py --liste        # komşular + canlılık
python tools\havuz_uctan_uca.py          # 6 adım: tünel+kayıt+relay+heartbeat+temizlik
python tools\ajan_is_paylasimi.py        # 5 adım: ajan iş paylaşımı
```

Beklenen (iki makine):
```
komşu sayısı : 1 (kendisi hariç)
  <makine-adi>     relay:<port>     yaş=2.4s   canlı=True
```

---

## 8 · Sınırlar (dürüst)

| Sınır | Açıklama |
|---|---|
| Relay gecikmesi | ~2 sn/iş; yalnız **ağır** işler için kârlı |
| Yük dengeleme | yok — ilk canlı komşu seçilir |
| Kütüphane içeriği | paylaşılmaz (yalnız yetenek meta verisi) |
| Aynı kullanıcı | sunucuda tek kullanıcı → o kullanıcıya erişen kümeye girer |
| Token döndürme | sunucuda `cluster.token` silinir; düğümlerde `peer_token` silinip yeniden başlatılır |
