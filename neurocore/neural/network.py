"""
Ağ (Network) Müfredatı — Temelden Uzmanlığa
============================================
Yerel modelin **ağ yapısını** en temel bit seviyesinden dağıtık sunucu
mimarilerine kadar öğrenmesi için katmanlı içerik.

Neden ayrı modül?
    `curriculum.py` genel programlama/bilgisayar bilgisi verir. Ağ ise
    kendi katmanlı yapısı olan, **protokol yığını** üzerine kurulu ayrı bir
    alandır ve çok daha derin anlatılmalıdır:

        Katman 1  Ağ Temelleri      — sinyal, topoloji, OSI/TCP-IP, adresleme
        Katman 2  Taşıma ve Protokol — TCP, UDP, QUIC, soket, port, akış
        Katman 3  Uygulama Protokolleri — HTTP, WebSocket, DNS, TLS
        Katman 4  Sunucu Mimarileri — main, proxy, reverse proxy, LB, dağıtık
        Katman 5  Oyun Sunucuları   — authoritative, dedicated, tick, lag comp.
        Katman 6  Godot Ağ          — ENet, MultiplayerAPI, RPC, senkronizasyon

Her ders `Lesson` olarak tutulur (bkz. `curriculum.Lesson`) ve eğitim çifti
üretir: `prompt` (soru) → `target` (açıklama + kod).

⚠ İçerik **gerçek protokollere** dayanır: port numaraları, başlık alanları,
zaman aşımları ve API adları doğrulanmış değerlerdir. Uydurma yok.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from neurocore.neural.curriculum import Lesson

#: Ağ müfredatı katmanları (sıra öğretim sırasıdır)
NETWORK_LAYERS: Sequence[str] = (
    "net_basics", "net_transport", "net_application",
    "net_servers", "net_game", "net_godot",
)

#: Katman etiketleri
NETWORK_LABELS: Dict[str, str] = {
    "net_basics": "Ağ Temelleri",
    "net_transport": "Taşıma Katmanı ve Protokoller",
    "net_application": "Uygulama Protokolleri",
    "net_servers": "Sunucu Mimarileri",
    "net_game": "Oyun Sunucuları",
    "net_godot": "Godot ve Ağ Programlama",
}


# ─── Katman 1: Ağ Temelleri ─────────────────────────────────────────────────

def _basics_lessons() -> List[Lesson]:
    """Sinyalden adreslemeye: ağın temel yapı taşları."""
    return [
        Lesson("net_basics", "Ağ nedir, neden katmanlara ayrılır?",
               "Ağ, iki veya daha fazla cihazın veri alışverişi yapmasıdır. "
               "Karmaşıklığı yönetmek için **katmanlara** ayrılır: her katman "
               "yalnızca kendi işini yapar ve altındaki katmana güvenir. "
               "Bu sayede kabloyu (fiziksel) değiştirmeden protokolü (mantıksal) "
               "değiştirebilirsiniz. Katmanlara ayırma, bir katmandaki hatanın "
               "diğerlerini bozmamasını ve her katmanın bağımsız test "
               "edilebilmesini sağlar.",
               ["ağ", "katman", "temel"]),
        Lesson("net_basics", "OSI modeli: 7 katman ne işe yarar?",
               "OSI modeli ağı 7 katmana böler: **1 Fiziksel** (kablo, sinyal, "
               "voltaj), **2 Veri Bağlantı** (MAC adresi, Ethernet çerçevesi, "
               "anahtar/switch), **3 Ağ** (IP adresi, yönlendirme/router), "
               "**4 Taşıma** (TCP/UDP, port, uçtan uca), **5 Oturum** (oturum "
               "yönetimi), **6 Sunum** (şifreleme, sıkıştırma, kodlama), "
               "**7 Uygulama** (HTTP, DNS, oyun protokolleri). Pratikte "
               "TCP/IP modeli 4 katman kullanır: Ağ Erişimi, İnternet, "
               "Taşıma, Uygulama.",
               ["osi", "katman", "tcp-ip"]),
        Lesson("net_basics", "IP adresi, alt ağ maskesi ve CIDR",
               "IPv4 adresi 32 bittir, noktalı dört sayı olarak yazılır "
               "(192.168.1.10). **Alt ağ maskesi** hangi bitlerin ağ, "
               "hangilerinin cihaz olduğunu söyler. **CIDR** bunu kısa yazar: "
               "`/24` = 255.255.255.0 → 256 adres, 254 kullanılabilir. "
               "`/16` = 65.536 adres. IPv6 128 bittir ve `2001:db8::1` gibi "
               "yazılır; adres uzayı pratikte tükenmez. Özel adres blokları: "
               "`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (RFC 1918), "
               "`127.0.0.0/8` (loopback).",
               ["ip", "cidr", "alt ağ"], code=(
                   "import ipaddress\n"
                   "\n"
                   "ag = ipaddress.ip_network('192.168.1.0/24')\n"
                   "print(f'ag adresi     : {ag.network_address}')\n"
                   "print(f'yayin adresi  : {ag.broadcast_address}')\n"
                   "print(f'kullanilabilir: {ag.num_addresses - 2}')\n"
                   "print(f'maske         : {ag.netmask}')\n"
                   "\n"
                   "adres = ipaddress.ip_address('192.168.1.10')\n"
                   "print(f'ozel mi       : {adres.is_private}')")),
        Lesson("net_basics", "MAC adresi, ARP ve yerel ağ",
               "**MAC adresi** ağ kartının donanım kimliğidir (48 bit, "
               "`AA:BB:CC:DD:EE:FF`). Yerel ağda cihazlar IP'yi MAC'e çevirmek "
               "için **ARP** kullanır: 'Bu IP kimin?' diye yayın yapar, sahibi "
               "MAC'ini bildirir. **Switch** MAC tablosuna göre çerçeveyi "
               "yalnızca hedef porta gönderir; **hub** ise herkese gönderirdi "
               "(artık kullanılmaz). ARP tablosu `arp -a` ile görülür.",
               ["mac", "arp", "switch"], code=(
                   "# Windows\n"
                   "arp -a                    # ARP tablosu\n"
                   "getmac /v                 # kendi MAC adresleri\n"
                   "\n"
                   "# Aynı bilgiyi Python'da okumak\n"
                   "import uuid\n"
                   "print(f'bu makinenin MACi: {uuid.getnode():012X}')")),
        Lesson("net_basics", "Yönlendirme (routing) ve ağ geçidi",
               "Yerel ağda olmayan hedefe giden trafik **ağ geçidine** "
               "(gateway, genelde router) verilir. Router **yönlendirme "
               "tablosuna** bakar: hedefe en uzun eşleşen kuralı seçer. "
               "**Varsayılan rota** (`0.0.0.0/0`) 'başka kural yoksa buraya' "
               "demektir. Yönlendirme protokolleri: **OSPF** (iç ağ, hızlı "
               "yakınsama), **BGP** (internet omurgası, politikalar), "
               "**RIP** (küçük ağ, hop sayısı).",
               ["yönlendirme", "router", "gateway"], code=(
                   "# Windows\n"
                   "route print               # yonlendirme tablosu\n"
                   "tracert 8.8.8.8           # yolu izle (her atlama)\n"
                   "\n"
                   "# Linux\n"
                   "# ip route show\n"
                   "# traceroute 8.8.8.8")),
        Lesson("net_basics", "NAT: bir IP'yi binlerce cihazla paylaşmak",
               "IPv4 adresleri tükendiği için **NAT** (Network Address "
               "Translation) kullanılır: ev ağınızdaki tüm cihazlar tek genel "
               "IP'yi paylaşır. Router, dışarı giden bağlantılar için "
               "`(kaynak IP:port) → (genel IP:geçici port)` eşlemesi tutar. "
               "**Dışarıdan içeriye** bağlantı kurulamaz — bu yüzden sunucu "
               "işletmek için **port yönlendirme** (port forwarding) veya "
               "**hole punching** gerekir. **CGNAT** (taşıyıcı NAT) ise "
               "kullanıcıya özel genel IP bile vermez; bu durumda port "
               "yönlendirme **mümkün değildir**.",
               ["nat", "port yönlendirme", "cgnat"], code=(
                   "# Dis IP'yi ogren (NAT sonrasi gorunen adres)\n"
                   "import urllib.request\n"
                   "with urllib.request.urlopen('https://api.ipify.org', timeout=10) as r:\n"
                   "    print('dis IP:', r.read().decode())\n"
                   "\n"
                   "# Yerel IP'ler\n"
                   "import socket\n"
                   "print('yerel:', socket.gethostbyname(socket.gethostname()))")),
        Lesson("net_basics", "Paket, çerçeve ve kapsülleme",
               "Veri aşağı katmanlara inerken her katman kendi başlığını "
               "ekler (**kapsülleme**): Uygulama verisi → TCP segmenti "
               "(TCP başlığı) → IP paketi (IP başlığı) → Ethernet çerçevesi "
               "(MAC başlığı) → sinyal. Alıcıda ters işlem yapılır. **MTU** "
               "(Maximum Transmission Unit) Ethernet'te 1500 bayttır; daha "
               "büyük paket **parçalanır** (fragmentation) ve bu performansı "
               "düşürür. TCP, MTU'yu aşmamak için **MSS** (Maximum Segment "
               "Size) kullanır: MSS = MTU − IP başlığı − TCP başlığı = "
               "1500 − 20 − 20 = 1460 bayt.",
               ["paket", "mtu", "kapsülleme"]),
        Lesson("net_basics", "Gecikme (latency), bant genişliği ve jitter",
               "**Gecikme** bir paketin gidiş-dönüş süresidir (RTT, ms). "
               "**Bant genişliği** saniyede taşınan veri (Mbps). **Jitter** "
               "gecikmenin dalgalanmasıdır — oyunda jitter, ortalama "
               "gecikmeden daha çok rahatsız eder. **Kayıp (packet loss)** "
               "yüzdesi TCP'de yeniden iletim, UDP'de veri kaybı demektir. "
               "Tipik değerler: aynı şehir 1-10 ms, kıtalar arası 100-200 ms, "
               "uydu 500+ ms. Işık hızı sınırı: İstanbul-New York fiber "
               "~60 ms teorik taban.",
               ["gecikme", "jitter", "bant genişliği"]),
        Lesson("net_basics", "DNS: isimden IP'ye",
               "**DNS** alan adını IP'ye çevirir. Sorgu zinciri: **özyinelemeli "
               "çözümleyici** (ISP veya 8.8.8.8) → **kök sunucular** → "
               "**TLD sunucuları** (`.com`) → **yetkili sunucular** "
               "(alanın kendi). Kayıt türleri: **A** (IPv4), **AAAA** (IPv6), "
               "**CNAME** (takma ad), **MX** (e-posta), **TXT** (doğrulama), "
               "**SRV** (servis keşfi — oyunlarda sunucu bulmak için). "
               "**TTL** kaydın önbellekte kalma süresidir; düşük TTL hızlı "
               "değişiklik, yüksek TTL az sorgu demektir.",
               ["dns", "srv", "ttl"], code=(
                   "import socket\n"
                   "\n"
                   "print(socket.gethostbyname('example.com'))\n"
                   "print(socket.gethostbyname_ex('example.com'))\n"
                   "\n"
                   "# SRV kaydi (oyun sunucusu kesfi)\n"
                   "# Windows: nslookup -type=SRV _minecraft._tcp.example.com")),
        Lesson("net_basics", "Firewall, port ve güvenlik duvarı kuralları",
               "**Güvenlik duvarı** hangi trafiğin geçeceğine karar verir: "
               "kaynak/hedef IP, port, protokol, yön. **Durum bilgili** "
               "(stateful) duvar, içeriden başlatılan bağlantının yanıtını "
               "otomatik izin verir; **durumsuz** her paketi ayrı değerlendirir. "
               "Sunucu işletirken yalnızca **gereken portlar** açılır (en az "
               "yetki). Bir portun dinlenip dinlenmediği `netstat -ano` veya "
               "`ss -tulpn` ile görülür.",
               ["firewall", "port", "güvenlik"], code=(
                   "# Windows: dinlenen portlar ve sahibi\n"
                   "netstat -ano | findstr LISTENING\n"
                   "\n"
                   "# Belirli port\n"
                   "netstat -ano | findstr :8080\n"
                   "\n"
                   "# Python ile port kontrolu\n"
                   "import socket\n"
                   "s = socket.socket()\n"
                   "print('8080 musait mi:', s.connect_ex(('127.0.0.1', 8080)) != 0)\n"
                   "s.close()")),
    ]


# ─── Katman 2: Taşıma Katmanı ───────────────────────────────────────────────

def _transport_lessons() -> List[Lesson]:
    """TCP, UDP, QUIC ve soket programlama."""
    return [
        Lesson("net_transport", "TCP nedir, nasıl güvenilirlik sağlar?",
               "**TCP** bağlantı yönelimli, güvenilir, sıralı bir protokoldür. "
               "Güvenilirliği üç mekanizmayla sağlar: **sıra numarası** (paket "
               "sırası), **onay (ACK)** (alındı bildirimi), **yeniden iletim** "
               "(zaman aşımında tekrar gönderim). Bağlantı **üçlü el sıkışma** "
               "ile kurulur: `SYN → SYN-ACK → ACK`. Kapanış dört adımdır "
               "(`FIN → ACK → FIN → ACK`). TCP başlığı 20 bayttır ve şu alanları "
               "taşır: kaynak/hedef port, sıra numarası, ACK numarası, "
               "**bayraklar** (SYN, ACK, FIN, RST, PSH, URG), pencere boyutu, "
               "sağlama toplamı.",
               ["tcp", "güvenilirlik", "el sıkışma"]),
        Lesson("net_transport", "UDP nedir, neden hızlıdır?",
               "**UDP** bağlantısız ve güvenilmezdir: **el sıkışma yok, onay "
               "yok, yeniden iletim yok, sıra garantisi yok**. Başlığı yalnızca "
               "8 bayttır (TCP'de 20). Bu yüzden gecikmesi düşüktür ve "
               "**gerçek zamanlı** uygulamalar için uygundur: sesli/görüntülü "
               "konuşma, canlı yayın, **oyunlar**. Kaybolan paketi beklemek "
               "yerine yeni veri göndermek daha iyidir — eski konum bilgisinin "
               "geç değeri yoktur. Güvenilirlik gerekirse **uygulama katmanında** "
               "eklenir (sıra numarası, ACK, yeniden iletim).",
               ["udp", "gerçek zamanlı", "gecikme"]),
        Lesson("net_transport", "TCP mi UDP mi? Karar ölçütleri",
               "Karar **veri kaybının maliyetine** göre verilir. "
               "**TCP seç** eğer: her bayt gerekliyse (dosya, HTTP, e-posta, "
               "veritabanı), sıra önemliyse, gecikme hoş görülürse. "
               "**UDP seç** eğer: eski veri değersizse (konum, ses), gecikme "
               "kritikse, kayıp tolere edilebilirse, çok oyuncuya yayın "
               "yapılacaksa. **Hibrit** yaklaşım yaygındır: oyunlarda güvenilir "
               "olaylar (satın alma, sohbet) TCP/WebSocket, konum güncellemesi "
               "UDP üzerinden gider.",
               ["tcp", "udp", "karar"], code=(
                   "# KARAR TABLOSU\n"
                   "#   Dosya indirme     -> TCP   (her bayt sart)\n"
                   "#   Web sayfasi       -> TCP   (HTTP/1.1, HTTP/2)\n"
                   "#   Video konferans   -> UDP   (gecikme kritik)\n"
                   "#   Oyun konumu       -> UDP   (eski veri degersiz)\n"
                   "#   Oyun envanteri    -> TCP   (kayip kabul edilemez)\n"
                   "#   DNS sorgusu       -> UDP   (kucuk, tek paket)\n"
                   "#   DNS bolge aktarim -> TCP   (buyuk, sirali)")),
        Lesson("net_transport", "Port numaraları ve servis eşlemesi",
               "**Port** 16 bittir (0–65535) ve aynı IP'de birden fazla "
               "servisi ayırır. Aralıklar: **0–1023** iyi bilinen (yönetici "
               "izni gerekir), **1024–49151** kayıtlı, **49152–65535** geçici "
               "(istemci tarafı). Kritik portlar: **20/21** FTP, **22** SSH, "
               "**25** SMTP, **53** DNS, **80** HTTP, **123** NTP, **443** "
               "HTTPS, **3306** MySQL, **5432** PostgreSQL, **6379** Redis, "
               "**27015–27030** Source oyun sunucuları, **7777** Unreal, "
               "**8080** alternatif HTTP.",
               ["port", "servis", "iana"]),
        Lesson("net_transport", "Soket programlama: sunucu ve istemci",
               "**Soket** = (IP, port, protokol) üçlüsü. Sunucu akışı: "
               "`socket() → bind() → listen() → accept() → recv/send → close`. "
               "İstemci akışı: `socket() → connect() → send/recv → close`. "
               "`bind()` portu **rezerve eder**; aynı porta ikinci kez "
               "bağlanmak hata verir (`Address already in use`). "
               "`SO_REUSEADDR` bu hatayı aşar (test sırasında pratik). "
               "`listen(backlog)` kabul bekleyen bağlantı kuyruğunun boyutudur.",
               ["soket", "sunucu", "istemci"], code=(
                   "import socket\n"
                   "\n"
                   "# --- Sunucu ---\n"
                   "sunucu = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                   "sunucu.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                   "sunucu.bind(('127.0.0.1', 9000))\n"
                   "sunucu.listen(5)\n"
                   "print('dinleniyor: 127.0.0.1:9000')\n"
                   "\n"
                   "baglanti, adres = sunucu.accept()\n"
                   "veri = baglanti.recv(1024)\n"
                   "baglanti.sendall(b'merhaba ' + veri)\n"
                   "baglanti.close(); sunucu.close()")),
        Lesson("net_transport", "TCP akış kontrolü ve tıkanıklık denetimi",
               "**Akış kontrolü** alıcıyı korur: alıcı **pencere boyutunu** "
               "(window) bildirir, gönderici bu kadar veri gönderir. "
               "**Tıkanıklık denetimi** ağı korur: **yavaş başlangıç** "
               "(congestion window ikiye katlanır), **AIMD** (kayıpta yarıya "
               "in, başarıda birer artır), **fast retransmit** (3 tekrarlı ACK). "
               "Algoritmalar: **CUBIC** (Linux varsayılanı), **BBR** (Google, "
               "gecikme tabanlı). Bufferbloat: aşırı tampon gecikmeyi şişirir.",
               ["tcp", "akış kontrolü", "tıkanıklık"]),
        Lesson("net_transport", "QUIC ve HTTP/3: UDP üzerinde güvenilirlik",
               "**QUIC**, UDP üzerine kurulu yeni nesil taşıma protokolüdür. "
               "TCP'nin **head-of-line blocking** sorununu çözer: bir paket "
               "kaybolduğunda diğer akışlar **beklemez**. TLS 1.3 yerleşiktir "
               "(1 RTT'de bağlantı, 0-RTT yeniden bağlantı). **Bağlantı "
               "geçişi** vardır: Wi-Fi'dan mobile geçince bağlantı kopmaz "
               "(connection ID). HTTP/3, QUIC üzerinde çalışır. Oyunlarda da "
               "kullanılmaya başlandı (düşük gecikme + güvenilir kanal).",
               ["quic", "http3", "head-of-line"]),
        Lesson("net_transport", "NAT geçişi: STUN, TURN ve hole punching",
               "Eşten eşe (P2P) bağlantı NAT yüzünden zordur. **STUN** "
               "sunucusu size 'dışarıdan nasıl görünüyorsunuz' der (genel "
               "IP:port). **Hole punching**: iki taraf da eşzamanlı dışarı "
               "paket gönderir, NAT eşlemeleri açılır. **Symmetric NAT** "
               "bunu bozar (her hedefe farklı port). O zaman **TURN** "
               "sunucusu aracı olur: trafik onun üzerinden akar (bant "
               "maliyeti yüksek ama garantili). Sıra: doğrudan → STUN → TURN.",
               ["nat", "stun", "turn", "p2p"]),
        Lesson("net_transport", "WebSocket: TCP üzerinde çift yönlü kanal",
               "HTTP istek-yanıt modelidir; sunucu **kendiliğinden** veri "
               "gönderemez. **WebSocket** TCP bağlantısını yükseltir "
               "(`Upgrade: websocket`) ve **tam çift yönlü** kanal açar. "
               "Çerçeve başlığı yalnızca 2-10 bayttır (HTTP başlıkları "
               "yüzlerce bayt). Tarayıcıdan doğrudan kullanılabilir — bu "
               "yüzden web tabanlı oyunlarda ve canlı panolarda yaygındır. "
               "Kapatma el sıkışması ve **ping/pong** ile bağlantı canlı "
               "tutulur.",
               ["websocket", "çift yönlü", "tarayıcı"], code=(
                   "# Sunucu (websockets kutuphanesi)\n"
                   "import asyncio, websockets\n"
                   "\n"
                   "async def isle(ws):\n"
                   "    async for mesaj in ws:\n"
                   "        await ws.send(f'yankı: {mesaj}')\n"
                   "\n"
                   "async def main():\n"
                   "    async with websockets.serve(isle, '127.0.0.1', 8765):\n"
                   "        await asyncio.Future()\n"
                   "\n"
                   "# asyncio.run(main())\n"
                   "# Istemci: javascript'te new WebSocket('ws://127.0.0.1:8765')")),
    ]


# ─── Katman 3: Uygulama Protokolleri ────────────────────────────────────────

def _application_lessons() -> List[Lesson]:
    """HTTP, TLS ve üst katman protokolleri."""
    return [
        Lesson("net_application", "HTTP nasıl çalışır?",
               "**HTTP** istemci-sunucu, istek-yanıt protokolüdür. İstek: "
               "`METOT /yol HTTP/1.1` + başlıklar + gövde. Metotlar: **GET** "
               "(oku), **POST** (oluştur), **PUT** (tümünü değiştir), "
               "**PATCH** (kısmen değiştir), **DELETE** (sil), **HEAD** "
               "(yalnızca başlık). Durum kodları: **2xx** başarı, **3xx** "
               "yönlendirme, **4xx** istemci hatası (400 bozuk istek, 401 "
               "kimlik yok, 403 yasak, 404 bulunamadı, 429 çok istek), "
               "**5xx** sunucu hatası (500 iç hata, 502 kötü ağ geçidi, "
               "503 kullanılamıyor, 504 zaman aşımı).",
               ["http", "metot", "durum kodu"]),
        Lesson("net_application", "HTTP/1.1, HTTP/2 ve HTTP/3 farkları",
               "**HTTP/1.1** her istek için ayrı bağlantı eğilimindedir "
               "(keep-alive ile tekrar kullanılır) ve **head-of-line blocking** "
               "yaşar. **HTTP/2** tek TCP bağlantısında **çoklama** "
               "(multiplexing) yapar, başlıkları sıkıştırır (HPACK), sunucu "
               "itmesi destekler — ama TCP kaybı tüm akışları bekletir. "
               "**HTTP/3** QUIC üzerinde çalışır: kayıp yalnızca bir akışı "
               "etkiler, bağlantı geçişi vardır, 0-RTT destekler.",
               ["http2", "http3", "çoklama"]),
        Lesson("net_application", "TLS: şifreleme, sertifika ve el sıkışma",
               "**TLS** üç şey sağlar: **gizlilik** (şifreleme), "
               "**bütünlük** (değiştirilemez), **kimlik doğrulama** "
               "(sertifika). **TLS 1.3 el sıkışması 1 RTT**'dir: ClientHello "
               "(desteklenen şifreler + anahtar paylaşımı) → ServerHello + "
               "sertifika + Finished → Finished. **Sertifika** bir **CA** "
               "(sertifika otoritesi) tarafından imzalanır; tarayıcı CA "
               "zincirini doğrular. **SNI** aynı IP'de birden fazla alan adı "
               "barındırmayı sağlar. `https` = HTTP + TLS, varsayılan port "
               "**443**.",
               ["tls", "sertifika", "https"]),
        Lesson("net_application", "REST, gRPC ve GraphQL karşılaştırması",
               "**REST**: HTTP metotları + kaynak yolları, JSON gövde. Basit "
               "ve önbelleklenebilir; aşırı/eksik veri çekebilir. **gRPC**: "
               "HTTP/2 + **Protobuf** (ikili, şema zorunlu), çift yönlü akış, "
               "kod üretimi. Servisler arası iletişimde hızlıdır ama tarayıcı "
               "desteği sınırlıdır. **GraphQL**: tek uç nokta, istemci hangi "
               "alanları istediğini söyler — aşırı çekmeyi önler; karmaşık "
               "sorgular sunucuyu yorabilir.",
               ["rest", "grpc", "graphql"]),
        Lesson("net_application", "Önbellek (cache) ve CDN",
               "**Önbellek** yanıtı saklar ve tekrar üretmez. HTTP başlıkları: "
               "**Cache-Control** (`max-age=3600`, `no-store`, `private`), "
               "**ETag** (içerik parmak izi), **If-None-Match** (değişti mi?), "
               "**Last-Modified**. **304 Not Modified** gövde göndermeden "
               "yanıt verir. **CDN** içeriği kullanıcıya yakın uç sunucularda "
               "tutar; ilk istek **origin**'e gider, sonrakiler uçtan gelir. "
               "**Cache invalidation** zor problemdir: içerik değişince "
               "dosya adına sürüm eklemek (`app.v2.js`) en güvenilir yoldur.",
               ["önbellek", "cdn", "etag"]),
        Lesson("net_application", "Kimlik doğrulama: oturum, token, JWT",
               "**Oturum çerezi** sunucuda durum tutar (session store) ve "
               "tarayıcı çerezi taşır. **Token** (JWT) durumsuzdur: imzalı "
               "JSON taşır, sunucu doğrulamak için veritabanına bakmaz — ama "
               "**iptal edilemez** (süresi dolana kadar geçerli). Doğru "
               "yaklaşım: kısa ömürlü **erişim tokeni** + uzun ömürlü "
               "**yenileme tokeni**. Saklama: `HttpOnly` + `Secure` çerez "
               "(XSS'e dayanıklı) veya bellek (yenilemede kaybolur). "
               "**OAuth2** üçüncü taraf yetkilendirme çerçevesidir.",
               ["kimlik", "jwt", "oauth"]),
        Lesson("net_application", "E-posta protokolleri: SMTP, IMAP, POP3",
               "**SMTP** (port 25/587) gönderir; **IMAP** (143/993) sunucuda "
               "tutup senkronize eder (çok cihaz); **POP3** (110/995) indirip "
               "siler (tek cihaz). Kimlik doğrulama ve şifreleme için "
               "**STARTTLS** veya doğrudan TLS portu kullanılır. Gönderen "
               "doğrulaması: **SPF** (hangi IP gönderebilir), **DKIM** "
               "(imza), **DMARC** (politika). Bunlar olmadan e-postanız "
               "spam'e düşer.",
               ["smtp", "imap", "spf"]),
        Lesson("net_application", "Zaman ve senkronizasyon: NTP",
               "**NTP** (port 123/UDP) saatleri senkronize eder. Katmanlı "
               "yapı (**stratum**): stratum 0 atom saati, 1 doğrudan bağlı, "
               "2+ ağ üzerinden. Tipik doğruluk: internet üzerinden 1-50 ms, "
               "yerel ağda <1 ms. **Neden önemli?** Dağıtık sistemlerde "
               "olay sırası, sertifika geçerliliği, log korelasyonu ve oyun "
               "zaman damgaları saate bağlıdır. **PTP** (IEEE 1588) mikrosaniye "
               "doğruluk verir.",
               ["ntp", "saat", "dağıtık"]),
    ]


__all__ = [
    "NETWORK_LAYERS", "NETWORK_LABELS",
    "_basics_lessons", "_transport_lessons", "_application_lessons",
]


# ─── Katman 4: Sunucu Mimarileri ────────────────────────────────────────────

def _server_lessons() -> List[Lesson]:
    """Main server, proxy, yük dengeleme, dağıtık yapı."""
    return [
        Lesson("net_servers", "Sunucu nedir? İstemci-sunucu modeli",
               "**Sunucu** bir kaynağı (veri, hesaplama, oyun durumu) "
               "paylaşan programdır; **istemci** ondan ister. Modelin özü: "
               "sunucu **tek doğruluk kaynağıdır** (single source of truth), "
               "istemciler ondan öğrenir. Karşıt model **P2P**'dir: her düğüm "
               "hem istemci hem sunucudur (BitTorrent, bazı oyunlar). "
               "İstemci-sunucu daha kolay doğrulanır ve hileyi önler; P2P "
               "daha ucuza ölçeklenir ama güven zordur.",
               ["sunucu", "istemci", "p2p"]),
        Lesson("net_servers", "Proxy server nedir, ne yapar?",
               "**Proxy** istemci ile sunucu arasında **aracıdır**: istemci "
               "isteği proxy'ye yollar, proxy sunucuya iletir. Kullanım "
               "amaçları: **önbellek** (tekrar eden isteği hızlı yanıtla), "
               "**anonimlik** (gerçek IP gizlenir), **içerik filtreleme** "
               "(okul/kurum), **günlükleme** (kim ne istedi), **bant "
               "tasarrufu**. **İleri proxy** (forward) istemciyi temsil eder; "
               "**ters proxy** (reverse) sunucuyu temsil eder. Fark: kimin "
               "arkasında durduğudur.",
               ["proxy", "önbellek", "anonimlik"]),
        Lesson("net_servers", "Ters proxy (reverse proxy) neden şart?",
               "**Ters proxy** sunucuların **önünde** durur ve istemciye "
               "sunucu gibi görünür. Sağladıkları: **yük dengeleme**, "
               "**TLS sonlandırma** (sertifika tek yerde), **sıkıştırma**, "
               "**önbellek**, **hız sınırlama** (rate limit), **sağlık "
               "denetimi** (ölü sunucuya göndermez), **yol yönlendirme** "
               "(`/api` → backend, `/static` → dosya). Örnekler: **nginx**, "
               "**Caddy**, **HAProxy**, **Traefik**. Uygulama sunucusu "
               "doğrudan internete açılmaz.",
               ["ters proxy", "nginx", "tls sonlandırma"]),
        Lesson("net_servers", "Yük dengeleme algoritmaları",
               "**Round Robin**: sırayla dağıt — eşit iş varsayar. "
               "**Ağırlıklı**: güçlü sunucuya daha çok. **En az bağlantı**: "
               "yükü gerçekten ölçer. **IP hash**: aynı istemci hep aynı "
               "sunucuya — oturum tutarlılığı sağlar. **En kısa yanıt "
               "süresi**: en hızlıya gönder. **Sağlık denetimi** başarısız "
               "sunucuyu havuzdan çıkarır; geri gelince yavaşça trafik "
               "verilir.",
               ["yük dengeleme", "round robin", "health check"]),
        Lesson("net_servers", "Mainserver ve proxy ilişkisi (katmanlı mimari)",
               "**Mainserver** (ana sunucu) iş mantığını ve durumu tutar: "
               "kimlik doğrulama, oyuncu verisi, maç eşleştirme, envanter. "
               "**Proxy** önünde durur ve şunları yapar: TLS sonlandırma, "
               "**kimlik ön denetimi** (geçersiz token'ı ana sunucuya "
               "göndermeden reddet), **hız sınırlama** (DDoS yumuşatma), "
               "**yönlendirme** (oyuncuyu doğru oyun sunucusuna gönder), "
               "**önbellek**. Kritik kural: ana sunucu **asla** doğrudan "
               "internete açılmaz — yalnızca proxy'den gelen trafiği kabul "
               "eder (özel ağ).",
               ["mainserver", "proxy", "katmanlı"], code=(
                   "# KATMANLI YAPI\n"
                   "#\n"
                   "#  Internet\n"
                   "#      |  (yalnizca 443 acik)\n"
                   "#      v\n"
                   "#   +----------------+\n"
                   "#   | PROXY / EDGE   |  TLS, rate limit, on denetim\n"
                   "#   +----------------+\n"
                   "#      |  (ozel ag, 10.x.x.x)\n"
                   "     v\n"
                   "#   +----------------+     +------------------+\n"
                   "#   | MAINSERVER     |---->| Veritabani       |\n"
                   "#   | is mantigi     |     +------------------+\n"
                   "#   +----------------+     +------------------+\n"
                   "#      |                   | Cache (Redis)    |\n"
                   "#      v                   +------------------+\n"
                   "  +----------------+\n"
                   "#   | OYUN SUNUCULARI|  mac basina bir surec\n"
                   "#   +----------------+")),
        Lesson("net_servers", "Dağıtık sistem: ölçekleme ve tutarlılık",
               "**Dikey ölçekleme** (daha güçlü makine) sınırlıdır; **yatay "
               "ölçekleme** (daha çok makine) tercih edilir. **CAP teoremi**: "
               "tutarlılık (C), erişilebilirlik (A), bölünme toleransı (P) "
               "üçünden en fazla ikisi aynı anda garanti edilebilir; ağ "
               "bölünmesi kaçınılmaz olduğu için pratikte **CP** veya **AP** "
               "seçilir. **Durumsuz (stateless) servis** kolay ölçeklenir; "
               "durum veritabanına/önbelleğe taşınır. **Idempotent** işlem "
               "tekrar güvenlidir (aynı istek iki kez gelse de sonuç aynı).",
               ["dağıtık", "cap", "ölçekleme"]),
        Lesson("net_servers", "Mesaj kuyruğu ve olay tabanlı mimari",
               "Servisler birbirini **doğrudan çağırmak** yerine kuyruğa "
               "mesaj bırakır: **gevşek bağlılık** (biri düşse diğeri "
               "etkilenmez), **tampon** (ani yük emilir), **tekrar deneme**, "
               "**sıralı işleme**. Araçlar: **RabbitMQ**, **Kafka** (yüksek "
               "hacim, kalıcı log), **Redis Pub/Sub** (hafif). Kalıplar: "
               "**kuyruk** (bir tüketici işler), **yayın-abone** (herkes "
               "alır), **olay akışı** (geçmiş saklanır).",
               ["kuyruk", "kafka", "olay"]),
        Lesson("net_servers", "Konteyner, orkestrasyon ve dağıtım",
               "**Konteyner** (Docker) uygulamayı bağımlılıklarıyla paketler: "
               "her yerde aynı çalışır. **İmaj** katmanlardan oluşur ve "
               "önbelleklenir; **konteyner** imajın çalışan örneğidir. "
               "**Orkestrasyon** (Kubernetes) kaç kopya çalışacağını, hangi "
               "sunucuda duracağını, sağlık denetimini, otomatik yeniden "
               "başlatmayı, yük dengelemeyi, sır ve yapılandırma yönetimini "
               "üstlenir. **Servis keşfi** dinamik adresleri çözer.",
               ["docker", "kubernetes", "orkestrasyon"]),
        Lesson("net_servers", "Gözlemlenebilirlik: log, metrik, izleme",
               "Üç sütun: **loglar** (ne oldu), **metrikler** (ne kadar — "
               "istek/sn, gecikme yüzdelik, hata oranı, CPU), **izler** "
               "(nerede yavaş — bir isteğin servisler arası yolu). "
               "**Yüzdelik (p50/p95/p99)** ortalamadan iyidir: ortalama 50 ms "
               "olsa da p99 2 sn ise kullanıcıların %1'i kötü deneyim yaşar. "
               "**SLI** ölçülen, **SLO** hedeflenen, **SLA** sözleşmeye bağlı "
               "değerdir.",
               ["gözlemlenebilirlik", "metrik", "slo"]),
        Lesson("net_servers", "Oyun sunucusu türleri: listen, dedicated, P2P",
               "**Listen server**: oyunculardan biri hem oynar hem sunuculuk "
               "yapar — ucuz ama ev sahibi çıkınca oyun biter ve avantajlıdır. "
               "**Dedicated server**: yalnızca sunuculuk yapar, oyuncu "
               "değildir — adil ve kararlı, maliyetlidir. **P2P**: oyuncular "
               "doğrudan bağlanır — sunucu maliyeti yok ama hile ve senkron "
               "sorunu büyüktür. **Authoritative server**: kararı sunucu "
               "verir. Rekabetçi oyunlarda **dedicated + authoritative** "
               "zorunludur.",
               ["dedicated", "listen", "authoritative"]),
        Lesson("net_servers", "Mainserver + proxy + oyun sunucusu akışı",
               "Tam akış: **1)** İstemci proxy'ye bağlanır (TLS). **2)** "
               "Proxy kimliği doğrular ve mainserver'a iletir. **3)** "
               "Mainserver **maç eşleştirme** yapar, uygun oyun sunucusunun "
               "adresini ve **biletini** (join token) döner. **4)** İstemci "
               "doğrudan oyun sunucusuna bağlanır (genelde UDP) ve bileti "
               "sunar. **5)** Oyun sunucusu bileti mainserver'a doğrulatır. "
               "**6)** Maç biter, sonuç mainserver'a bildirilir. Bu ayrım, "
               "oyun sunucusunun kimlik doğrulama yükünü taşımamasını sağlar.",
               ["akış", "eşleştirme", "bilet"], code=(
                   "# AKIS\n"
                   "#   Istemci --TLS--> Proxy --ozel ag--> MainServer\n"
                   "#                                           |\n"
                   "#                        eslestirme + bilet  |\n"
                   "#                                           v\n"
                   "#   Istemci <---- sunucu adresi + bilet ---- MainServer\n"
                   "#      |\n"
                   "#      +--UDP--> Oyun Sunucusu --bilet dogrula--> MainServer\n"
                   "#                     |\n"
                   "#                     +-- sonuc --> MainServer --> Odul")),
    ]


# ─── Katman 5: Oyun Sunucuları ──────────────────────────────────────────────

def _game_lessons() -> List[Lesson]:
    """Oyun ağı: tick, tahmin, telafi, senkronizasyon."""
    return [
        Lesson("net_game", "Tick oranı ve simülasyon döngüsü",
               "Sunucu oyunu **tick** adı verilen sabit adımlarla ilerletir "
               "(`fizik_dt = 1/60`). Tick sırası: **1)** istemci girdilerini "
               "uygula, **2)** fiziği ilerlet, **3)** çarpışmaları çöz, "
               "**4)** oyun kurallarını işle, **5)** durumu istemcilere "
               "yayınla. Tick **sabit** olmalıdır; kare hızına bağlamak "
               "determinizmi bozar. 20 Hz strateji için yeter, 64–128 Hz "
               "rekabetçi FPS için gereklidir. Tick uzarsa sunucu yetişemez "
               "(tick overrun).",
               ["tick", "simülasyon", "determinizm"]),
        Lesson("net_game", "İstemci tahmini (client-side prediction)",
               "Ağ gecikmesi 100 ms ise girdiyi beklemek oyunu oynanamaz "
               "kılar. **Tahmin**: istemci girdisini **hemen** yerel olarak "
               "uygular, sonra sunucuya gönderir. Sunucu doğru durumu "
               "döndüğünde istemci tahminini **düzeltir**. Yanlış tahminde "
               "**düzeltme (reconciliation)**: istemci son onaylanmış "
               "durumdan itibaren girdileri yeniden oynatır. Görsel sıçramayı "
               "önlemek için yumuşak geçiş uygulanır.",
               ["tahmin", "düzeltme", "gecikme"]),
        Lesson("net_game", "Gecikme telafisi (lag compensation)",
               "Sunucu, oyuncunun ateş ettiği **ana** durumu değil, o anda "
               "**istemcinin gördüğü** durumu değerlendirmelidir. Yöntem: "
               "sunucu son ~1 saniyelik durum geçmişini saklar; ateş "
               "bildirimi geldiğinde oyuncunun gecikmesi kadar **geri sarar** "
               "(rewind), isabeti o anda hesaplar. Bu, yüksek gecikmeli "
               "oyuncuya avantaj verir (peeker's advantage) — bu yüzden üst "
               "sınır konur (ör. 200 ms).",
               ["lag compensation", "rewind", "isabet"]),
        Lesson("net_game", "Ara değerleme (interpolation) ve ekstrapolasyon",
               "Sunucu saniyede 20 güncelleme gönderirse hareket sıçrar. "
               "**Ara değerleme**: istemci son iki durumu alıp arada yumuşak "
               "geçiş yapar; genelde ~100 ms **geçmişte** render eder "
               "(tampon). **Ekstrapolasyon**: yeni veri gelmezse son hız ve "
               "yönden ileri tahmin eder — kayıpta akıcılık sağlar ama yanlış "
               "olabilir. Kural: **diğer oyuncular interpole, kendi oyuncun "
               "tahmin edilir**.",
               ["interpolation", "ekstrapolasyon", "tampon"]),
        Lesson("net_game", "Yetkili sunucu ve hile önleme",
               "**Authoritative server** tüm kararları verir: konum, isabet, "
               "hasar, envanter. İstemci yalnızca **girdi** gönderir ('ileri "
               "basılı', 'ateş ettim'), sonuç değil. Böylece istemci 'ben "
               "öldürdüm' diyemez. Ek korumalar: **girdi doğrulama** "
               "(imkânsız hız reddedilir), **sunucu tarafı kontrol** (duvar "
               "içinden geçme), **hız sınırı** (saniyede en fazla N olay), "
               "**görüş alanı denetimi** (görmediği veriyi gönderme — "
               "wallhack önleme).",
               ["authoritative", "hile", "girdi doğrulama"]),
        Lesson("net_game", "İlgi yönetimi (interest management)",
               "Sunucu tüm oyunculara her şeyi göndermez. **İlgi yönetimi**: "
               "her oyuncuya yalnızca görüş/bölge içindeki nesneleri gönder. "
               "Yöntemler: **mesafe tabanlı**, **ızgara/bölge** (grid, "
               "spatial hash), **portallı** (oda geçişi), **görüş konisi** "
               "(FPS). **Önceliklendirme**: bant yetmezse yakın/hareketli/"
               "hedef nesneler önce gönderilir. 100 oyuncu × 200 nesne = "
               "20.000 güncelleme; ilgi yönetimi bunu 100 × 20'ye indirir.",
               ["ilgi yönetimi", "grid", "öncelik"]),
        Lesson("net_game", "Bant genişliği bütçesi ve sıkıştırma",
               "Her istemciye saniyede gönderilebilecek veri **bütçelenir** "
               "(ör. 32 KB/s). Tasarruf yolları: **nicemleme (quantization)** "
               "(konumu float yerine 16 bit tam sayı), **delta sıkıştırma** "
               "(yalnızca değişeni gönder), **bit paketleme** (bayrak 1 bit), "
               "**değişken hız** (uzak nesne seyrek güncellenir), "
               "**önbellek** (değişmeyen veri bir kez).",
               ["bant", "nicemleme", "delta"]),
        Lesson("net_game", "Oyun sunucusu yaşam döngüsü",
               "Bir maç için sunucu süreci: **1) Oluştur** (container/VM, "
               "~1-30 sn), **2) Hazırla** (harita, mod, port), **3) Duyur** "
               "(mainserver'a 'hazırım, adresim bu'), **4) Kabul et** "
               "(oyuncular biletiyle bağlanır), **5) Çalıştır** (tick "
               "döngüsü), **6) Bitir** (sonucu bildir), **7) Kapat** "
               "(kaynakları bırak). Ölçek için **sunucu havuzu** tutulur: "
               "hazır bekleyen boş sunucular anında atanır.",
               ["yaşam döngüsü", "havuz", "orkestrasyon"]),
        Lesson("net_game", "Eşleştirme (matchmaking) algoritması",
               "**1)** Oyuncular kuyruğa girer (mod, bölge, beceri). **2)** "
               "**Beceri puanı** (ELO/Glicko/TrueSkill) ile yakın oyuncular "
               "gruplanır. **3)** Zaman geçtikçe kabul aralığı **genişletilir** "
               "(bekleyen oyuncu oyun bulmalı). **4)** **Takım dengeleme** "
               "(toplam beceri yakın, rol dağılımı uygun). **5)** Bölge/gecikme "
               "sınırı (<80 ms tercih). **6)** Uygun sunucu atanır. Kötü "
               "eşleştirme, iyi ağ kodundan daha çok oyuncu kaybettirir.",
               ["eşleştirme", "elo", "takım"]),
        Lesson("net_game", "Sunucu tarafı doğrulama ve ceza",
               "Sunucu şüpheli davranışı **ölçer**: isabet oranı aşırı yüksek, "
               "tepki süresi insan altı, hareket imkânsız, paket deseni "
               "düzensiz. **Otomatik** tepki: düzeltme, uyarı, maç iptali, "
               "geçici yasak. **Yanlış pozitiften kaçınmak** için tek ölçüte "
               "dayanmak yerine birden fazla sinyal birleştirilir ve **kanıt "
               "kaydedilir** (oyun tekrarı, log). Karar insan incelemesine "
               "açık olmalıdır.",
               ["doğrulama", "anti-cheat", "kanıt"]),
    ]


# ─── Katman 6: Godot ve Ağ Programlama ──────────────────────────────────────

def _godot_lessons() -> List[Lesson]:
    """Godot çok oyunculu API, RPC, senkronizasyon."""
    return [
        Lesson("net_godot", "Godot çok oyunculu mimarisi",
               "Godot'ta ağ katmanı **`MultiplayerAPI`** ile yönetilir: "
               "`MultiplayerPeer` (ENet, WebSocket, WebRTC) bağlantıyı, "
               "`MultiplayerAPI` ise RPC ve senkronizasyonu sağlar. "
               "**Sunucu** `create_server(port, max_clients)`, **istemci** "
               "`create_client(address, port)` çağırır. "
               "`multiplayer.peer_connected` / `peer_disconnected` "
               "sinyalleriyle olaylar izlenir. **Sahne replikasyonu**: "
               "`MultiplayerSpawner`, sunucuda oluşturulan nesneleri "
               "istemcilere otomatik kopyalar.",
               ["godot", "multiplayerapi", "enet"], code=(
                   "# Sunucu\n"
                   "var peer := ENetMultiplayerPeer.new()\n"
                   "peer.create_server(7777, 16)\n"
                   "multiplayer.multiplayer_peer = peer\n"
                   "\n"
                   "# Istemci\n"
                   "var peer := ENetMultiplayerPeer.new()\n"
                   "peer.create_client(\"127.0.0.1\", 7777)\n"
                   "multiplayer.multiplayer_peer = peer\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    multiplayer.peer_connected.connect(_on_baglan)\n"
                   "    multiplayer.peer_disconnected.connect(_on_ayril)\n"
                   "\n"
                   "func _on_baglan(id: int) -> void:\n"
                   "    print(\"oyuncu katildi: \", id)\n"
                   "\n"
                   "func _on_ayril(id: int) -> void:\n"
                   "    print(\"oyuncu ayrildi: \", id)")),
        Lesson("net_godot", "RPC: uzaktan fonksiyon çağrısı",
               "Godot'ta **RPC** bir fonksiyonu başka makinede çalıştırır. "
               "İşaretleyici: **`@rpc(\"any_peer\", \"call_remote\", "
               "\"reliable\")`**. Modlar: **yetki** (`authority` yalnız "
               "sunucu, `any_peer` herkes), **çağrı** (`call_remote`, "
               "`call_local` yerelde de çalışsın), **aktarım** (`reliable` "
               "kayıp yok, `unreliable` hızlı, `unreliable_ordered` sıralı "
               "hızlı). **Güvenlik:** `any_peer` ile gelen çağrıda "
               "`multiplayer.get_remote_sender_id()` doğrulanmalıdır — aksi "
               "hâlde istemci sunucu komutu çalıştırabilir.",
               ["rpc", "godot", "güvenlik"], code=(
                   "# Sunucu -> tum istemciler (guvenilir)\n"
                   "@rpc(\"authority\", \"call_remote\", \"reliable\")\n"
                   "func skor_guncelle(yeni_skor: int) -> void:\n"
                   "    $UI/Skor.text = str(yeni_skor)\n"
                   "\n"
                   "# Istemci -> sunucu (girdi gonderme, hizli)\n"
                   "@rpc(\"any_peer\", \"call_remote\", \"unreliable_ordered\")\n"
                   "func girdi_gonder(yon: Vector2) -> void:\n"
                   "    var gonderen := multiplayer.get_remote_sender_id()\n"
                   "    # GUVENLIK: yalnizca sunucu isler, gonderen dogrulanir\n"
                   "    if not multiplayer.is_server():\n"
                   "        return\n"
                   "    _girdiyi_uygula(gonderen, yon)")),
        Lesson("net_godot", "MultiplayerSynchronizer ile durum yayını",
               "**`MultiplayerSynchronizer`** seçilen özellikleri otomatik "
               "eşitler. `replication_config` içine eklenen özellikler "
               "(`position`, `rotation`, özel değişkenler) belirtilen "
               "aralıkta gönderilir: **replication_interval** ve "
               "**delta_interval**. **Yetki (authority)** kimin yazacağını "
               "belirler: genelde sunucu yazar, istemciler okur. İstemci "
               "girdisi ayrı RPC ile gönderilir, sonra sunucu senkronize "
               "eder.",
               ["synchronizer", "godot", "yetki"], code=(
                   "# Sahne: Player (CharacterBody3D)\n"
                   "#   + MultiplayerSynchronizer (replication_interval=0.05)\n"
                   "#       replication_config: position, velocity, saglik\n"
                   "\n"
                   "func _enter_tree() -> void:\n"
                   "    # Sunucu bu dugumun yetkilisidir\n"
                   "    $MultiplayerSynchronizer.set_multiplayer_authority(1)\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    # Yalnizca yetkili taraf fiziği ilerletir\n"
                   "    if not is_multiplayer_authority():\n"
                   "        return\n"
                   "    velocity.y -= 9.8 * delta\n"
                   "    move_and_slide()")),
        Lesson("net_godot", "MultiplayerSpawner ile nesne oluşturma",
               "**`MultiplayerSpawner`** yeni nesneleri tüm istemcilere "
               "kopyalar. Ayarlar: **spawn_path** (nesnelerin ekleneceği "
               "düğüm), **spawn_limit**, **auto_spawn_list** (hangi sahne "
               "türleri). Sunucu `spawner.spawn(veri)` çağırır, istemcilerde "
               "otomatik oluşur. **Önemli:** nesneyi elle `add_child` ile "
               "eklemek yerine spawner kullanılmalıdır; aksi hâlde "
               "istemcilerde oluşmaz. Oyuncu çıkınca `despawn` çağrılır.",
               ["spawner", "godot", "nesne"]),
        Lesson("net_godot", "ENet, WebSocket ve WebRTC seçimi",
               "**ENet** (UDP tabanlı): varsayılan ve en hızlı; kanal "
               "sistemleri hazır. Masaüstü ve mobil için idealdir. "
               "**WebSocket**: TCP tabanlı, **tarayıcıdan** çalışır (HTML5 "
               "ihracı) ama gecikmesi yüksektir. **WebRTC**: tarayıcıda UDP "
               "benzeri düşük gecikme + P2P; kurulumu karmaşıktır (STUN/TURN "
               "gerekir). Seçim: masaüstü → ENet, web + düşük gecikme → "
               "WebRTC, web + basitlik → WebSocket.",
               ["enet", "websocket", "webrtc"], code=(
                   "# Masaustu/mobil icin ENet (en hizli)\n"
                   "var peer := ENetMultiplayerPeer.new()\n"
                   "\n"
                   "# Tarayici (HTML5) icin WebSocket\n"
                   "# var peer := WebSocketMultiplayerPeer.new()\n"
                   "# peer.create_server(7777)   # sunucu\n"
                   "# peer.create_client(\"ws://127.0.0.1:7777\")  # istemci\n"
                   "\n"
                   "peer.create_server(7777, 16)\n"
                   "multiplayer.multiplayer_peer = peer")),
        Lesson("net_godot", "Sunucu tarafı Godot: headless çalıştırma",
               "Oyun sunucusu **arayüzsüz** çalışır: `godot --headless` "
               "(Godot 4). Sunucu sahnesi yalnızca mantık ve çarpışma içerir; "
               "görsel yoktur. **Fizik sunucuda** ilerletilir, istemciye "
               "durum gönderilir. **Yapılandırma** komut satırından okunur "
               "(`--port`, `--max-players`, `--map`). **Çıkış kodu**: hata "
               "varsa 1 döndürülür ki orkestratör yeniden başlatabilsin. "
               "Log'lar stdout'a yazılır ve toplanır.",
               ["headless", "godot", "sunucu"], code=(
                   "# Sunucu betigi: komut satiri argumanlarini oku\n"
                   "func _ready() -> void:\n"
                   "    var args := OS.get_cmdline_args()\n"
                   "    var port := 7777\n"
                   "    var maks := 16\n"
                   "    for i in args.size():\n"
                   "        if args[i] == \"--port\":\n"
                   "            port = int(args[i + 1])\n"
                   "        elif args[i] == \"--max-players\":\n"
                   "            maks = int(args[i + 1])\n"
                   "    _sunucu_baslat(port, maks)\n"
                   "    print(\"SUNUCU_HAZIR \", port, \" \", maks)\n"
                   "\n"
                   "# Calistirma:\n"
                   "# godot --headless --path . -- --port 7777 --max-players 16")),
        Lesson("net_godot", "Güvenlik: istemciye asla güvenme",
               "Godot'ta `@rpc(\"any_peer\")` **her istemcinin** o fonksiyonu "
               "çağırabilmesi demektir. Kurallar: **1)** Sunucu tarafında "
               "`multiplayer.is_server()` denetle. **2)** "
               "`get_remote_sender_id()` ile kimin çağırdığını doğrula. "
               "**3)** Parametreleri doğrula (menzil, tip, hız sınırı). "
               "**4)** Hassas işlem (hasar, para, envanter) yalnız sunucuda. "
               "**5)** İstemciden gelen 'ben kazandım' **asla** kabul "
               "edilmez.",
               ["güvenlik", "godot", "doğrulama"], code=(
                   "# YANLIS: dogrulama yok\n"
                   "@rpc(\"any_peer\")\n"
                   "func hasar_ver(hedef: int, miktar: int) -> void:\n"
                   "    saglik[hedef] -= miktar      # KOTU\n"
                   "\n"
                   "# DOGRU: sunucu dogrular\n"
                   "@rpc(\"any_peer\", \"call_remote\", \"reliable\")\n"
                   "func hasar_iste(hedef: int) -> void:\n"
                   "    if not multiplayer.is_server():\n"
                   "        return\n"
                   "    var gonderen := multiplayer.get_remote_sender_id()\n"
                   "    if not _gecerli_hedef(gonderen, hedef):\n"
                   "        return                       # menzil/gorus disi\n"
                   "    saglik[hedef] -= 10              # miktari sunucu secer")),
        Lesson("net_godot", "Web tabanlı oyun: Godot HTML5 + WebSocket",
               "Godot'un **HTML5 ihracı** ENet'i desteklemez (tarayıcı ham "
               "UDP açamaz). Web ihracında **WebSocketMultiplayerPeer** "
               "kullanılır. Sunucu tarafı iki seçenek: **1)** Godot sunucu da "
               "WebSocket dinler, **2)** nginx/ters proxy WebSocket'i Godot "
               "sunucusuna yönlendirir. TLS için `wss://` gerekir (tarayıcı "
               "kuralı). Gecikme ENet'ten yüksektir; bu yüzden web oyunları "
               "daha çok tur tabanlı tasarlanır.",
               ["html5", "websocket", "godot"]),
        Lesson("net_godot", "Godot'ta ağ hata ayıklama",
               "Araçlar: **`multiplayer.get_peers()`** bağlı oyuncular, "
               "**`multiplayer.get_unique_id()`** kendi kimliğin, "
               "**`multiplayer.has_multiplayer_peer()`** bağlantı var mı. "
               "Yaygın hatalar: **1)** İstemci `create_server` çağırıyor "
               "(rol karışıklığı). **2)** Synchronizer'ın `authority` değeri "
               "yanlış (iki taraf da yazıyor → titreme). **3)** `any_peer` "
               "RPC'de doğrulama yok. **4)** Sunucu sahneyi yüklemeden "
               "istemci bağlanıyor. **5)** Güvenlik duvarı portu kapalı. "
               "Test: sunucu + iki istemci aynı makinede, farklı pencerelerde.",
               ["hata ayıklama", "godot", "test"], code=(
                   "# Baglanti durumu kontrolu\n"
                   "func durum_yazdir() -> void:\n"
                   "    if not multiplayer.has_multiplayer_peer():\n"
                   "        print(\"baglanti yok\")\n"
                   "        return\n"
                   "    print(\"kimlik: \", multiplayer.get_unique_id())\n"
                   "    print(\"sunucu muyum: \", multiplayer.is_server())\n"
                   "    print(\"bagli oyuncular: \", multiplayer.get_peers())\n"
                   "\n"
                   "# Test duzeni (ayni makine):\n"
                   "#   1. pencere: --server --port 7777\n"
                   "#   2. pencere: --client --port 7777\n"
                   "#   3. pencere: --client --port 7777")),
    ]


# ─── Müfredat ───────────────────────────────────────────────────────────────

def build_network_curriculum(
        layers: Sequence[str] | None = None) -> List[Lesson]:
    """
    Ağ müfredatını üretir.

    Args:
        layers: Yalnızca bu katmanlar (None = hepsi, sıralı)

    Returns:
        Ders listesi
    """
    builders = {
        "net_basics": _basics_lessons,
        "net_transport": _transport_lessons,
        "net_application": _application_lessons,
        "net_servers": _server_lessons,
        "net_game": _game_lessons,
        "net_godot": _godot_lessons,
    }
    istenen = list(layers) if layers else list(NETWORK_LAYERS)
    dersler: List[Lesson] = []
    for ad in istenen:
        uretici = builders.get(ad)
        if uretici is not None:
            dersler.extend(uretici())
    return dersler


def network_stats(layers: Sequence[str] | None = None) -> Dict[str, object]:
    """Ağ müfredatı istatistikleri."""
    dersler = build_network_curriculum(layers)
    per_layer: Dict[str, Dict[str, int]] = {}
    for ders in dersler:
        kayit = per_layer.setdefault(ders.layer,
                                     {"lessons": 0, "chars": 0, "code": 0})
        kayit["lessons"] += 1
        kayit["chars"] += len(ders.as_document())
        kayit["code"] += 1 if ders.code.strip() else 0
    return {
        "total_lessons": len(dersler),
        "total_chars": sum(item["chars"] for item in per_layer.values()),
        "per_layer": per_layer,
        "layers": [NETWORK_LABELS.get(a, a)
                   for a in (layers or NETWORK_LAYERS)],
    }


__all__ = [
    "NETWORK_LAYERS", "NETWORK_LABELS", "build_network_curriculum",
    "network_stats",
]
