"""
NeuroCore Müfredatı
====================
Yerel modelin **sıfırdan** öğrenmesi için katmanlı ders içeriği.

Neden bu modül?
    Ajan deneyimlerinden öğrenme (sohbetler, görevler) **dar** bir veri
    kümesi üretir: aynı tür görevler, sınırlı sözcük dağarcığı. Modelin
    temel kavramları (sayı sistemleri, donanım, işletim sistemi, diller,
    kütüphaneler, 3B modelleme) öğrenmesi için **yapılandırılmış** bir
    müfredat gerekir.

Katmanlar (sıralı öğretilir):
    1. `BINARY`    — ikili sayı sistemi, mantık kapıları, bit/bayt, kodlama
    2. `HARDWARE`  — CPU, bellek, veri yolu, BIOS/UEFI açılış süreci
    3. `OS`        — çekirdek, süreç, bellek yönetimi, dosya sistemi, kabuk
    4. `LANGUAGES` — programlama dilleri, Python ve standart kütüphanesi
    5. `LIBRARIES` — kütüphaneler ve çerçeveler (numpy, asyncio, torch, bpy)
    6. `BLENDER`   — bpy ile 3B nesne oluşturma, malzeme, render, dışa aktarma

Her ders `Lesson` olarak tutulur ve **eğitim çifti** üretir:

    prompt (girdi)  : başlık + soru
    target (hedef)  : ders gövdesi

Bu, modelin "soru → açıklama" biçimini öğrenmesini sağlar. İçerik **Türkçe**
yazılır (kullanıcının dili) ama teknik terimler İngilizce bırakılır; model
her iki biçimi de görmelidir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

#: Müfredat katmanları (sıra önemlidir: temelden karmaşığa)
#: Yerel katmanlar (bu modülde tanımlı)
LAYERS: Sequence[str] = (
    "binary", "hardware", "os", "languages", "libraries", "blender",
)

#: Ağ katmanları — `network.py` içinde tanımlı, burada yalnızca **sıra**
#: bilgisi tutulur. Böylece `build_curriculum()` ağ derslerini de üretir ve
#: tokenizer/kasa/eğitim hattı hepsini birlikte görür.
NETWORK_LAYER_NAMES: Sequence[str] = (
    "net_basics", "net_transport", "net_application",
    "net_servers", "net_game", "net_godot",
)

#: Lowpoly/prosedürel içerik katmanları — `lowpoly.py` içinde tanımlı
LOWPOLY_LAYER_NAMES: Sequence[str] = (
    "lowpoly_temel", "lowpoly_bitki", "lowpoly_arazi",
    "lowpoly_doku", "lowpoly_detay",
)

#: Godot oyun üretim katmanları — `godot_learn.py` içinde tanımlı
GODOT_LAYER_NAMES: Sequence[str] = (
    "godot_temel", "godot_3b", "godot_nesne", "godot_materyal",
    "godot_karakter", "godot_seviye", "godot_girdi", "godot_arayuz",
    "godot_ses", "godot_ag",
)

#: Tüm katmanlar (yerel + ağ + lowpoly + godot)
ALL_LAYERS: Sequence[str] = (tuple(LAYERS) + tuple(NETWORK_LAYER_NAMES)
                             + tuple(LOWPOLY_LAYER_NAMES)
                             + tuple(GODOT_LAYER_NAMES))

#: Katman etiketleri (arayüz ve rapor için)
LAYER_LABELS: Dict[str, str] = {
    "binary": "İkili Sayı Sistemi",
    "hardware": "Donanım ve Açılış",
    "os": "İşletim Sistemi",
    "languages": "Programlama Dilleri",
    "libraries": "Kütüphaneler",
    "blender": "Blender ile 3B Modelleme",
    # Ağ katmanları (network.py ile aynı etiketler)
    "net_basics": "Ağ Temelleri",
    "net_transport": "Taşıma Katmanı ve Protokoller",
    "net_application": "Uygulama Protokolleri",
    "net_servers": "Sunucu Mimarileri",
    "net_game": "Oyun Sunucuları",
    "net_godot": "Godot ve Ağ Programlama",
    # Lowpoly katmanları (lowpoly.py ile aynı etiketler)
    "lowpoly_temel": "Lowpoly Temelleri",
    "lowpoly_bitki": "Ağaç ve Çiçek Modelleme",
    "lowpoly_arazi": "Arazi (Terrain) Üretimi",
    "lowpoly_doku": "Material Katmanları (Texture)",
    "lowpoly_detay": "Su, Yol ve İyileştirmeler",
    # Godot oyun üretim katmanları (godot_learn.py ile aynı etiketler)
    "godot_temel": "Godot Temelleri",
    "godot_3b": "3B Dünya ve Mesh",
    "godot_nesne": "Stylized Lowpoly Nesneler",
    "godot_materyal": "Material ve Texture",
    "godot_karakter": "Karakter, İskelet ve Animasyon",
    "godot_seviye": "Seviye Tasarımı (Maze/Arena/Mağara)",
    "godot_girdi": "Girdi ve Tuş Atamaları",
    "godot_arayuz": "Arayüz (UI/HUD)",
    "godot_ses": "Ses ve Müzik",
    "godot_ag": "Ağ (Master/Proxy Server)",
}


@dataclass
class Lesson:
    """Tek bir müfredat dersi."""
    layer: str
    title: str
    body: str
    tags: List[str] = field(default_factory=list)
    code: str = ""

    @property
    def label(self) -> str:
        """Katman etiketi."""
        return LAYER_LABELS.get(self.layer, self.layer)

    def prompt(self) -> str:
        """
        Eğitim çiftinin **girdi** tarafı.

        Soru biçiminde yazılır ki model "soru → açıklama" kalıbını öğrensin.
        """
        return f"{self.label} — {self.title}"

    def target(self) -> str:
        """Eğitim çiftinin **hedef** tarafı (açıklama + kod)."""
        parts = [self.body.strip()]
        if self.code.strip():
            parts.append("```python\n" + self.code.strip() + "\n```")
        return "\n\n".join(parts)

    def as_document(self) -> str:
        """
        Tokenizer eğitimi için **ham belge**.

        Başlık ve gövde birlikte verilir; tokenizer teknik terimleri
        (fonksiyon adları, İngilizce kavramlar) tek parça olarak öğrensin.
        """
        parts = [f"# {self.prompt()}", self.body.strip()]
        if self.code.strip():
            parts.append(self.code.strip())
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serileştirilebilir hâl."""
        return {
            "layer": self.layer, "label": self.label, "title": self.title,
            "body": self.body, "code": self.code, "tags": self.tags,
        }


# ─── Katman 1: İkili Sayı Sistemi ───────────────────────────────────────────

def _binary_lessons() -> List[Lesson]:
    """İkili sayı sistemi, mantık ve veri temsili."""
    return [
        Lesson("binary", "İkili (binary) sayı sistemi nedir?",
               "İkili sayı sistemi yalnızca iki rakam kullanır: 0 ve 1. "
               "Bilgisayarlar neden ikili kullanır? Çünkü elektronik devreler "
               "iki durumu güvenilir biçimde ayırt eder: akım var (1) veya yok (0). "
               "Onluk sistemde her basamak 10'un kuvvetidir; ikili sistemde her "
               "basamak 2'nin kuvvetidir. 1101 ikili sayısı: "
               "1×8 + 1×4 + 0×2 + 1×1 = 13.",
               ["binary", "sayı sistemi", "temel"]),
        Lesson("binary", "İkiliden onluğa çevirme",
               "Sağdan sola her bit 2'nin artan kuvvetiyle çarpılır ve toplanır. "
               "10110 = 16 + 4 + 2 = 22. Tersine çevirme için sayı sürekli 2'ye "
               "bölünür ve kalanlar tersten yazılır: 22 → 11 kalan 0, 5 kalan 1, "
               "2 kalan 1, 1 kalan 0, 0 kalan 1 → 10110.",
               ["binary", "dönüşüm"], code=(
                   "def ikiliden_onluk(metin: str) -> int:\n"
                   "    \"\"\"'10110' -> 22\"\"\"\n"
                   "    return int(metin, 2)\n"
                   "\n"
                   "def onluktan_ikilik(sayi: int) -> str:\n"
                   "    \"\"\"22 -> '10110'\"\"\"\n"
                   "    return format(sayi, 'b')\n"
                   "\n"
                   "print(ikiliden_onluk('10110'))   # 22\n"
                   "print(onluktan_ikilik(22))       # 10110")),
        Lesson("binary", "Bit, bayt ve önekler",
               "Bit en küçük birimdir (0 veya 1). 8 bit = 1 bayt. Bir bayt "
               "0-255 arası 256 farklı değer taşır. Önekler: 1 KB = 1024 bayt, "
               "1 MB = 1024 KB, 1 GB = 1024 MB. 32 bit = 4 bayt, 64 bit = 8 bayt. "
               "Bir IPv4 adresi 32 bit, bir MAC adresi 48 bittir.",
               ["binary", "bit", "bayt"]),
        Lesson("binary", "Mantık kapıları (AND, OR, NOT, XOR)",
               "Mantık kapıları bitler üzerinde işlem yapar. AND: ikisi de 1 ise 1. "
               "OR: biri 1 ise 1. NOT: tersini alır. XOR: farklıysa 1, aynıysa 0. "
               "XOR'un özel özelliği: aynı değerle iki kez XOR'lamak sayıyı geri "
               "verir; bu yüzden şifrelemede ve hata denetiminde kullanılır.",
               ["binary", "mantık", "kapı"], code=(
                   "a, b = 0b1100, 0b1010\n"
                   "print(a & b, bin(a & b))   # AND  -> 8  0b1000\n"
                   "print(a | b, bin(a | b))   # OR   -> 14 0b1110\n"
                   "print(a ^ b, bin(a ^ b))   # XOR  -> 6  0b0110\n"
                   "print(~a & 0b1111)         # NOT  -> 3")),
        Lesson("binary", "İkili sayılarda toplama ve taşma",
               "İkili toplama onluk gibidir: 0+0=0, 1+0=1, 1+1=10 (elde 1). "
               "8 bitlik bir baytta 255 + 1 = 256 taşar; elde biti kaybolur ve "
               "sonuç 0 olur. Bu **taşma (overflow)** olayıdır ve yazılım "
               "hatalarının klasik kaynağıdır. İşaretli tam sayılarda en sol bit "
               "işareti belirtir (two's complement).",
               ["binary", "toplama", "taşma"]),
        Lesson("binary", "Metin kodlama: ASCII ve UTF-8",
               "Her karakter bir sayıya eşlenir. ASCII 7 bit kullanır: 'A' = 65, "
               "'a' = 97, '0' = 48. UTF-8 değişken uzunluktadır: ASCII karakterler "
               "1 bayt, Türkçe 'ş' 2 bayt, emoji 4 bayt. Türkçe metinlerde "
               "kodlama hatası **mojibake** üretir; bu yüzden dosyalar UTF-8 "
               "olarak açılmalıdır.",
               ["binary", "ascii", "utf-8", "kodlama"], code=(
                   "print(ord('A'), ord('a'))        # 65 97\n"
                   "print(chr(65))                   # A\n"
                   "print('ş'.encode('utf-8'))       # b'\\xc5\\x9f' (2 bayt)\n"
                   "print(len('ş'.encode('utf-8')))  # 2")),
        Lesson("binary", "Ses ve görüntü nasıl sayıya dönüşür?",
               "Analog bir sinyal (ses, ışık) belirli aralıklarla **örneklenir** "
               "(sampling) ve her örnek bir sayıya yuvarlanır (quantization). "
               "Ses: saniyede 44100 örnek, her örnek 16 bit. Görüntü: her piksel "
               "için kırmızı-yeşil-mavi parlaklığı, 8'er bit (toplam 24 bit). "
               "1920×1080 bir kare = 1920 × 1080 × 3 bayt ≈ 6 MB.",
               ["binary", "örnekleme", "ses", "görüntü"]),
        Lesson("binary", "İkili veride hata denetimi ve düzeltme",
               "Veri iletiminde bitler bozulabilir. **Eşlik biti (parity)** tek "
               "bit hatasını yakalar. **Sağlama toplamı (checksum)** ve "
               "**CRC** daha güçlüdür. **Hamming kodu** hatayı yakalamakla kalmaz, "
               "düzeltir. Modern sistemlerde SHA-256 gibi kriptografik özetler "
               "bütünlük denetimi için kullanılır.",
               ["binary", "hata", "crc", "sha"], code=(
                   "import hashlib\n"
                   "veri = b'merhaba'\n"
                   "print(hashlib.sha256(veri).hexdigest()[:16])\n"
                   "# Aynı veri -> aynı özet; tek bit değişse özet tamamen değişir")),
        Lesson("binary", "Onaltılık (hexadecimal) gösterim",
               "16 tabanlı sistem 0-9 ve A-F kullanır. Her hex basamağı tam 4 bite "
               "karşılık gelir; bu yüzden ikili veriyi kısa ve okunur biçimde "
               "gösterir. 0xFF = 255, 0x10 = 16. Renk kodları (#FF8800), bellek "
               "adresleri ve dosya imzaları hex ile yazılır.",
               ["binary", "hex", "gösterim"], code=(
                   "print(hex(255), bin(255), oct(255))   # 0xff 0b11111111 0o377\n"
                   "print(int('FF', 16))                  # 255")),
        Lesson("binary", "Program neden ikili koda derlenir?",
               "İşlemci yalnızca makine kodu anlar: opcode + operand'lardan oluşan "
               "sayılar. Derleyici yüksek seviyeli kodu bu sayılara çevirir. "
               "Toplama işlemi x86-64'te `add` opcode'una karşılık gelir. "
               "**Yorumlayıcı** (Python) kodu satır satır çalıştırır; **derleyici** "
               "(C, Rust) önce tamamen makine koduna çevirir. Bu yüzden derlenen "
               "kod genelde daha hızlıdır.",
               ["binary", "derleyici", "makine kodu"]),
    ]


# ─── Katman 2: Donanım ve Açılış ────────────────────────────────────────────

def _hardware_lessons() -> List[Lesson]:
    """CPU, bellek, veri yolu ve açılış süreci."""
    return [
        Lesson("hardware", "CPU nasıl çalışır?",
               "İşlemci üç aşamalı döngü çalıştırır: **getir (fetch)** → komutu "
               "bellekten al, **çöz (decode)** → hangi işlem olduğunu belirle, "
               "**yürüt (execute)** → işlemi yap. Saat hızı (GHz) saniyedeki "
               "döngü sayısıdır. Çekirdek (core) sayısı aynı anda kaç komut "
               "akışının yürütülebileceğini belirler.",
               ["hardware", "cpu", "fetch-decode-execute"]),
        Lesson("hardware", "Bellek hiyerarşisi: kayıtçı, önbellek, RAM, disk",
               "Hız ve boyut ters orantılıdır. **Kayıtçılar (registers)** en hızlı "
               "ve en küçük. **L1/L2/L3 önbellek** CPU içindedir. **RAM** daha "
               "büyük ama yavaş; elektrik kesilince silinir. **SSD/HDD** kalıcıdır "
               "ve en yavaştır. Tipik gecikmeler: kayıtçı 1 döngü, L1 ~4, RAM ~200, "
               "SSD ~100.000, disk ~10.000.000 döngü.",
               ["hardware", "bellek", "önbellek", "ram"]),
        Lesson("hardware", "Veri yolu (bus) ve çevre birimleri",
               "Veri yolu, bileşenler arasında veri taşıyan hatlardır. **Adres "
               "yolu** hangi bellek hücresine erişileceğini, **veri yolu** içeriği "
               "taşır. Modern sistemlerde PCIe ekran kartı ve SSD için, USB çevre "
               "birimleri için, SATA diskler için kullanılır. Kesme (interrupt) "
               "mekanizması, çevre biriminin CPU'yu anında haberdar etmesini sağlar.",
               ["hardware", "bus", "pcie", "kesme"]),
        Lesson("hardware", "Bilgisayar açılırken ne olur? (POST → BIOS → Bootloader)",
               "1. Güç verilir, CPU sıfırlanır ve **donanım adresindeki** ilk "
               "komutu okur. 2. **POST** (Power-On Self-Test) donanımı sınar. "
               "3. **BIOS/UEFI** firmware'ı çalışır; diskleri ve aygıtları tanır. "
               "4. **Önyükleyici (bootloader)** diskteki işletim sistemini yükler "
               "(GRUB, Windows Boot Manager). 5. **Çekirdek (kernel)** devreye girer "
               "ve sürücüleri başlatır. 6. Kullanıcı alanı süreçleri başlar.",
               ["hardware", "bios", "uefi", "boot"]),
        Lesson("hardware", "UEFI ile BIOS arasındaki fark",
               "BIOS 16 bit gerçek modda çalışır, en fazla 2 TB diskten önyükleme "
               "yapar ve MBR bölüm tablosu kullanır. UEFI 32/64 bit çalışır, GPT "
               "bölümlemesiyle 9 ZB'a kadar destekler, **Secure Boot** ile imzasız "
               "önyükleyicileri engeller ve daha hızlı açılır. Modern sistemlerde "
               "UEFI, eski uyumluluk için CSM katmanı sunar.",
               ["hardware", "uefi", "secure-boot", "gpt"]),
        Lesson("hardware", "GPU neden yapay zekâ için önemli?",
               "GPU binlerce küçük çekirdek içerir ve **aynı işlemi** çok sayıda "
               "veri üzerinde paralel yapar (SIMD). Yapay sinir ağlarının özü "
               "matris çarpımıdır; bu işlem paralelleştirilmeye çok uygundur. "
               "CPU ~16 çekirdekle sınırlıyken GPU binlerce iş parçacığı çalıştırır. "
               "VRAM, model ağırlıklarını tutar; yetmezse eğitim durur.",
               ["hardware", "gpu", "cuda", "paralel"]),
        Lesson("hardware", "Sanal bellek ve sayfalama",
               "İşletim sistemi her sürece **sanal adres alanı** verir. Adresler "
               "**sayfalara** (tipik 4 KB) bölünür ve **sayfa tablosu** ile fiziksel "
               "adrese çevrilir. RAM yetmezse sayfalar diske taşınır (swap/pagefile); "
               "buna **sayfa hatası (page fault)** denir. Bu mekanizma süreçleri "
               "birbirinden yalıtır: bir süreç diğerinin belleğini göremez.",
               ["hardware", "sanal bellek", "sayfalama", "mmu"]),
        Lesson("hardware", "Kaynak nezaketi: sistemi kilitlemeden çalışmak",
               "Ağır işler (eğitim, derleme, render) kaynağı doyurursa kullanıcı "
               "bilgisayarı kullanamaz. Doğru yaklaşım: **boştaki kaynağın %90'ını** "
               "kullanmak. Örnek: CPU %50 kullanılıyorsa %45'e kadar çık; sistem "
               "%90 boştaysa %81'ini kullan. VRAM için üst sınır koymak (ör. %85) "
               "ve dolduğunda işi duraklatmak gerekir.",
               ["hardware", "kaynak", "nezaket", "performans"], code=(
                   "import psutil\n"
                   "\n"
                   "def harcanabilir(kullanim: float) -> float:\n"
                   "    \"\"\"Boştaki kaynağın %90'ı.\"\"\"\n"
                   "    return max(0.0, (1.0 - kullanim) * 0.90)\n"
                   "\n"
                   "cpu = psutil.cpu_percent() / 100\n"
                   "print(f'CPU %{cpu*100:.0f} dolu -> %{harcanabilir(cpu)*100:.0f} kullan')")),
    ]


# ─── Katman 3: İşletim Sistemi ──────────────────────────────────────────────

def _os_lessons() -> List[Lesson]:
    """Çekirdek, süreç, bellek, dosya sistemi ve kabuk."""
    return [
        Lesson("os", "İşletim sistemi nedir, ne yapar?",
               "İşletim sistemi donanımla uygulamalar arasında aracıdır. Dört temel "
               "görevi vardır: **süreç yönetimi** (kim ne zaman çalışacak), "
               "**bellek yönetimi** (kim hangi belleği kullanacak), "
               "**dosya sistemi** (veri nasıl saklanacak) ve **aygıt yönetimi** "
               "(sürücüler). Ayrıca güvenlik ve kullanıcı yalıtımı sağlar.",
               ["os", "temel", "görevler"]),
        Lesson("os", "Çekirdek (kernel) ve kullanıcı alanı",
               "Kod iki ayrıcalık seviyesinde çalışır. **Çekirdek alanı** donanıma "
               "doğrudan erişir. **Kullanıcı alanı** kısıtlıdır; donanım için "
               "**sistem çağrısı (syscall)** yapar. `open`, `read`, `write`, `fork` "
               "birer sistem çağrısıdır. Bu ayrım bir sürecin diğerini veya "
               "çekirdeği bozmasını engeller.",
               ["os", "kernel", "syscall", "kullanıcı alanı"]),
        Lesson("os", "Süreç ve iş parçacığı (process vs thread)",
               "**Süreç** kendi bellek alanına sahiptir; yalıtılmıştır ama "
               "oluşturması pahalıdır. **İş parçacığı (thread)** aynı sürecin "
               "belleğini paylaşır; hafiftir ama veri yarışı riski taşır. "
               "Süreç durumları: hazır (ready), çalışıyor (running), bekliyor "
               "(waiting), sonlandı (terminated).",
               ["os", "süreç", "thread"], code=(
                   "import threading\n"
                   "\n"
                   "def isci(ad: str) -> None:\n"
                   "    print(f'{ad} calisiyor')\n"
                   "\n"
                   "threads = [threading.Thread(target=isci, args=(f't{i}',)) for i in range(3)]\n"
                   "for t in threads: t.start()\n"
                   "for t in threads: t.join()")),
        Lesson("os", "Zamanlama (scheduling) algoritmaları",
               "İşlemci aynı anda tek süreç çalıştırır; **zamanlayıcı** sırayı "
               "belirler. **Round Robin**: her sürece eşit dilim. **Öncelik "
               "(priority)**: önemli iş önce. **En kısa iş önce**: ortalama bekleme "
               "süresini düşürür. **Çok seviyeli kuyruk**: etkileşimli süreçler "
               "öne alınır. Bağlam değiştirme (context switch) maliyetlidir.",
               ["os", "zamanlama", "round-robin"]),
        Lesson("os", "Kilitlenme (deadlock) ve önlenmesi",
               "İki süreç birbirinin tuttuğu kaynağı beklerse kilitlenme olur. "
               "Dört koşul birlikte gerçekleşmelidir: karşılıklı dışlama, tut-ve-"
               "bekle, öncelenemezlik, döngüsel bekleme. Çözümler: kaynakları "
               "**sıralı** edinmek, **zaman aşımı** koymak, tek seferde tüm "
               "kaynağı istemek.",
               ["os", "deadlock", "kilit"]),
        Lesson("os", "Dosya sistemi nasıl çalışır?",
               "Dosya sistemi veriyi **bloklara** böler ve bir **dizin (inode/MFT)** "
               "yapısıyla izler. NTFS (Windows), ext4 (Linux), APFS (macOS) yaygındır. "
               "Yol kavramları: kök dizin, mutlak yol (`C:\\\\proje\\\\a.py`), göreli "
               "yol (`a.py`), `..` üst dizin. İzinler okuma/yazma/çalıştırma olarak "
               "kullanıcı-grup-diğerleri için ayrı verilir.",
               ["os", "dosya sistemi", "ntfs", "izinler"], code=(
                   "from pathlib import Path\n"
                   "\n"
                   "kok = Path('.')\n"
                   "for item in sorted(kok.iterdir())[:5]:\n"
                   "    tur = 'dizin' if item.is_dir() else f'{item.stat().st_size} bayt'\n"
                   "    print(f'{item.name:28} {tur}')")),
        Lesson("os", "Kabuk (shell): komutları kim yorumlar?",
               "Kabuk kullanıcının yazdığı metni sistem çağrılarına çevirir. "
               "**PowerShell** nesne döndürür, **cmd** metin, **bash** metin. "
               "Boru (`|`) bir komutun çıktısını diğerine verir; yönlendirme (`>`) "
               "dosyaya yazar. Kabuk ayrıca ortam değişkenleri, betikler ve "
               "iş kontrolü yönetir.",
               ["os", "kabuk", "powershell", "bash"], code=(
                   "# PowerShell\n"
                   "Get-ChildItem -Filter *.py | Select-Object -First 5 Name, Length\n"
                   "\n"
                   "# Aynı işi Python'da yapmak (daha taşınabilir):\n"
                   "from pathlib import Path\n"
                   "for p in sorted(Path('.').glob('*.py'))[:5]:\n"
                   "    print(p.name, p.stat().st_size)")),
        Lesson("os", "Süreçler arası iletişim (IPC)",
               "Süreçler yalıtılmış olduğu için veri paylaşmak için mekanizma "
               "gerekir: **boru (pipe)**, **paylaşımlı bellek**, **mesaj kuyruğu**, "
               "**soket**, **sinyal**. Soketler ağ üzerinden de çalışır; bu yüzden "
               "istemci-sunucu mimarisinin temelidir.",
               ["os", "ipc", "pipe", "socket"]),
        Lesson("os", "Kullanıcı, izin ve güvenlik",
               "Her işlem bir kullanıcı adına çalışır. Windows'ta **UAC** yönetici "
               "yetkisini onaylatır; Linux'ta `sudo` geçici yükseltme yapar. "
               "**Sandbox** (kum havuzu) bir sürecin erişebileceği dosya ve ağ "
               "kaynaklarını sınırlar. İyi yazılım en az yetki ilkesiyle çalışır: "
               "gerekmeyen izni istemez.",
               ["os", "güvenlik", "izin", "sandbox"]),
        Lesson("os", "Sürücü (driver) nedir?",
               "Sürücü, donanımı işletim sistemine tanıtan koddur. Üretici "
               "donanımın kayıtçılarına nasıl yazılacağını sürücüde bildirir. "
               "Çekirdek modu sürücüler hızlıdır ama hatası sistemi çökertir; "
               "kullanıcı modu sürücüler daha güvenlidir (Windows'ta UMDF).",
               ["os", "sürücü", "donanım"]),
    ]


# ─── Katman 4: Programlama Dilleri ──────────────────────────────────────────

def _language_lessons() -> List[Lesson]:
    """Dil kavramları ve Python."""
    return [
        Lesson("languages", "Programlama dili nedir, nasıl çalışır?",
               "Dil, insanın niyetini makinenin yapabileceği işlemlere çeviren "
               "kurallar bütünüdür. **Derlenen** diller (C, Rust, Go) önce makine "
               "koduna çevrilir. **Yorumlanan** diller (Python, JavaScript) satır "
               "satır yürütülür. **Sanal makine** (Java, C#) ara bir koda derlenir. "
               "Değişkenler, tipler, operatörler, akış kontrolü, fonksiyonlar her "
               "dilde bulunur.",
               ["dil", "derleme", "yorumlama"]),
        Lesson("languages", "Veri tipleri ve değişkenler",
               "Tip, değerin nasıl saklandığını ve hangi işlemlerin geçerli "
               "olduğunu belirler. Temel tipler: tam sayı (int), ondalık (float), "
               "mantıksal (bool), metin (str). Python'da tip **dinamiktir** (çalışma "
               "anında belirlenir), C'de **statiktir** (derleme anında). Tip "
               "dönüşümü açık (`int('5')`) veya örtük (`1 + 2.0`) olabilir.",
               ["dil", "tip", "değişken"], code=(
                   "sayi: int = 42\n"
                   "pi: float = 3.14\n"
                   "aktif: bool = True\n"
                   "ad: str = 'NeuroCore'\n"
                   "print(type(sayi).__name__, type(pi).__name__)\n"
                   "print(f'{ad} - {sayi} - {pi}')")),
        Lesson("languages", "Koşullar ve döngüler",
               "`if/elif/else` dallanma sağlar. `for` bilinen sayıda, `while` koşul "
               "sürdükçe döner. `break` döngüyü bitirir, `continue` sonraki tura "
               "geçer. Sonsuz döngüden kaçınmak için sayaç veya zaman sınırı "
               "koymak iyi alışkanlıktır.",
               ["dil", "koşul", "döngü"], code=(
                   "toplam = 0\n"
                   "for i in range(1, 6):\n"
                   "    if i % 2 == 0:\n"
                   "        continue\n"
                   "    toplam += i\n"
                   "print(f'tek sayilarin toplami: {toplam}')   # 9")),
        Lesson("languages", "Fonksiyonlar, parametreler ve dönüş değeri",
               "Fonksiyon tekrarı önler ve kodu adlandırır. Parametreler konumlu "
               "veya anahtarlı verilebilir. Varsayılan değer, `*args` (sınırsız "
               "konumlu), `**kwargs` (sınırsız anahtarlı) desteklenir. Fonksiyon "
               "**tek iş** yapmalıdır; bu test edilebilirliği artırır.",
               ["dil", "fonksiyon"], code=(
                   "def ortalama(*sayilar: float, yuvarla: int = 2) -> float:\n"
                   "    \"\"\"Sayıların ortalamasını verir.\"\"\"\n"
                   "    if not sayilar:\n"
                   "        return 0.0\n"
                   "    return round(sum(sayilar) / len(sayilar), yuvarla)\n"
                   "\n"
                   "print(ortalama(3, 4, 5))        # 4.0\n"
                   "print(ortalama(1, 2, yuvarla=0)) # 2.0")),
        Lesson("languages", "Veri yapıları: liste, sözlük, küme, demet",
               "**Liste** sıralı ve değiştirilebilir. **Demet (tuple)** sıralı ve "
               "değiştirilemez. **Sözlük (dict)** anahtar-değer eşler, arama O(1). "
               "**Küme (set)** tekrarsız eleman tutar, kesişim/birleşim hızlıdır. "
               "Doğru yapıyı seçmek algoritmayı belirgin biçimde hızlandırır.",
               ["dil", "veri yapısı"], code=(
                   "kayitlar = {'a': 1, 'b': 2}\n"
                   "kayitlar['c'] = 3\n"
                   "print(kayitlar.get('yok', 0))          # 0\n"
                   "print({1, 2, 3} & {2, 3, 4})           # {2, 3}\n"
                   "print(sorted(kayitlar.items()))")),
        Lesson("languages", "Hata yönetimi: try/except/else/finally",
               "Hata **yakalanmalı**, yutulmamalıdır. `try` riskli kodu, `except` "
               "belirli hatayı, `else` hata olmazsa çalışacak kodu, `finally` her "
               "durumda çalışacak temizliği barındırır. Çok geniş `except "
               "Exception` kullanmak gerçek hatayı gizler; özel istisna sınıfları "
               "tanımlamak daha iyidir.",
               ["dil", "hata", "istisna"], code=(
                   "class VeriHatasi(Exception):\n"
                   "    \"\"\"Alan hatası.\"\"\"\n"
                   "\n"
                   "def oku(yol: str) -> str:\n"
                   "    try:\n"
                   "        with open(yol, encoding='utf-8') as f:\n"
                   "            return f.read()\n"
                   "    except FileNotFoundError as exc:\n"
                   "        raise VeriHatasi(f'dosya yok: {yol}') from exc")),
        Lesson("languages", "Nesne yönelimli programlama (sınıf, kalıtım)",
               "**Sınıf** veriyi (alan) ve davranışı (metot) bir arada tutar. "
               "**Kalıtım** ortak davranışı paylaşır. **Kapsülleme** iç durumu "
               "gizler. **Çok biçimlilik** aynı arayüzün farklı gerçekleşmelerine "
               "izin verir. Kompozisyon (içerme) çoğu zaman kalıtımdan daha "
               "esnektir.",
               ["dil", "oop", "sınıf"], code=(
                   "class Sekil:\n"
                   "    def alan(self) -> float:\n"
                   "        raise NotImplementedError\n"
                   "\n"
                   "class Daire(Sekil):\n"
                   "    def __init__(self, r: float) -> None:\n"
                   "        self.r = r\n"
                   "    def alan(self) -> float:\n"
                   "        return 3.14159 * self.r ** 2\n"
                   "\n"
                   "print(f'{Daire(2).alan():.2f}')")),
        Lesson("languages", "Algoritma karmaşıklığı (Big-O)",
               "Karmaşıklık, girdi büyüdükçe işin nasıl arttığını gösterir. "
               "O(1) sabit, O(log n) logaritmik (ikili arama), O(n) doğrusal, "
               "O(n log n) sıralama, O(n²) iç içe döngü, O(2ⁿ) üstel. Aynı işi "
               "yapan iki kod arasındaki fark, büyük veride dramatik olur.",
               ["dil", "algoritma", "big-o"], code=(
                   "def ikili_arama(sirali: list, hedef: int) -> int:\n"
                   "    \"\"\"O(log n) arama.\"\"\"\n"
                   "    alt, ust = 0, len(sirali) - 1\n"
                   "    while alt <= ust:\n"
                   "        orta = (alt + ust) // 2\n"
                   "        if sirali[orta] == hedef:\n"
                   "            return orta\n"
                   "        if sirali[orta] < hedef:\n"
                   "            alt = orta + 1\n"
                   "        else:\n"
                   "            ust = orta - 1\n"
                   "    return -1")),
        Lesson("languages", "Özyineleme (recursion) ve yığın",
               "Fonksiyonun kendini çağırmasıdır. Her çağrı **çağrı yığınında** "
               "yer kaplar; derinlik sınırı aşılırsa `RecursionError` olur. "
               "**Taban durumu** şarttır, yoksa sonsuz döngü olur. Bazı problemler "
               "(ağaç gezme, böl-ve-yönet) özyineleme ile doğal biçimde çözülür.",
               ["dil", "özyineleme", "yığın"], code=(
                   "def faktoriyel(n: int) -> int:\n"
                   "    if n <= 1:          # taban durumu\n"
                   "        return 1\n"
                   "    return n * faktoriyel(n - 1)\n"
                   "\n"
                   "print(faktoriyel(5))   # 120")),
        Lesson("languages", "Test yazmak neden zorunlu?",
               "Test, kodun beklenen davranışını kilitler ve değişiklikte "
               "bozulmayı yakalar. **Birim testi** tek fonksiyonu, **entegrasyon "
               "testi** bileşenlerin birlikte çalışmasını sınar. İyi test "
               "**belirleyicidir** (aynı girdi → aynı sonuç) ve dış dünyaya "
               "bağımlı değildir.",
               ["dil", "test", "kalite"], code=(
                   "import unittest\n"
                   "\n"
                   "def topla(a: int, b: int) -> int:\n"
                   "    return a + b\n"
                   "\n"
                   "class TestTopla(unittest.TestCase):\n"
                   "    def test_pozitif(self) -> None:\n"
                   "        self.assertEqual(topla(2, 3), 5)\n"
                   "    def test_negatif(self) -> None:\n"
                   "        self.assertEqual(topla(-2, 2), 0)\n"
                   "\n"
                   "# python -m unittest  ile çalıştırılır")),
    ]


# ─── Katman 5: Kütüphaneler ─────────────────────────────────────────────────

def _library_lessons() -> List[Lesson]:
    """Standart kütüphane ve yaygın paketler."""
    return [
        Lesson("libraries", "Kütüphane nedir, neden kullanılır?",
               "Kütüphane, başkalarının yazdığı ve test ettiği hazır koddur. "
               "Tekerleği yeniden icat etmemek için kullanılır. **Standart "
               "kütüphane** dille birlikte gelir (Python'da `os`, `json`, `math`). "
               "**Üçüncü taraf** paketler `pip` ile kurulur. Her bağımlılık bir "
               "risktir: güncelleme, güvenlik, uyumluluk. Bu yüzden sayıları "
               "ölçülü tutulur.",
               ["kütüphane", "pip", "bağımlılık"]),
        Lesson("libraries", "Dosya ve yol işlemleri: pathlib, os, shutil",
               "`pathlib` yolları nesne olarak yönetir ve platformdan bağımsızdır. "
               "`os` ortam değişkenleri ve süreç işlemleri sağlar. `shutil` kopyalama "
               "ve taşıma yapar. Yol birleştirmek için `+` yerine `/` operatörü "
               "kullanmak Windows'ta ayırıcı hatasını önler.",
               ["kütüphane", "pathlib", "dosya"], code=(
                   "from pathlib import Path\n"
                   "import os, shutil\n"
                   "\n"
                   "kok = Path(os.environ.get('TEMP', '.')) / 'ornek'\n"
                   "kok.mkdir(exist_ok=True)\n"
                   "(kok / 'a.txt').write_text('merhaba', encoding='utf-8')\n"
                   "print((kok / 'a.txt').read_text(encoding='utf-8'))\n"
                   "shutil.rmtree(kok, ignore_errors=True)")),
        Lesson("libraries", "JSON ve veri serileştirme",
               "**Serileştirme** veriyi saklanabilir/aktarılabilir biçime çevirir. "
               "JSON evrensel ve okunurdur ama tip bilgisi taşımaz (tarih metne "
               "döner). `pickle` Python'a özgüdür ve güvenlik riski taşır (çalışan "
               "kod içerebilir). `msgpack` hızlı ve küçüktür.",
               ["kütüphane", "json", "serileştirme"], code=(
                   "import json\n"
                   "\n"
                   "veri = {'ad': 'NeuroCore', 'surum': 1, 'etiketler': ['yerel', 'ajan']}\n"
                   "metin = json.dumps(veri, ensure_ascii=False, indent=2)\n"
                   "print(metin)\n"
                   "print(json.loads(metin)['etiketler'][0])")),
        Lesson("libraries", "numpy ile hızlı sayısal işlem",
               "Python listeleri yavaştır çünkü her eleman ayrı bir nesnedir. "
               "`numpy` veriyi **bitişik bellekte** tutar ve işlemleri C dilinde "
               "yapar. **Vektörleştirme** (döngüsüz işlem) 10-100 kat hız sağlar. "
               "Yapay zekânın temeli matris çarpımıdır; numpy bunu donanım "
               "hızlandırmasıyla yapar.",
               ["kütüphane", "numpy", "vektörleştirme"], code=(
                   "import numpy as np\n"
                   "\n"
                   "a = np.arange(1_000_000)\n"
                   "print(a.sum())                       # döngüsüz toplam\n"
                   "m = np.random.rand(3, 4)\n"
                   "print(m.shape, np.round(m.mean(), 3))")),
        Lesson("libraries", "torch (PyTorch) ile tensör ve otomatik türev",
               "`torch` tensörler (GPU'da çalışabilen çok boyutlu diziler) ve "
               "**otomatik türev** sağlar. `requires_grad=True` işaretlenen tensör "
               "için hesaplanan işlem grafiği tutulur; `backward()` gradyanları "
               "verir. Eğitim döngüsü: ileri geçiş → kayıp → `backward()` → "
               "optimizasyon adımı.",
               ["kütüphane", "torch", "tensör", "gradyan"], code=(
                   "import torch\n"
                   "\n"
                   "x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
                   "y = (x ** 2).sum()\n"
                   "y.backward()\n"
                   "print(x.grad)          # dy/dx = 2x -> [2, 4, 6]\n"
                   "print(torch.cuda.is_available())")),
        Lesson("libraries", "asyncio ile eşzamanlı G/Ç",
               "G/Ç beklerken işlemci boş kalır. `asyncio` tek iş parçacığında "
               "**bekleme sırasında** başka göreve geçer. `async def` ile tanımlanan "
               "fonksiyon `await` ile beklenir. `asyncio.gather` görevleri paralel "
               "başlatır. Bu model CPU yoğun işler için değil, ağ/dosya bekleyen "
               "işler içindir.",
               ["kütüphane", "asyncio", "eşzamanlılık"], code=(
                   "import asyncio\n"
                   "\n"
                   "async def indir(ad: str, saniye: float) -> str:\n"
                   "    await asyncio.sleep(saniye)\n"
                   "    return f'{ad} bitti'\n"
                   "\n"
                   "async def main() -> None:\n"
                   "    sonuc = await asyncio.gather(indir('a', 0.2), indir('b', 0.1))\n"
                   "    print(sonuc)\n"
                   "\n"
                   "asyncio.run(main())")),
        Lesson("libraries", "Sözlü işlemler: requests, aiohttp, urllib",
               "Ağ isteği için önce **standart kütüphane** (`urllib`) yeterlidir. "
               "`requests` okunur ama ek bağımlılıktır. Eşzamanlı indirme için "
               "`aiohttp`. Her istekte **zaman aşımı** ve **User-Agent** başlığı "
               "vermek gerekir; bazı sunucular boş kimliği reddeder (HTTP 403).",
               ["kütüphane", "http", "urllib", "requests"], code=(
                   "import urllib.request, json\n"
                   "\n"
                   "istek = urllib.request.Request('https://example.com')\n"
                   "istek.add_header('User-Agent', 'NeuroCore/0.1')\n"
                   "with urllib.request.urlopen(istek, timeout=15) as yanit:\n"
                   "    print(yanit.status, len(yanit.read()))")),
        Lesson("libraries", "Görüntü işleme: Pillow",
               "`Pillow` (PIL) görüntüyü açar, dönüştürür ve kaydeder. Pikseller "
               "numpy dizisi olarak okunabilir; bu yüzden görüntü işleme ile "
               "yapay zekâ arasında köprü kurar. Ekran görüntüsü alıp karşılaştırmak "
               "(görsel doğrulama) için de kullanılır.",
               ["kütüphane", "pillow", "görüntü"], code=(
                   "from PIL import Image\n"
                   "\n"
                   "img = Image.new('RGB', (64, 64), (30, 30, 40))\n"
                   "img.putpixel((10, 10), (255, 200, 0))\n"
                   "img.save('ornek.png')\n"
                   "print(Image.open('ornek.png').size)")),
        Lesson("libraries", "Sistem izleme: psutil",
               "`psutil` CPU, bellek, disk ve süreç bilgilerini okur. Kaynak "
               "nezaketi politikası bu veriye dayanır: boştaki kaynağın %90'ı "
               "hesaplanır, sistem dolarsa ağır iş duraklatılır.",
               ["kütüphane", "psutil", "izleme"], code=(
                   "import psutil, shutil\n"
                   "\n"
                   "print(f'CPU: %{psutil.cpu_percent(interval=0.5):.0f}')\n"
                   "print(f'RAM: %{psutil.virtual_memory().percent:.0f}')\n"
                   "disk = shutil.disk_usage('.')\n"
                   "print(f'Disk bos: {disk.free / 1024**3:.1f} GB')")),
        Lesson("libraries", "Web çatısı: FastAPI ve uvicorn",
               "**FastAPI** tip ipuçlarından otomatik doğrulama ve dokümantasyon "
               "üretir. `uvicorn` ASGI sunucusudur. `async def` uç noktaları "
               "eşzamanlı istekleri karşılar. Bloklayan iş (ağır hesap) doğrudan "
               "çağrılırsa sunucu donar; ayrı iş parçacığına alınmalıdır.",
               ["kütüphane", "fastapi", "uvicorn", "web"], code=(
                   "from fastapi import FastAPI\n"
                   "\n"
                   "app = FastAPI()\n"
                   "\n"
                   "@app.get('/saglik')\n"
                   "async def saglik() -> dict:\n"
                   "    return {'ok': True}\n"
                   "\n"
                   "# uvicorn main:app --reload  ile çalıştırılır")),
    ]


# ─── Katman 6: Blender ──────────────────────────────────────────────────────

def _blender_lessons() -> List[Lesson]:
    """bpy ile 3B nesne oluşturma."""
    return [
        Lesson("blender", "Blender'ı Python ile yönetmek: bpy modülü",
               "Blender içindeki her şey Python'dan erişilebilir: sahne, nesne, "
               "malzeme, render. `bpy` modülü yalnızca Blender'ın kendi Python "
               "yorumlayıcısında bulunur; normal Python'da içe aktarılamaz. Betik "
               "iki biçimde çalıştırılır: `blender --background --python betik.py` "
               "(başsız) veya Blender içindeki Scripting sekmesinden.",
               ["blender", "bpy", "temel"]),
        Lesson("blender", "Sahneyi temizlemek ve nesne eklemek",
               "Yeni işe başlarken varsayılan küp, ışık ve kamera temizlenir. "
               "Nesne eklemek için `bpy.ops.mesh.primitive_*_add()` kullanılır. "
               "Eklenen nesne `bpy.context.active_object` olur. Konum ve ölçek "
               "`obj.location` ve `obj.scale` ile ayarlanır.",
               ["blender", "sahne", "nesne"], code=(
                   "import bpy\n"
                   "\n"
                   "# Sahneyi temizle\n"
                   "bpy.ops.object.select_all(action='SELECT')\n"
                   "bpy.ops.object.delete(use_global=False)\n"
                   "\n"
                   "# Küp ekle ve konumlandır\n"
                   "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))\n"
                   "kup = bpy.context.active_object\n"
                   "kup.name = 'BenimKup'\n"
                   "kup.scale = (1.5, 1.0, 1.0)\n"
                   "print(kup.name, tuple(kup.location))")),
        Lesson("blender", "Temel şekiller ve parametreleri",
               "Blender'ın hazır ilkel şekilleri vardır: küp (`primitive_cube_add`), "
               "küre (`primitive_uv_sphere_add`, `segments`/`ring_count`), silindir "
               "(`primitive_cylinder_add`, `vertices`/`radius`/`depth`), koni, "
               "torus (`major_radius`/`minor_radius`), düzlem. Parametreleri doğru "
               "vermek, sonradan düzenlemekten daha temizdir.",
               ["blender", "ilkel şekiller"], code=(
                   "import bpy\n"
                   "\n"
                   "bpy.ops.object.select_all(action='SELECT')\n"
                   "bpy.ops.object.delete(use_global=False)\n"
                   "\n"
                   "bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16,\n"
                   "                                     location=(-3, 0, 1))\n"
                   "bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.8, depth=2,\n"
                   "                                    location=(0, 0, 1))\n"
                   "bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1, depth=2,\n"
                   "                                location=(3, 0, 1))\n"
                   "print('3 nesne eklendi:', [o.name for o in bpy.data.objects])")),
        Lesson("blender", "Nesneyi taşıma, döndürme, ölçekleme",
               "Dönüşümler `location` (konum), `rotation_euler` (radyan cinsinden "
               "açı) ve `scale` (ölçek) ile yapılır. Dereceyi radyana çevirmek için "
               "`math.radians` kullanılır. Ölçeklemeyi uygulamak (`apply`) için "
               "`bpy.ops.object.transform_apply` çağrılır; aksi hâlde dışa "
               "aktarımda beklenmeyen sonuçlar çıkar.",
               ["blender", "dönüşüm"], code=(
                   "import bpy, math\n"
                   "\n"
                   "bpy.ops.mesh.primitive_cube_add(size=1)\n"
                   "kup = bpy.context.active_object\n"
                   "kup.location = (2, 0, 0.5)\n"
                   "kup.rotation_euler = (0, 0, math.radians(45))\n"
                   "kup.scale = (2, 1, 1)\n"
                   "\n"
                   "bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)\n"
                   "print('dönüşüm uygulandı')")),
        Lesson("blender", "Malzeme oluşturma ve renklendirme",
               "Malzeme `bpy.data.materials.new()` ile üretilir ve nesnenin "
               "`data.materials` listesine eklenir. **Principled BSDF** düğümü "
               "temel yüzey özelliklerini taşır: `Base Color`, `Metallic`, "
               "`Roughness`, `Emission`. Renkler RGBA ve **doğrusal** uzayda "
               "verilir (0-1 arası).",
               ["blender", "malzeme", "renk"], code=(
                   "import bpy\n"
                   "\n"
                   "bpy.ops.mesh.primitive_uv_sphere_add(radius=1)\n"
                   "nesne = bpy.context.active_object\n"
                   "\n"
                   "malzeme = bpy.data.materials.new('KirmiziParlak')\n"
                   "malzeme.use_nodes = True\n"
                   "bsdf = malzeme.node_tree.nodes['Principled BSDF']\n"
                   "bsdf.inputs['Base Color'].default_value = (0.9, 0.15, 0.1, 1.0)\n"
                   "bsdf.inputs['Roughness'].default_value = 0.35\n"
                   "bsdf.inputs['Metallic'].default_value = 0.1\n"
                   "\n"
                   "nesne.data.materials.append(malzeme)")),
        Lesson("blender", "Işık ve kamera yerleştirme",
               "Sahne ışık ve kamera olmadan render edilemez. Işık türleri: "
               "`POINT`, `SUN`, `AREA`, `SPOT`. Kamera `bpy.ops.object.camera_add` "
               "ile eklenir ve `bpy.context.scene.camera` olarak atanır. Kamera "
               "hedefe yönlendirmek için bir `Track To` kısıtı kullanılabilir.",
               ["blender", "ışık", "kamera"], code=(
                   "import bpy, math\n"
                   "\n"
                   "bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))\n"
                   "gunes = bpy.context.active_object\n"
                   "gunes.data.energy = 3.0\n"
                   "\n"
                   "bpy.ops.object.camera_add(location=(7, -7, 5),\n"
                   "                          rotation=(math.radians(63), 0, math.radians(45)))\n"
                   "bpy.context.scene.camera = bpy.context.active_object\n"
                   "print('ışık ve kamera hazır')")),
        Lesson("blender", "Başsız (headless) render almak",
               "`--background` ile arayüz açılmadan render alınır. Motor seçimi "
               "önemlidir: **Cycles** gerçekçi ama yavaş, **EEVEE** hızlı. Sunucuda "
               "GPU yoksa **Cycles CPU** güvenli seçimdir. Örnek sayısı (`samples`) "
               "düşürülerek önizleme hızlandırılır. Render ayarları "
               "`scene.render` altındadır.",
               ["blender", "render", "cycles", "eevee"], code=(
                   "import bpy\n"
                   "\n"
                   "sahne = bpy.context.scene\n"
                   "sahne.render.engine = 'CYCLES'\n"
                   "sahne.cycles.device = 'CPU'          # GPU yoksa güvenli\n"
                   "sahne.cycles.samples = 16            # hızlı önizleme\n"
                   "sahne.render.resolution_x = 320\n"
                   "sahne.render.resolution_y = 240\n"
                   "sahne.render.filepath = 'cikti.png'\n"
                   "sahne.render.image_settings.file_format = 'PNG'\n"
                   "bpy.ops.render.render(write_still=True)\n"
                   "print('render bitti')")),
        Lesson("blender", "Modeli dışa aktarma: glTF ve FBX",
               "Oyun motorları ve diğer araçlar için model dışa aktarılır. "
               "**glTF (.glb)** web ve Godot için idealdir. **FBX** Unity/Unreal "
               "için yaygındır. Dışa aktarmadan önce **dönüşümleri uygulamak** ve "
               "gereksiz nesneleri silmek gerekir. `use_selection=True` ile yalnızca "
               "seçilenler aktarılır.",
               ["blender", "gltf", "fbx", "dışa aktarma"], code=(
                   "import bpy\n"
                   "\n"
                   "bpy.ops.object.select_all(action='SELECT')\n"
                   "bpy.ops.export_scene.gltf(filepath='model.glb',\n"
                   "                          export_format='GLB',\n"
                   "                          use_selection=True)\n"
                   "print('glb yazıldı')")),
        Lesson("blender", "Blender betiğinde sık yapılan hatalar",
               "1. `bpy` modülünü normal Python'da içe aktarmaya çalışmak (yalnızca "
               "Blender içinde vardır). 2. `bpy.ops` çağrılarının **bağlama** "
               "(context) bağımlı olması; nesne aktif değilse işlem sessizce "
               "başarısız olur. 3. `BLENDER_EEVEE_NEXT` gibi **artık var olmayan** "
               "motor adları kullanmak. 4. Betik istisna fırlattığında Blender'ın "
               "bunu **yutması**; `--python-exit-code 1` verilmezse hata "
               "görünmez. 5. Uzun işlerde zaman aşımı koymamak.",
               ["blender", "hata", "ipucu"], code=(
                   "# Doğru çağrı biçimi (hata görünür olsun):\n"
                   "# blender --background --python-exit-code 1 --python betik.py\n"
                   "\n"
                   "# Betik içinde bağlamı doğrula:\n"
                   "import bpy\n"
                   "nesne = bpy.context.active_object\n"
                   "if nesne is None:\n"
                   "    raise RuntimeError('aktif nesne yok — önce nesne ekleyin')\n"
                   "print('aktif nesne:', nesne.name)")),
        Lesson("blender", "Örnek: 3 nesneli kompozisyon oluştur",
               "Tam bir sahne kurulumu: temizlik → zemin → nesneler → malzemeler → "
               "ışık → kamera → render. Bu sıra her projede tekrarlanır; betiği "
               "fonksiyonlara bölmek tekrar kullanımı kolaylaştırır.",
               ["blender", "örnek", "kompozisyon"], code=(
                   "import bpy, math\n"
                   "\n"
                   "def temizle() -> None:\n"
                   "    bpy.ops.object.select_all(action='SELECT')\n"
                   "    bpy.ops.object.delete(use_global=False)\n"
                   "\n"
                   "def malzeme(ad: str, renk: tuple, pürüzlülük: float = 0.4):\n"
                   "    m = bpy.data.materials.new(ad)\n"
                   "    m.use_nodes = True\n"
                   "    b = m.node_tree.nodes['Principled BSDF']\n"
                   "    b.inputs['Base Color'].default_value = (*renk, 1.0)\n"
                   "    b.inputs['Roughness'].default_value = pürüzlülük\n"
                   "    return m\n"
                   "\n"
                   "temizle()\n"
                   "\n"
                   "# Zemin\n"
                   "bpy.ops.mesh.primitive_plane_add(size=20)\n"
                   "zemin = bpy.context.active_object\n"
                   "zemin.data.materials.append(malzeme('Zemin', (0.2, 0.2, 0.24), 0.8))\n"
                   "\n"
                   "# Üç nesne\n"
                   "for i, (op, renk) in enumerate([\n"
                   "        (bpy.ops.mesh.primitive_cube_add, (0.85, 0.2, 0.2)),\n"
                   "        (bpy.ops.mesh.primitive_uv_sphere_add, (0.2, 0.6, 0.9)),\n"
                   "        (bpy.ops.mesh.primitive_cylinder_add, (0.3, 0.8, 0.35))]):\n"
                   "    op(location=(i * 3 - 3, 0, 1))\n"
                   "    n = bpy.context.active_object\n"
                   "    n.data.materials.append(malzeme(f'Malzeme{i}', renk))\n"
                   "\n"
                   "# Işık + kamera\n"
                   "bpy.ops.object.light_add(type='SUN', location=(6, -6, 12))\n"
                   "bpy.context.active_object.data.energy = 4.0\n"
                   "bpy.ops.object.camera_add(location=(10, -10, 7),\n"
                   "    rotation=(math.radians(62), 0, math.radians(45)))\n"
                   "bpy.context.scene.camera = bpy.context.active_object\n"
                   "print('kompozisyon hazır:', len(bpy.data.objects), 'nesne')")),
    ]


# ─── Müfredat ───────────────────────────────────────────────────────────────

def build_curriculum(layers: Optional[Sequence[str]] = None) -> List[Lesson]:
    """
    Müfredatı üretir.

    Args:
        layers: Yalnızca bu katmanlar (None = hepsi, sıralı)

    Returns:
        Ders listesi (katman sırasına göre)
    """
    builders = {
        "binary": _binary_lessons,
        "hardware": _hardware_lessons,
        "os": _os_lessons,
        "languages": _language_lessons,
        "libraries": _library_lessons,
        "blender": _blender_lessons,
    }
    wanted = list(layers) if layers else list(ALL_LAYERS)
    lessons: List[Lesson] = []
    for name in wanted:
        # Ağ katmanları ayrı modülden gelir (döngüsel import olmasın diye
        # burada, çağrı anında içe aktarılır)
        if name in NETWORK_LAYER_NAMES:
            from neurocore.neural.network import build_network_curriculum

            lessons.extend(build_network_curriculum([name]))
            continue
        if name in LOWPOLY_LAYER_NAMES:
            from neurocore.neural.lowpoly import build_lowpoly_curriculum

            lessons.extend(build_lowpoly_curriculum([name]))
            continue
        if name in GODOT_LAYER_NAMES:
            from neurocore.neural.godot_learn import build_godot_curriculum

            lessons.extend(build_godot_curriculum([name]))
            continue
        builder = builders.get(name)
        if builder is not None:
            lessons.extend(builder())
    return lessons


# ─── Müfredat Genişletme ────────────────────────────────────────────────────

#: Farklı soru kalıpları — aynı içerik farklı yollarla sorulur.
#:
#: Neden? Model bir kavramı **tek kalıpla** öğrenirse, gerçek kullanımda farklı
#: sorulduğunda tanıyamaz. Her ders birçok farklı kalıpla sunulur; bu hem veri
#: kümesini büyütür hem genellemeyi artırır.
#:
#: ⚠ Kalıplar **biçim** değiştirir, **içerik** uydurmaz: hedef metin aynı
#: derstir. Böylece yapay veri, gerçek bilgiyi seyreltmez.
QUESTION_PATTERNS: Sequence[tuple] = (
    ("tanım", "{title} nedir?"),
    ("nasıl", "{title} nasıl çalışır?"),
    ("neden", "{title} neden önemlidir?"),
    ("uygulama", "{title} — nasıl uygulanır?"),
    ("örnek", "{title} — örnek göster"),
    ("özet", "{title} konusunu özetle"),
    ("hata", "{title} konusunda sık yapılan hatalar nelerdir?"),
    ("karşılaştır", "{title} — alternatifleriyle karşılaştır"),
    ("başlangıç", "{title} — sıfırdan nasıl başlarım?"),
    ("ileri", "{title} — ileri düzey ayrıntıları nelerdir?"),
    ("tuzak", "{title} öğrenirken hangi tuzaklara düşülür?"),
    ("kod", "{title} için kod örneği yaz"),
    ("gerçek", "{title} gerçek bir projede nasıl kullanılır?"),
    ("performans", "{title} performansı nasıl etkiler?"),
    ("güvenlik", "{title} güvenlik açısından nelere dikkat edilmeli?"),
    ("test", "{title} nasıl test edilir?"),
    ("hata-ayıklama", "{title} ile ilgili bir hata nasıl ayıklanır?"),
    ("öğret", "{title} konusunu bir başkasına nasıl anlatırsın?"),
    ("karar", "{title} — hangi durumda hangi yaklaşım seçilmeli?"),
    ("sadeleştir", "{title} konusunu basitçe açıkla"),
)


def expand_curriculum(lessons: Optional[Sequence[Lesson]] = None,
                      patterns: int = 6,
                      layers: Optional[Sequence[str]] = None) -> List[Lesson]:
    """
    Müfredatı **soru kalıplarıyla** genişletir.

    Her ders, farklı soru kalıplarıyla birden fazla eğitim çiftine dönüşür.
    Hedef metin değişmez (bilgi uydurulmaz), yalnızca **soru biçimi** değişir —
    böylece model kavramı farklı soruluşlarda tanır.

    Args:
        lessons: Kaynak dersler (None = `build_curriculum()`)
        patterns: Ders başına kalıp sayısı (en fazla `len(QUESTION_PATTERNS)`)
        layers: Yalnızca bu katmanlar

    Returns:
        Genişletilmiş ders listesi
    """
    base = list(lessons) if lessons else build_curriculum(layers)
    selected = QUESTION_PATTERNS[:max(1, min(patterns, len(QUESTION_PATTERNS)))]
    expanded: List[Lesson] = []
    for lesson in base:
        for kind, template in selected:
            expanded.append(Lesson(
                layer=lesson.layer,
                title=template.format(title=lesson.title),
                body=lesson.body,
                code=lesson.code,
                tags=list(lesson.tags) + [kind],
            ))
    return expanded


def curriculum_stats(layers: Optional[Sequence[str]] = None,
                     expand: bool = False,
                     patterns: int = 6) -> Dict[str, object]:
    """Müfredat istatistikleri (katman başına ders ve karakter)."""
    lessons = (expand_curriculum(patterns=patterns, layers=layers)
               if expand else build_curriculum(layers))
    per_layer: Dict[str, Dict[str, int]] = {}
    for lesson in lessons:
        entry = per_layer.setdefault(lesson.layer, {"lessons": 0, "chars": 0})
        entry["lessons"] += 1
        entry["chars"] += len(lesson.as_document())
    return {
        "total_lessons": len(lessons),
        "total_chars": sum(item["chars"] for item in per_layer.values()),
        "per_layer": per_layer,
        "expanded": expand,
        "patterns": patterns if expand else 0,
        "layers": [LAYER_LABELS.get(name, name)
                   for name in (layers or LAYERS)],
    }


__all__ = [
    "Lesson", "build_curriculum", "expand_curriculum", "curriculum_stats",
    "LAYERS", "ALL_LAYERS", "NETWORK_LAYER_NAMES",
    "LOWPOLY_LAYER_NAMES", "GODOT_LAYER_NAMES",
    "LAYER_LABELS", "QUESTION_PATTERNS",
]
