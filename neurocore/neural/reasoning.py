"""
Akıl Yürütme Müfredatı
=======================
Modele **düşünmeyi** öğretir: adım adım akıl yürütme, karşılaştırma, karar
verme ve yorumlama.

Neden ayrı bir müfredat?
    `curriculum.py` **bilgiyi** öğretir ("X nedir"). Ama kullanıcının istediği
    yetenek farklı: *düşünerek en etkili yolu bulma*. Bu, bilgiyi **kullanma**
    becerisidir ve ayrı örnekler gerektirir:

        Bilgi     : "Blender'da küp eklemek için primitive_cube_add kullanılır."
        Düşünme   : "Küp mü küre mi? Sahne kompozisyonu için küp köşeli
                     yüzey verir; hedefim keskin hatlar → küp seçiyorum.
                     Alternatif: bezier eğrisi, ama fazla karmaşık."

Her ders bir **akıl yürütme izi (reasoning trace)** içerir. Model bu izleri
görerek yalnızca cevabı değil, **cevaba giden yolu** öğrenir.

Beş düşünme biçimi (her biri farklı bir soruyu yanıtlar):

    ANALYZE     — Bu problem gerçekte ne istiyor? (problemi anlama)
    COMPARE     — Hangi yaklaşım daha etkili, neden? (seçenek değerlendirme)
    DECIDE      — Hangi yolu seçiyorum ve neden? (karar + gerekçe)
    DIAGNOSE    — Bu hata neden oluyor, nasıl düzeltilir? (hata ayıklama)
    INTERPRET   — Bu bilgi ne anlama geliyor, ne yapmalıyım? (yorumlama)

Veri biçimi:
    İstem  : "DÜŞÜN: <problem>"
    Hedef  : "ADIM 1: …\\nADIM 2: …\\nKARAR: …"

Bu biçim, ajan katmanının modele "düşün" diyebilmesini sağlar; model de
adım adım yol üretir (veya aday yollar arasından en tutarlısını seçer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Düşünme biçimleri (sıra öğretim sırasıdır)
THINKING_MODES: Sequence[str] = (
    "analyze", "compare", "decide", "diagnose", "interpret",
)

#: Görünen etiketler
MODE_LABELS: Dict[str, str] = {
    "analyze": "Problemi Anla",
    "compare": "Seçenekleri Karşılaştır",
    "decide": "Karar Ver",
    "diagnose": "Hatayı Teşhis Et",
    "interpret": "Yorumla",
}

#: İstem öneki — ajan bu öneki kullanarak modelden akıl yürütme ister
THINK_PREFIX = "DÜŞÜN:"


@dataclass
class ReasoningLesson:
    """
    Tek bir akıl yürütme dersi.

    `trace` alanı **adım adım** yolu içerir; model bunu hedef olarak öğrenir.
    """
    mode: str
    problem: str
    trace: str
    topic: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Düşünme biçiminin görünen adı."""
        return MODE_LABELS.get(self.mode, self.mode)

    def prompt(self) -> str:
        """Eğitim çiftinin girdisi."""
        return f"{THINK_PREFIX} {self.problem}"

    def target(self) -> str:
        """Eğitim çiftinin hedefi (adım adım iz)."""
        return self.trace.strip()

    def as_document(self) -> str:
        """Tokenizer ve kasa için ham belge."""
        return f"{self.label} — {self.problem}\n\n{self.trace.strip()}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "mode": self.mode, "label": self.label, "problem": self.problem,
            "trace": self.trace, "topic": self.topic, "tags": list(self.tags),
        }


# ─── Düşünme Biçimi 1: Problemi Anla ────────────────────────────────────────

def _analyze_lessons() -> List[ReasoningLesson]:
    """Problemi anlama: gerçekte ne isteniyor?"""
    return [
        ReasoningLesson(
            "analyze",
            "Kullanıcı 'sitem yavaş' diyor. Ne yapmalıyım?",
            "ADIM 1: İfade belirsiz — 'yavaş' ölçülebilir değil.\n"
            "ADIM 2: Hangi katman yavaş? Ağ, veritabanı, ön yüz, arka plan işi?\n"
            "ADIM 3: Ölçüm olmadan tahmin etmek yanlış yola götürür.\n"
            "ADIM 4: Önce ölçmeliyim: yanıt süresi, CPU/RAM, sorgu sayısı.\n"
            "KARAR: Tahmin etmek yerine ölçüm isteyeceğim. Sorulacak soru: "
            "'Hangi sayfa ve hangi işlem yavaş, ne kadar sürüyor?'\n"
            "NEDEN: Yanlış katmanı iyileştirmek zaman kaybıdır.",
            topic="hata ayıklama", tags=["belirsizlik", "ölçüm"]),
        ReasoningLesson(
            "analyze",
            "Görev: 'Blender'da bir masa modeli oluştur.' Gerçekte ne gerekiyor?",
            "ADIM 1: 'Masa' birden fazla parça demek: tabla + 4 bacak.\n"
            "ADIM 2: Ölçüler belirtilmemiş → makul varsayım gerekli "
            "(tabla 1,2×0,8 m, yükseklik 0,75 m).\n"
            "ADIM 3: Malzeme belirtilmemiş → ahşap görünümü makul varsayım.\n"
            "ADIM 4: Çıktı biçimi belirtilmemiş → .blend + .glb mantıklı.\n"
            "KARAR: Varsayımları **açıkça yazıp** modeli kurmalıyım; "
            "sessiz varsayım kullanıcıyı yanıltır.\n"
            "NEDEN: Belirsizlikte en iyi yol, varsayımı görünür kılmaktır.",
            topic="blender", tags=["belirsizlik", "varsayım"]),
        ReasoningLesson(
            "analyze",
            "Python betiği 10.000 satırlık dosyayı satır satır okuyor. Sorun ne?",
            "ADIM 1: 'Satır satır okuma' G/Ç çağrısı sayısını artırır.\n"
            "ADIM 2: Her okuma sistem çağrısı → disk/ağ gecikmesi.\n"
            "ADIM 3: Dosya belleğe sığıyorsa tek seferde okumak yeterli.\n"
            "ADIM 4: Sığmıyorsa tamponlu (buffered) okuma kullanılmalı.\n"
            "KARAR: Asıl problem algoritma değil, **G/Ç deseni**.\n"
            "NEDEN: Darboğazı doğru katmanda aramak gerekir.",
            topic="performans", tags=["g/ç", "darboğaz"]),
        ReasoningLesson(
            "analyze",
            "Model eğitim kaybı 0,2'ye düştü ama üretim saçma. Neden?",
            "ADIM 1: Düşük kayıp **eğitim verisinde** iyi olduğunu gösterir.\n"
            "ADIM 2: Üretim farklı bir dağılım → genelleme sorunu.\n"
            "ADIM 3: Olası nedenler: veri çok küçük (ezber), model aşırı büyük, "
            "düzenlileştirme yok.\n"
            "ADIM 4: Ölçüm: 3,4M token / 208M parametre = parametre başına "
            "0,016 token. Sağlıklı oran ~20 token/parametre.\n"
            "KARAR: Sorun **veri ölçeği**, model değil.\n"
            "NEDEN: Kayıp tek başına yeterli ölçü değildir; orana bakılır.",
            topic="yapay zeka", tags=["genelleme", "ölçek"]),
    ]


# ─── Düşünme Biçimi 2: Seçenekleri Karşılaştır ──────────────────────────────

def _compare_lessons() -> List[ReasoningLesson]:
    """Karşılaştırma: hangi yaklaşım daha etkili?"""
    return [
        ReasoningLesson(
            "compare",
            "Blender'da 100 nesne oluşturulacak: döngü mü, kopyalama mı?",
            "SEÇENEK A: Her nesne için `bpy.ops.mesh.primitive_cube_add()`.\n"
            "  + Basit, okunur.\n"
            "  − Her `bpy.ops` çağrısı sahneyi yeniden değerlendirir → yavaş.\n"
            "SEÇENEK B: Bir küp oluştur, `bpy.data.objects.new()` ile kopyala.\n"
            "  + Sahne güncellemesi bir kez → çok daha hızlı.\n"
            "  − Biraz daha fazla kod.\n"
            "KARŞILAŞTIRMA: 100 nesnede A ≈ 4-8 sn, B ≈ 0,3 sn.\n"
            "KARAR: B'yi seçiyorum. Eşik: 20+ nesne → veri API'si; "
            "az nesne → `bpy.ops` okunurluk için yeterli.\n"
            "NEDEN: `bpy.ops` bağlam bağımlıdır ve tekrar tekrar çağrılması "
            "ölçeklenmez.",
            topic="blender", tags=["performans", "api"]),
        ReasoningLesson(
            "compare",
            "Ağ istekleri: `requests` mi `aiohttp` mi?",
            "SEÇENEK A: `requests` — senkron, basit, olgun.\n"
            "  + Tek istek veya az sayıda istek için ideal.\n"
            "  − 100 istekte sırayla bekler → toplam süre = toplam gecikme.\n"
            "SEÇENEK B: `aiohttp` — eşzamanlı (async).\n"
            "  + 100 istek paralel → süre ≈ en yavaş istek.\n"
            "  − Ek bağımlılık, async bağlam gerekir.\n"
            "KARŞILAŞTIRMA: 100 istek × 200 ms → A ≈ 20 sn, B ≈ 0,5 sn.\n"
            "KARAR: Az istek → `requests`; çok istek → `aiohttp`. "
            "Eşik ~10 istek.\n"
            "NEDEN: G/Ç bekleyen işte eşzamanlılık kazancı doğrusaldır.",
            topic="kütüphaneler", tags=["eşzamanlılık", "ağ"]),
        ReasoningLesson(
            "compare",
            "Model eğitimi: CPU mu GPU mu?",
            "SEÇENEK A: CPU — 16 çekirdek, 32 GB RAM.\n"
            "  + Büyük model sığar (RAM sınırı yok).\n"
            "  − Matris çarpımı 20-50× yavaş.\n"
            "SEÇENEK B: GPU — 1080 Ti, 11 GB VRAM.\n"
            "  + Paralel matris çarpımı çok hızlı.\n"
            "  − VRAM sınırı: ~740M parametre üstü sığmaz.\n"
            "KARŞILAŞTIRMA: 208M modelde ölçüldü: GPU ~5,5 sn/adım, "
            "CPU ~60 sn/adım (≈11×).\n"
            "KARAR: Model VRAM'e sığıyorsa GPU; sığmıyorsa CPU + daha küçük "
            "toplu iş. Eşik: ~700M parametre.\n"
            "NEDEN: Yapay sinir ağı eğitimi matris çarpımına indirgenir; "
            "GPU bunun için tasarlanmıştır.",
            topic="donanım", tags=["gpu", "cpu", "ölçek"]),
        ReasoningLesson(
            "compare",
            "Bilgi saklama: dosya mı, veritabanı mı, bellek mi?",
            "SEÇENEK A: Bellek (dict) — en hızlı, süreç kapanınca kaybolur.\n"
            "SEÇENEK B: JSON dosyası — okunur, atomik yazma gerekir, "
            "eşzamanlı erişimde bozulur.\n"
            "SEÇENEK C: SQLite — eşzamanlı okuma, indeks, sorgu, kalıcı.\n"
            "KARŞILAMA: 500 notta JSON okuma ≈ 5 ms, yazma ≈ 20 ms; "
            "SQLite okuma ≈ 1 ms (indeksli).\n"
            "KARAR: <100 kayıt → JSON; >100 veya arama gerekli → SQLite; "
            "oturum içi geçici → bellek.\n"
            "NEDEN: Veri hacmi büyüdükçe tüm dosyayı okumak sürdürülemez.",
            topic="kütüphaneler", tags=["depolama", "ölçek"]),
    ]


# ─── Düşünme Biçimi 3: Karar Ver ────────────────────────────────────────────

def _decide_lessons() -> List[ReasoningLesson]:
    """Karar verme: hangi yolu seçiyorum ve neden?"""
    return [
        ReasoningLesson(
            "decide",
            "Bir Python betiği 30 saniyeden uzun sürüyor. Ne yapmalıyım?",
            "ADIM 1: Önce **ölç**: hangi satır/çağrı yavaş? (`cProfile`)\n"
            "ADIM 2: Darboğaz G/Ç ise → eşzamanlılık (`asyncio`) veya toplu işlem.\n"
            "ADIM 3: Darboğaz CPU ise → algoritma karmaşıklığını düşür "
            "(O(n²) → O(n log n)) veya `numpy` ile vektörleştir.\n"
            "ADIM 4: Darboğaz dış servis ise → önbellek (cache) ekle.\n"
            "KARAR: Profil çıktısına göre ilerleyeceğim; "
            "ölçmeden iyileştirme yapmayacağım.\n"
            "GEREKÇE: En sık hata, yanlış katmanı iyileştirmektir. "
            "Ölçüm, doğru katmanı gösterir.",
            topic="performans", tags=["profil", "karar"]),
        ReasoningLesson(
            "decide",
            "Kullanıcı 'modern bir işletim sistemi yaz' dedi. Nereden başlarım?",
            "ADIM 1: Bu **yıllar** süren bir proje; tek oturumda bitmez.\n"
            "ADIM 2: Doğru yaklaşım: en küçük çalışan parçadan başla "
            "(önyükleme + ekrana metin yazma).\n"
            "ADIM 3: Sonra sırayla: kesme yönetimi, bellek sayfalama, "
            "süreç zamanlama, dosya sistemi, sürücüler.\n"
            "ADIM 4: Her katman çalıştıktan sonra test edilmeli "
            "(gerçek donanım veya emülatör).\n"
            "KARAR: Aşamalı ilerleyeceğim ve her aşamayı doğrulayacağım. "
            "İlk hedef: QEMU'da önyüklenen, 'Merhaba' yazan çekirdek.\n"
            "GEREKÇE: Büyük hedefi küçük doğrulanabilir adımlara bölmek, "
            "hem ilerlemeyi görünür kılar hem hatayı erken yakalar.",
            topic="işletim sistemi", tags=["planlama", "aşamalı"]),
        ReasoningLesson(
            "decide",
            "Model üretim yapamıyor. Veriyi mi büyütmeli, modeli mi?",
            "ADIM 1: Ölçülen oran: 3,4M token / 208M parametre = "
            "0,016 token/parametre.\n"
            "ADIM 2: Sağlıklı aralık ~20 token/parametre → 4 milyar token "
            "gerekir (elimizde 3,4 milyon; ~1.200× eksik).\n"
            "ADIM 3: Modeli büyütmek bu oranı **daha da kötüleştirir**.\n"
            "ADIM 4: Doğru sıra: önce veri (hem oranı düzeltir hem bilgi "
            "ekler), sonra model.\n"
            "KARAR: Önce veriyi büyüteceğim; model ancak veri yeterli "
            "olduğunda büyütülür.\n"
            "GEREKÇE: Eksik veriyle büyük model ezberler; kayıp düşer ama "
            "üretim bozulur (ölçüldü: kayıp 0,2 ama çıktı tutarsız).",
            topic="yapay zeka", tags=["ölçek", "karar", "veri"]),
        ReasoningLesson(
            "decide",
            "Ajan bir görevi 24 adımda bitiremedi. Ne yapmalıyım?",
            "ADIM 1: Adım sınırı bir **koruma**dır: sonsuz döngüyü ve "
            "kaynak israfını engeller.\n"
            "ADIM 2: Sınırı yükseltmek yalnızca belirtiyi gizler.\n"
            "ADIM 3: Asıl soru: görev neden çok adım aldı? "
            "Belirsiz hedef, yanlış araç seçimi veya tekrarlanan deneme.\n"
            "ADIM 4: Doğru çözüm: görevi **alt görevlere böl** ve her birini "
            "ayrı çalıştır.\n"
            "KARAR: Görevi böleceğim; sınırı yükseltmeyeceğim.\n"
            "GEREKÇE: Uzun görevler hata ayıklaması zor ve maliyetlidir; "
            "bölmek hem doğrulanabilir hem sürdürülebilir.",
            topic="planlama", tags=["adım sınırı", "bölme"]),
    ]


# ─── Düşünme Biçimi 4: Hatayı Teşhis Et ─────────────────────────────────────

def _diagnose_lessons() -> List[ReasoningLesson]:
    """Hata teşhisi: neden oluyor, nasıl düzeltilir?"""
    return [
        ReasoningLesson(
            "diagnose",
            "Blender betiği hiçbir çıktı vermeden başarısız oluyor. Neden?",
            "BELİRTİ: Betik çalışıyor görünüyor ama nesne oluşmuyor, hata da "
            "görünmüyor.\n"
            "OLASI NEDEN 1: `bpy.ops` çağrıları **bağlam** (context) bağımlı; "
            "nesne aktif değilse işlem sessizce başarısız olur.\n"
            "OLASI NEDEN 2: Betik istisna atıyor ama Blender çıktı kodunu "
            "sıfır döndürdüğü için hata görünmüyor.\n"
            "AYIRT ETME: `blender --background --python-exit-code 1 --python "
            "betik.py` ile çalıştır → istisna çıkış koduna yansır.\n"
            "ÇÖZÜM: 1) `--python-exit-code 1` ekle. 2) Aktif nesneyi denetle: "
            "`if bpy.context.active_object is None: raise`. 3) Betiği "
            "try/except ile sarmala ve hatayı `print` ile bildir.\n"
            "DOĞRULAMA: Betiği bilerek boz, çıkış kodunun 1 olduğunu gör.",
            topic="blender", tags=["sessiz hata", "bağlam"]),
        ReasoningLesson(
            "diagnose",
            "Eğitim 'unscale_() has already been called' hatası veriyor. Neden?",
            "BELİRTİ: Eğitim 42 adım sonra çöküyor, ilerleme kaydedilmiyor.\n"
            "OLASI NEDEN: `GradScaler.unscale_()` bir optimizer için "
            "çağrıldıktan sonra `step()` çağrılmadan tekrar çağrılıyor.\n"
            "KAYNAK: NaN gradyan dalı `step()` çağırmadan `continue` ediyor; "
            "scaler'ın iç bayrağı sıfırlanmıyor.\n"
            "AYIRT ETME: Hatanın **ilk** oluştuğu adımı bul; öncesinde NaN "
            "gradyan uyarısı var mı?\n"
            "ÇÖZÜM: Atlanan adımda `scaler = GradScaler()` ile yeni scaler kur "
            "(bayrak temizlenir, ölçek makul değere döner).\n"
            "DOĞRULAMA: NaN gradyan üretip sonraki adımın çalıştığını gör.",
            topic="yapay zeka", tags=["amp", "gradyan"]),
        ReasoningLesson(
            "diagnose",
            "API 403 hatası veriyor ama anahtar doğru. Neden?",
            "BELİRTİ: `HTTP 403 error code: 1010`\n"
            "OLASI NEDEN 1: Anahtar geçersiz → ama 403 değil 401 dönerdi.\n"
            "OLASI NEDEN 2: Bölge kısıtı → 403 döner.\n"
            "OLASI NEDEN 3: **Bot koruması** → istek kimliği eksik "
            "(User-Agent yok).\n"
            "AYIRT ETME: `Python-urllib/3.x` varsayılan kimliği engellenir; "
            "tarayıcı kimliği gönderince 200 döner.\n"
            "ÇÖZÜM: İstek başlığına `User-Agent` ekle.\n"
            "DOĞRULAMA: Aynı istek başlıklı/başlıksız karşılaştır.\n"
            "DERS: HTTP kodunu **tek başına** yorumlamak yanıltır; "
            "başlıklar ve gövde birlikte okunur.",
            topic="kütüphaneler", tags=["http", "403", "başlık"]),
        ReasoningLesson(
            "diagnose",
            "Eğitim 8 turun 7'sinde 'kaynaklar dolu' diyip duruyor. Neden?",
            "BELİRTİ: GPU gerçekte boş (%0 kullanım, 10,5 GB boş) ama "
            "kaynak kapısı kapalı.\n"
            "OLASI NEDEN 1: Sistem gerçekten dolu → ölçüm aksini söylüyor.\n"
            "OLASI NEDEN 2: **Kendi ayırdığı** bellek sistem sanılıyor → "
            "eğitim VRAM'i doldurunca sonraki turun ön kontrolü dolu görür.\n"
            "OLASI NEDEN 3: Kapanış VRAM'i bırakmıyor → ölçüldü: "
            "`shutdown()` sonrası 1450 MiB'de kalıyor.\n"
            "AYIRT ETME: Süreç kapandıktan sonra `nvidia-smi` ile bak; "
            "`gc.collect()` + `torch.cuda.empty_cache()` sonrası tekrar bak.\n"
            "ÇÖZÜM: 1) Süreç kapanışında bellek boşalt. 2) Kendi işinin "
            "yarattığı yükü 'sistem dolu' sayma (kullanıcı istekli işte "
            "eşiği yükselt). 3) Turlar arasında motoru yeniden yükle.\n"
            "DOĞRULAMA: 8 turun 8'inin tamamlanması.",
            topic="donanım", tags=["vram", "kaynak", "kapanış"]),
    ]


# ─── Düşünme Biçimi 5: Yorumla ──────────────────────────────────────────────

def _interpret_lessons() -> List[ReasoningLesson]:
    """Yorumlama: bu bilgi ne anlama geliyor, ne yapmalıyım?"""
    return [
        ReasoningLesson(
            "interpret",
            "Ölçüm: eğitim kaybı 4,62 → 0,24. Bu ne anlama geliyor?",
            "VERİ: Kayıp %95 düştü.\n"
            "İYİ YORUM: Model eğitim verisindeki örüntüleri öğrendi.\n"
            "KÖTÜ YORUM: 'Model artık iyi metin yazıyor' — kayıp bunu "
            "söylemez.\n"
            "AYIRT ETME: Kayıp yalnızca **eğitim dağılımında** başarıyı "
            "ölçer. Genelleme için ayrı doğrulama kümesi ve gerçek üretim "
            "testi gerekir.\n"
            "EKSİK BİLGİ: Doğrulama kaybı, üretim kalitesi, ezber oranı.\n"
            "NE YAPMALIYIM: Kayıp düşüşünü **kutlamadan önce** üretim testi "
            "yap. Ölçüldü: kayıp 0,24 iken üretim tutarsız çıktı.\n"
            "DERS: Tek metrik yeterli değildir; her metrik bir soruyu "
            "yanıtlar, diğerlerini sessiz bırakır.",
            topic="yapay zeka", tags=["metrik", "yorum"]),
        ReasoningLesson(
            "interpret",
            "Ölçüm: tokenizer 1,73 → 4,16 karakter/token oldu. Ne kazandık?",
            "VERİ: Aynı metin 2,4× daha az token ile temsil ediliyor.\n"
            "DOĞRUDAN KAZANÇ: Aynı bağlam penceresine 2,4× fazla içerik sığar; "
            "aynı donanımda 2,4× fazla veri işlenir.\n"
            "DOLAYLI KAZANÇ: Eğitim adımı başına daha fazla bilgi → "
            "yakınsama hızlanır.\n"
            "MALİYET: Gömme katmanı büyüdü (1.259 → 4.301 satır) ve "
            "**token kimlikleri değişti**. Eski öğrenilmiş temsiller yeni "
            "kimliklerle uyuşmaz → model baştan öğrenmek zorunda.\n"
            "ÖLÇÜLDÜ: Kayıp 8,08'e sıçradı, sonra 1,42'ye indi.\n"
            "NE YAPMALIYIM: Tokenizer değişimi tek seferlik olmalı; sık "
            "değiştirmek öğrenilmiş temsili çöpe atar.",
            topic="yapay zeka", tags=["tokenizer", "maliyet"]),
        ReasoningLesson(
            "interpret",
            "Ölçüm: yerel model Blender kod seçiminde %70, rastgele %25. "
            "Bu ne anlama geliyor?",
            "VERİ: 2,8× rastgeleden iyi.\n"
            "İYİ YORUM: Model kod parçalarının **bağlama uygunluğunu** "
            "ayırt edebiliyor — üretemese bile.\n"
            "SINIR: %70 kesin değil; 10 sorunun 3'ünde yanlış. Bu yüzden "
            "sonuç 'öneri' olarak sunulmalı, karar olarak değil.\n"
            "AYIRT ETME: Rastgele tabanı bilmeden %70 anlamsızdır. "
            "4 adaylı soruda taban %25; 2 adaylıda %50 olurdu.\n"
            "NE YAPMALIYIM: Aracı, **belirsizlik bildirecek** biçimde "
            "kullan: skor farkı küçükse 'belirsiz' de ve karar verme.\n"
            "DERS: Bir metriğin anlamı, **taban çizgisine** göre belirlenir.",
            topic="yapay zeka", tags=["metrik", "taban"]),
        ReasoningLesson(
            "interpret",
            "Ölçüm: model kendi eğitim verisini bile geri getiremiyor (0/3). "
            "Bu ne anlama geliyor?",
            "VERİ: Ezber bile yok.\n"
            "YORUM 1: Kayıp düşük ama ezber yok → model **istatistiksel "
            "örüntüleri** öğrendi, dizileri değil.\n"
            "YORUM 2: Bu aslında iyi haber: aşırı öğrenme (overfitting) yok.\n"
            "YORUM 3: Ama üretim için yeterli değil; üretim dizisel tutarlılık "
            "gerektirir.\n"
            "AYIRT ETME: 3,4M token / 208M parametre → parametre başına "
            "0,016 token. Dil modeli için sağlıklı oran ~20.\n"
            "NE YAPMALIYIM: Veriyi 1.000× büyütmeden veya modeli küçültmeden "
            "üretim beklenmemeli.\n"
            "DERS: 'Ezberleyemiyor ama kayıp düşük' tablosu, veri açlığının "
            "imzasıdır.",
            topic="yapay zeka", tags=["ezber", "öğrenme", "yorum"]),
    ]


# ─── Müfredat ───────────────────────────────────────────────────────────────

def build_reasoning_curriculum(
        modes: Optional[Sequence[str]] = None) -> List[ReasoningLesson]:
    """
    Akıl yürütme müfredatını üretir.

    Args:
        modes: Yalnızca bu düşünme biçimleri (None = hepsi)

    Returns:
        Ders listesi (düşünme biçimi sırasına göre)
    """
    builders = {
        "analyze": _analyze_lessons,
        "compare": _compare_lessons,
        "decide": _decide_lessons,
        "diagnose": _diagnose_lessons,
        "interpret": _interpret_lessons,
    }
    istenen = list(modes) if modes else list(THINKING_MODES)
    dersler: List[ReasoningLesson] = []
    for ad in istenen:
        uretici = builders.get(ad)
        if uretici is not None:
            dersler.extend(uretici())
    return dersler


def reasoning_stats(modes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Akıl yürütme müfredatı istatistikleri."""
    dersler = build_reasoning_curriculum(modes)
    per_mode: Dict[str, Dict[str, int]] = {}
    for ders in dersler:
        kayit = per_mode.setdefault(ders.mode, {"lessons": 0, "chars": 0})
        kayit["lessons"] += 1
        kayit["chars"] += len(ders.as_document())
    return {
        "total_lessons": len(dersler),
        "total_chars": sum(item["chars"] for item in per_mode.values()),
        "per_mode": per_mode,
        "modes": list(modes) if modes else list(THINKING_MODES),
    }


__all__ = [
    "ReasoningLesson", "build_reasoning_curriculum", "reasoning_stats",
    "THINKING_MODES", "MODE_LABELS", "THINK_PREFIX",
]
