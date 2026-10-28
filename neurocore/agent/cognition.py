"""
Bilişsel Katman — hata taksonomisi · hedef durumu · öz-yansıma · doğrulama
==========================================================================
Amaç: ajanın **ham zekâsını** değil, **tutarlılığını** artırmak.

⚠ Dürüst sınır
--------------
Bu katman modeli akıllandırmaz. Bulut modelinin (ajanın beyni) yeteneği
tavanı belirler. Katman şunları garanti eder:

```
1. HATA TAKSONOMİSİ : aynı hatayı körlemesine tekrarlamaz; hata türüne
                      göre **farklı** kurtarma uygular
2. HEDEF DURUMU      : ne istendi, ne biliniyor, ne denendi — yapılandırılmış
                      tutulur; bağlam şişmeden taşınır
3. ÖZ-YANSIMA        : belirli aralıkta "hedefe yaklaşıyor muyum?" diye sorar;
                      gerekiyorsa **yeniden planlatır**
4. KANIT DOĞRULAMA   : nihai yanıt, araç çıktılarıyla destekleniyor mu?
                      Desteklenmeyen iddia varsa **bir kez** düzeltme ister
```

Her çağrı **sınırlıdır** (en fazla 2 yansıma, 1 düzeltme) ve başarısız
olursa ajanı **bozmaz** — sessizce devre dışı kalır.

Katman sözleşmesi: `initialize(config, event_bus)` / `get_stats()`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.cognition")

#: Tek yansıma/doğrulama çağrısında üretilecek en fazla token
YANSIMA_TOKEN = 320
DOGRULAMA_TOKEN = 320

#: Bağlama girecek hedef özetinin üst sınırı (karakter)
OZET_SINIRI = 1200


# ─── 1) Hata taksonomisi ────────────────────────────────────────────────────

class HataTuru(str, Enum):
    """Araç hatasının türü — kurtarma politikasını belirler."""

    YOK = "yok"
    SOZDIZIMI = "sozdizimi"        # sözdizimi / derleme hatası
    IZIN = "izin"                  # yetki / onay / sandbox
    AG = "ag"                      # ağ / zaman aşımı / sağlayıcı
    ARAC_YOK = "arac_yok"          # araç adı uydurulmuş
    ARGUMAN = "arguman"            # eksik/yanlış parametre
    KAYNAK = "kaynak"              # RAM/GPU/disk dolu
    BULUNAMADI = "bulunamadi"      # dosya/yol yok
    BILINMEYEN = "bilinmeyen"


#: Tür → (eylem, modele verilecek talimat)
#
# ⚠ Ölçülen sorun: hata türü ayırt edilmeden aynı çağrı 3 kez tekrarlanıyordu
# (`REPEAT_LIMIT`). Türüne göre **farklı** kurtarma vermek tur sayısını düşürür.
KURTARMA: Dict[HataTuru, Dict[str, str]] = {
    HataTuru.SOZDIZIMI: {
        "eylem": "duzelt_ve_tekrar",
        "talimat": ("Sözdizimi/derleme hatası. Kodu **oku**, hatalı satırı bul, "
                    "tek bir düzeltme yap ve yeniden çalıştır. Aynı komutu "
                    "değiştirmeden tekrar etme."),
    },
    HataTuru.IZIN: {
        "eylem": "kullaniciya_bildir",
        "talimat": ("Yetki/onay hatası. Bu işlem izin gerektiriyor. Alternatif "
                    "bir yol dene; yoksa kullanıcıya **hangi izin** gerektiğini "
                    "açıkça söyle."),
    },
    HataTuru.AG: {
        "eylem": "bekle_tekrar",
        "talimat": ("Ağ/sağlayıcı hatası. Kısa bir bekleme sonrası **bir kez** "
                    "tekrar dene; yine olmazsa elindeki bilgiyle devam et."),
    },
    HataTuru.ARAC_YOK: {
        "eylem": "dogru_adi_kullan",
        "talimat": ("Böyle bir araç yok. Yalnız **listede verilen** araç adlarını "
                    "kullan; en yakın işlevi olan araçla devam et."),
    },
    HataTuru.ARGUMAN: {
        "eylem": "argumani_duzelt",
        "talimat": ("Parametre hatası. Aracın şemasına bak: zorunlu alanları "
                    "tam ve doğru tipte ver."),
    },
    HataTuru.KAYNAK: {
        "eylem": "kucult",
        "talimat": ("Kaynak dolu. İşi **küçült** (daha az dosya/örnek/adım) veya "
                    "daha hafif bir araç seç."),
    },
    HataTuru.BULUNAMADI: {
        "eylem": "kesfet",
        "talimat": ("Dosya/yol bulunamadı. Önce dizini **listele** veya "
                    "çalışma alanı haritasını çıkar, sonra doğru yolu kullan."),
    },
    HataTuru.BILINMEYEN: {
        "eylem": "farkli_yol",
        "talimat": ("Bilinmeyen hata. Aynı yaklaşımı tekrarlama; **farklı** bir "
                    "yol dene veya durumu kullanıcıya bildir."),
    },
}

_KALIPLAR = (
    (HataTuru.IZIN, ("permission", "izin", "erişim reddedildi", "access denied",
                     "not permitted", "onay", "sandbox", "denied")),
    (HataTuru.AG, ("timeout", "zaman aşımı", "connection", "bağlantı", "network",
                   "ağ", "429", "503", "unreachable", "ssl")),
    (HataTuru.KAYNAK, ("out of memory", "cuda", "bellek yetersiz", "no space",
                       "disk dolu", "vram", "kaynaklar_dolu")),
    # ⚠ SIRA ÖLÇÜLDÜ: "missing" hem ARGUMAN hem BULUNAMADI kalıbıydı ve
    # "TypeError: missing required argument" yanlışlıkla BULUNAMADI
    # sınıflanıyordu (test yakaladı). ARGUMAN artık önce gelir ve
    # BULUNAMADI listesinden "missing" çıkarıldı.
    (HataTuru.ARGUMAN, ("missing", "required", "invalid argument", "typeerror",
                        "geçersiz", "zorunlu", "beklenen", "expected")),
    (HataTuru.BULUNAMADI, ("not found", "bulunamadı", "no such file", "yok:",
                           "does not exist", "filenotfound")),
    (HataTuru.SOZDIZIMI, ("syntaxerror", "syntax error", "sozdizimi", "sözdizimi",
                          "unexpected token", "indentationerror", "compile",
                          "derleme", "traceback")),
    (HataTuru.ARAC_YOK, ("unknown tool", "araç yok", "no such tool",
                         "not available")),
)


def siniflandir(metin: str) -> HataTuru:
    """
    Hata metnini türe ayırır.

    ⚠ Sıra önemli: "permission denied" hem izin hem bulunamadı gibi
    görünebilir; en **özgül** tür önce denenir (KALIPLAR sırası).
    """
    kucuk = str(metin or "").lower()
    if not kucuk.strip():
        return HataTuru.YOK
    for tur, anahtarlar in _KALIPLAR:
        if any(a in kucuk for a in anahtarlar):
            return tur
    return HataTuru.BILINMEYEN


def kurtarma_talimati(tur: HataTuru) -> str:
    """Tür için modele verilecek kurtarma talimatı."""
    return KURTARMA.get(tur, KURTARMA[HataTuru.BILINMEYEN])["talimat"]


# ─── 2) Hedef durumu ───────────────────────────────────────────────────────

@dataclass
class HedefDurumu:
    """
    Bir çalışmanın **yapılandırılmış** durumu.

    Amaç: ham mesaj geçmişi şişerken kritik bilgi (ne istendi, ne öğrenildi,
    ne denendi) kaybolmasın.
    """

    hedef: str
    facts: List[str] = field(default_factory=list)
    attempts: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    #: En fazla kaç olgu/deneme saklanır (bağlam sınırı)
    MAX_FACT = 12
    MAX_ATTEMPT = 16

    def olgu_ekle(self, metin: str) -> None:
        """Doğrulanmış bilgi parçası ekler (tekrarı atlar)."""
        temiz = " ".join(str(metin or "").split())[:240]
        if not temiz or temiz in self.facts:
            return
        self.facts.append(temiz)
        if len(self.facts) > self.MAX_FACT:
            self.facts = self.facts[-self.MAX_FACT:]

    def deneme_ekle(self, metin: str) -> None:
        """Yapılan deneme (araç + sonuç) ekler."""
        temiz = " ".join(str(metin or "").split())[:160]
        if not temiz:
            return
        self.attempts.append(temiz)
        if len(self.attempts) > self.MAX_ATTEMPT:
            self.attempts = self.attempts[-self.MAX_ATTEMPT:]

    def ozet(self, sinir: int = OZET_SINIRI) -> str:
        """Bağlama girecek kısa metin."""
        parcalar = [f"HEDEF: {self.hedef[:300]}"]
        if self.facts:
            parcalar.append("BİLİNENLER:\n" +
                            "\n".join(f"- {f}" for f in self.facts))
        if self.attempts:
            parcalar.append("DENENENLER:\n" +
                            "\n".join(f"- {a}" for a in self.attempts[-8:]))
        if self.open_questions:
            parcalar.append("AÇIK SORULAR:\n" +
                            "\n".join(f"- {q}" for q in self.open_questions[:6]))
        return "\n\n".join(parcalar)[:sinir]


# ─── 3) Öz-yansıma ─────────────────────────────────────────────────────────

@dataclass
class Yansima:
    """Öz-yansıma sonucu."""

    verdict: str = "devam"        # devam | yeniden_planla | cevapla | bildir
    missing: List[str] = field(default_factory=list)
    next_action: str = ""
    raw: str = ""
    ok: bool = True
    error: str = ""

    @classmethod
    def coz(cls, metin: str) -> "Yansima":
        """Modelin JSON yanıtını ayrıştırır (dayanıklı)."""
        veri = _json_cikar(metin)
        if not isinstance(veri, dict):
            return cls(ok=False, error="json çözülemedi", raw=str(metin)[:400])
        verdict = str(veri.get("verdict") or veri.get("karar") or "devam").lower()
        if verdict not in ("devam", "yeniden_planla", "cevapla", "bildir"):
            verdict = "devam"
        eksik = veri.get("missing") or veri.get("eksik") or []
        if isinstance(eksik, str):
            eksik = [eksik]
        return cls(
            verdict=verdict,
            missing=[str(e)[:160] for e in (eksik or [])][:5],
            next_action=str(veri.get("next") or veri.get("sonraki") or "")[:300],
            raw=str(metin)[:400],
        )


_YANSIMA_ISTEMI = """Sen bir ajanın **öz-yansıma** denetçisisin. Verilen hedefe
ne kadar yaklaşıldığını değerlendir. Yalnızca JSON döndür, başka metin yazma:

{{"verdict": "devam|yeniden_planla|cevapla|bildir",
  "missing": ["eksik bilgi/kanıt", "..."],
  "next": "sıradaki en iyi tek eylem"}}

Kurallar:
- "cevapla": hedef için yeterli kanıt var, yanıt verilebilir
- "yeniden_planla": mevcut yaklaşım ilerlemiyor, farklı plan gerekir
- "bildir": kullanıcı girdisi/izni olmadan ilerlenemez
- "devam": doğru yolda, sürdür
- missing: en fazla 3 madde, kısa

HEDEF: {hedef}

DURUM:
{durum}

SON HATA: {hata}
"""


def yansit(router: Any, hedef: str, durum: str, last_error: str = "",
           *, max_tokens: int = YANSIMA_TOKEN) -> Yansima:
    """
    Ajanın ilerlemesini değerlendirir (tek, kısa çağrı).

    ⚠ Çağrı başarısız olursa `ok=False` döner ve ajan **normal akışına**
    devam eder — bilişsel katman asla akışı kilitlemez.
    """
    if router is None:
        return Yansima(ok=False, error="router yok")
    istem = _YANSIMA_ISTEMI.format(
        hedef=str(hedef)[:300], durum=str(durum)[:1500],
        hata=str(last_error)[:300] or "—")
    try:
        from neurocore.agent.llm import LLMMessage

        yanit = router.chat([LLMMessage("user", istem)], max_tokens=max_tokens,
                            temperature=0.0)
    except Exception as exc:                     # ağ/uyumluluk hatası
        logger.debug(f"Yansıma çağrısı başarısız: {type(exc).__name__}: {exc}")
        return Yansima(ok=False, error=f"{type(exc).__name__}: {exc}")
    if not getattr(yanit, "ok", False):
        return Yansima(ok=False, error=str(getattr(yanit, "error", ""))[:200])
    sonuc = Yansima.coz(getattr(yanit, "text", "") or "")
    if not sonuc.ok:
        logger.debug(f"Yansıma ayrıştırılamadı: {sonuc.raw[:120]}")
    return sonuc


# ─── 4) Kanıt doğrulama ────────────────────────────────────────────────────

@dataclass
class Dogrulama:
    """Nihai yanıtın kanıta dayanıp dayanmadığı."""

    supported: bool = True
    score: float = 1.0
    unsupported: List[str] = field(default_factory=list)
    raw: str = ""
    ok: bool = True
    error: str = ""

    @classmethod
    def coz(cls, metin: str) -> "Dogrulama":
        veri = _json_cikar(metin)
        if not isinstance(veri, dict):
            return cls(ok=False, error="json çözülemedi", raw=str(metin)[:400])
        destek = veri.get("supported", veri.get("destekli"))
        if isinstance(destek, str):
            destek = destek.lower() in ("true", "evet", "1", "yes")
        puan = veri.get("score", veri.get("puan", 1.0))
        try:
            puan = float(puan)
        except (TypeError, ValueError):
            puan = 1.0 if destek else 0.0
        eksik = veri.get("unsupported") or veri.get("desteksiz") or []
        if isinstance(eksik, str):
            eksik = [eksik]
        return cls(
            supported=bool(destek if destek is not None else puan >= 0.6),
            score=max(0.0, min(1.0, puan)),
            unsupported=[str(e)[:200] for e in (eksik or [])][:5],
            raw=str(metin)[:400],
        )


_DOGRULAMA_ISTEMI = """Sen bir **kanıt denetçisisin**. Nihai yanıtın, araç
çıktılarıyla desteklenip desteklenmediğini değerlendir. Yalnızca JSON döndür:

{{"supported": true|false, "score": 0.0-1.0,
  "unsupported": ["kanıtsız iddia", "..."]}}

Kurallar:
- Yalnızca **araç çıktısında görülen** bilgi desteklidir.
- Genel bilgi (ör. tanım) desteklenmiş sayılır; **somut dosya/sayı/sürüm**
  iddiası araç çıktısında yoksa desteksizdir.
- unsupported: en fazla 3 madde, kısa.

HEDEF: {hedef}

ARAÇ ÇIKTILARI (kanıt):
{kanit}

NİHAİ YANIT:
{cevap}
"""


def dogrula(router: Any, hedef: str, cevap: str, kanit: str,
            *, max_tokens: int = DOGRULAMA_TOKEN) -> Dogrulama:
    """
    Nihai yanıtı kanıta karşı denetler (tek, kısa çağrı).

    ⚠ Bu bir **kapı değil, uyarıdır**: desteksiz çıkarsa yanıt
    reddedilmez; bir kez düzeltme istenir, olmazsa yanıt yine verilir
    (kullanıcıyı boş bırakmamak için).
    """
    if router is None or not str(cevap or "").strip():
        return Dogrulama(ok=False, error="router/cevap yok")
    istem = _DOGRULAMA_ISTEMI.format(
        hedef=str(hedef)[:300], kanit=str(kanit)[:2500],
        cevap=str(cevap)[:2500])
    try:
        from neurocore.agent.llm import LLMMessage

        yanit = router.chat([LLMMessage("user", istem)], max_tokens=max_tokens,
                            temperature=0.0)
    except Exception as exc:
        logger.debug(f"Doğrulama çağrısı başarısız: {type(exc).__name__}: {exc}")
        return Dogrulama(ok=False, error=f"{type(exc).__name__}: {exc}")
    if not getattr(yanit, "ok", False):
        return Dogrulama(ok=False, error=str(getattr(yanit, "error", ""))[:200])
    sonuc = Dogrulama.coz(getattr(yanit, "text", "") or "")
    if not sonuc.ok:
        logger.debug(f"Doğrulama ayrıştırılamadı: {sonuc.raw[:120]}")
    return sonuc


_DUZELTME_ISTEMI = """Yanıtında kanıtla desteklenmeyen kısımlar var. Yanıtı
**düzelt**: yalnızca araç çıktısıyla desteklenen bilgileri bırak, desteksiz
somut iddiaları çıkar veya "doğrulanamadı" diye işaretle. Kısa tut.

HEDEF: {hedef}

DESTEKSİZ İDDİALAR:
{eksikler}

ARAÇ ÇIKTILARI:
{kanit}

ESKİ YANIT:
{cevap}

YENİ YANIT (yalnızca metin):
"""


def duzelt(router: Any, hedef: str, cevap: str, eksikler: Sequence[str],
           kanit: str, *, max_tokens: int = 700) -> Optional[str]:
    """Desteksiz yanıt için tek düzeltme turu."""
    if router is None:
        return None
    istem = _DUZELTME_ISTEMI.format(
        hedef=str(hedef)[:300],
        eksikler="\n".join(f"- {e}" for e in list(eksikler)[:5]) or "- (belirsiz)",
        kanit=str(kanit)[:2000], cevap=str(cevap)[:2000])
    try:
        from neurocore.agent.llm import LLMMessage

        yanit = router.chat([LLMMessage("user", istem)], max_tokens=max_tokens,
                            temperature=0.2)
    except Exception as exc:
        logger.debug(f"Düzeltme çağrısı başarısız: {type(exc).__name__}: {exc}")
        return None
    if not getattr(yanit, "ok", False):
        return None
    yeni = str(getattr(yanit, "text", "") or "").strip()
    return yeni or None


# ─── 5) Yardımcılar ────────────────────────────────────────────────────────

def _json_cikar(metin: str) -> Any:
    """
    Metinden **ilk** JSON nesnesini çıkarır.

    ⚠ Modeller JSON'u kod bloğu içinde veya açıklamayla birlikte döndürür;
    düz `json.loads` bu yüzden çoğu zaman başarısız olur.
    """
    metin = str(metin or "").strip()
    if not metin:
        return None
    try:
        return json.loads(metin)
    except json.JSONDecodeError:
        pass
    temiz = re.sub(r"```(?:json)?", "", metin).replace("```", "")
    for bas, son in (("{", "}"), ("[", "]")):
        i, j = temiz.find(bas), temiz.rfind(son)
        if 0 <= i < j:
            try:
                return json.loads(temiz[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ─── 6) Katman sınıfı (sayaçlı) ────────────────────────────────────────────

class CognitionLayer:
    """
    Bilişsel katmanın durum tutan sarmalayıcısı.

    Ajan döngüsü bunu kullanır; sayaçlar `get_stats()` ile raporlanır.
    """

    name = "agent.cognition"

    def __init__(self, *, max_reflect: int = 2, verify: bool = True,
                 reflect_every: int = 8) -> None:
        self.max_reflect = max(0, int(max_reflect))
        self.verify = bool(verify)
        self.reflect_every = max(2, int(reflect_every))
        self._initialized = False
        self._router: Any = None
        self._counter = {
            "reflect_calls": 0, "reflect_replan": 0, "reflect_fail": 0,
            "verify_calls": 0, "verify_unsupported": 0,
            "correct_calls": 0, "correct_ok": 0, "classify": 0,
        }
        self._hata_sayaci: Dict[str, int] = {}

    def initialize(self, config: Any = None, event_bus: Any = None, *,
                   router: Any = None) -> None:
        """Router'ı bağlar (idempotent)."""
        if router is not None:
            self._router = router
        self._initialized = True

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._counter,
            "initialized": self._initialized,
            "max_reflect": self.max_reflect,
            "verify": self.verify,
            "reflect_every": self.reflect_every,
            "hata_turleri": dict(self._hata_sayaci),
        }

    # ── hata ──
    def siniflandir(self, metin: str) -> HataTuru:
        tur = siniflandir(metin)
        self._counter["classify"] += 1
        self._hata_sayaci[tur.value] = self._hata_sayaci.get(tur.value, 0) + 1
        return tur

    # ── yansıma ──
    def yansit(self, hedef: str, durum: str,
               last_error: str = "") -> Yansima:
        self._counter["reflect_calls"] += 1
        sonuc = yansit(self._router, hedef, durum, last_error)
        if not sonuc.ok:
            self._counter["reflect_fail"] += 1
        elif sonuc.verdict == "yeniden_planla":
            self._counter["reflect_replan"] += 1
        return sonuc

    # ── doğrulama ──
    def dogrula(self, hedef: str, cevap: str, kanit: str) -> Dogrulama:
        self._counter["verify_calls"] += 1
        sonuc = dogrula(self._router, hedef, cevap, kanit)
        if sonuc.ok and not sonuc.supported:
            self._counter["verify_unsupported"] += 1
        return sonuc

    def duzelt(self, hedef: str, cevap: str, eksikler: Sequence[str],
               kanit: str) -> Optional[str]:
        self._counter["correct_calls"] += 1
        yeni = duzelt(self._router, hedef, cevap, eksikler, kanit)
        if yeni:
            self._counter["correct_ok"] += 1
        return yeni
