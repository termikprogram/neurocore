"""
Kontrol noktası — yarıda kesilen işi **kaldığı yerden** sürdürme
================================================================
Kullanıcı sorusu: "İşlem yarıda kesildiğinde yapılanları, nerede kaldığını,
hangi adımda kaldığını görüp oradan devam edecek mi?"

Cevap: **evet** — bu modül her çalışmanın durumunu diske yazar:

```text
.neurocore/kontrol.json        son çalışmanın durumu (üzerine yazılır)
.neurocore/kontrol_gecmis.jsonl  tamamlanan/kesilen çalışmaların arşivi
```

Kaydedilenler
-------------
```
gorev          : görev metni
adimlar        : tamamlanan araç adımları (araç + kısa argüman + ok/hata)
dosyalar       : değişen dosyalar
sebep          : "tamamlandı" · "kullanıcı durdurdu" · "süre" · "hata"
puzzle         : iç içe faz ağacı (aktif parça dahil)
olgular        : hedef durumundaki kanıtlar (tekrar iş yapılmasın)
acik_sorular   : cevaplanmamış walkthrough soruları
```

Devam kuralı
------------
```
yeni görev benzerliği >= 0.50  ∧  önceki çalışma tamamlanmamış
        → "KALDIĞIN YER" özeti mesaja eklenir, adımlar tekrarlanmaz
        → puzzle ağacı ve olgular geri yüklenir
```
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.checkpoint")

#: Devam için gereken en düşük görev benzerliği
MIN_BENZERLIK = 0.50

#: Saklanan en fazla adım (dosya şişmesin)
MAX_ADIM = 120


@dataclass
class Kontrol:
    """Bir çalışmanın kaydedilebilir durumu."""

    gorev: str = ""
    workspace: str = ""
    basladi: float = field(default_factory=time.time)
    guncellendi: float = field(default_factory=time.time)
    adimlar: List[Dict[str, Any]] = field(default_factory=list)
    dosyalar: List[str] = field(default_factory=list)
    sebep: str = ""
    tamamlandi: bool = False
    puzzle: Dict[str, Any] = field(default_factory=dict)
    olgular: List[str] = field(default_factory=list)
    acik_sorular: List[str] = field(default_factory=list)

    # ── sorgular ──
    def arac_sayisi(self) -> int:
        return len(self.adimlar)

    def son_adim(self) -> str:
        """En son yapılan işin tek satır özeti."""
        if not self.adimlar:
            return ""
        son = self.adimlar[-1]
        return f"{son.get('tool', '?')} — {son.get('ozet', '')}"[:160]

    def ozet(self, sinir: int = 1400) -> str:
        """
        "Nerede kaldık?" metni — hem kullanıcıya hem modele gösterilir.
        """
        if not self.gorev:
            return ""
        satirlar = [f"KALDIĞIN YER — {self.gorev[:140]}"]
        durum = "tamamlandı" if self.tamamlandi else (
            self.sebep or "yarıda kaldı")
        satirlar.append(f"Durum: {durum} · {self.arac_sayisi()} araç adımı")
        if self.adimlar:
            satirlar.append("")
            satirlar.append("Yapılanlar (son 8):")
            for adim in self.adimlar[-8:]:
                isaret = "✓" if adim.get("ok") else "✗"
                satirlar.append(
                    f"  {isaret} {adim.get('tool', '?')} — "
                    f"{str(adim.get('ozet', ''))[:110]}")
        if self.dosyalar:
            satirlar.append("")
            satirlar.append("Değişen dosyalar: "
                            + ", ".join(self.dosyalar[:12]))
        if self.olgular:
            satirlar.append("")
            satirlar.append("Elde edilen olgular:")
            satirlar.extend(f"  - {o[:120]}" for o in self.olgular[-6:])
        if self.acik_sorular:
            satirlar.append("")
            satirlar.append("Cevaplanmamış sorular:")
            satirlar.extend(f"  ? {s[:120]}" for s in self.acik_sorular[:5])
        if not self.tamamlandi:
            satirlar.append("")
            satirlar.append("Devam et: tamamlanan adımları TEKRARLAMA, "
                            "kaldığın yerden sürdür.")
        return "\n".join(satirlar)[:sinir]

    # ── serileştirme ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gorev": self.gorev[:600],
            "workspace": self.workspace,
            "basladi": self.basladi,
            "guncellendi": self.guncellendi,
            "adimlar": self.adimlar[-MAX_ADIM:],
            "dosyalar": self.dosyalar[:40],
            "sebep": self.sebep[:200],
            "tamamlandi": bool(self.tamamlandi),
            "puzzle": self.puzzle,
            "olgular": self.olgular[:20],
            "acik_sorular": self.acik_sorular[:10],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Kontrol":
        return cls(
            gorev=str(d.get("gorev") or "")[:600],
            workspace=str(d.get("workspace") or ""),
            basladi=float(d.get("basladi") or time.time()),
            guncellendi=float(d.get("guncellendi") or time.time()),
            adimlar=[a for a in (d.get("adimlar") or [])
                     if isinstance(a, dict)][:MAX_ADIM],
            dosyalar=[str(x) for x in (d.get("dosyalar") or [])][:40],
            sebep=str(d.get("sebep") or "")[:200],
            tamamlandi=bool(d.get("tamamlandi")),
            puzzle=d.get("puzzle") if isinstance(d.get("puzzle"), dict) else {},
            olgular=[str(x) for x in (d.get("olgular") or [])][:20],
            acik_sorular=[str(x) for x in (d.get("acik_sorular") or [])][:10],
        )

    # ── kalıcılık ──
    def kaydet(self, yol: Path) -> bool:
        """Atomik yazım: geçici dosya → taşı."""
        try:
            yol.parent.mkdir(parents=True, exist_ok=True)
            gecici = yol.with_suffix(".tmp")
            gecici.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                         indent=1), encoding="utf-8")
            gecici.replace(yol)
            return True
        except OSError as exc:
            logger.debug(f"Kontrol noktası yazılamadı: {exc}")
            return False

    @classmethod
    def yukle(cls, yol: Path) -> Optional["Kontrol"]:
        try:
            veri = json.loads(Path(yol).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(veri) if isinstance(veri, dict) else None

    # ── arşiv ──
    @staticmethod
    def arsivle(kontrol: "Kontrol", yol: Path) -> None:
        """Tamamlanan/kesilen çalışmayı JSONL arşivine ekler."""
        if not kontrol.gorev:
            return
        try:
            yol.parent.mkdir(parents=True, exist_ok=True)
            with yol.open("a", encoding="utf-8") as dosya:
                dosya.write(json.dumps(kontrol.to_dict(), ensure_ascii=False)
                            + "\n")
        except OSError as exc:
            logger.debug(f"Arşiv yazılamadı: {exc}")


def benzer(a: str, b: str) -> float:
    """İki görev metni arasındaki benzerlik (playbook ile aynı ölçü)."""
    try:
        from neurocore.agent.playbooks import benzerlik

        return benzerlik(a, b)
    except Exception:
        return 0.0


#: ⚠ Açık "devam" niyeti: kullanıcı yalnız "devam et" yazdığında görev
#: benzerliği 0 çıkar ve sürdürme tetiklenmezdi. Bu kalıplar benzerliği atlar.
DEVAM_KALIPLARI = (
    "devam", "devam et", "kaldığın yerden", "kaldigi yerden", "sürdür",
    "sur dur", "continue", "resume", "go on", "carry on", "where were we",
)


def devam_istegi(gorev: str) -> bool:
    """Görev metni açıkça "devam" istiyor mu?"""
    kucuk = str(gorev or "").strip().lower()
    if not kucuk or len(kucuk) > 40:
        return False
    return any(kalip in kucuk for kalip in DEVAM_KALIPLARI)


def devam_edilebilir(kontrol: Optional[Kontrol], gorev: str,
                     *, esik: float = MIN_BENZERLIK) -> bool:
    """
    Bu kontrol noktası yeni görev için kullanılabilir mi?

    ⚠ İki yol: (1) görev benzerliği ≥ eşik, (2) açık "devam" isteği.
    """
    if kontrol is None or kontrol.tamamlandi or not kontrol.gorev:
        return False
    if not kontrol.adimlar:
        return False
    if devam_istegi(gorev):
        return True
    return benzer(kontrol.gorev, gorev) >= esik
