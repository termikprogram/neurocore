"""
Refactor parite denetimi — "taşıdım" mı, "sildim" mi?
=====================================================
ÖLÇÜLEN OLAY (TitoWarWeb, player_controller.gd bölme):
```
git diff --numstat : 706 silinen · 35 eklenen   → 671 satır KAYIP
Yeni dosya         : 427 satır (hiç büyümedi)
Headless testler   : geçti! (oyuncu sahnesi çalışmadığı için görülmedi)
```
Ajan "taşıdım" dedi, gerçekte **sildi**. Bu modül, çalışma bittiğinde
değişiklikleri ölçer ve kayıp varsa çalışmayı **bitmiş saymaz**.

Kural
-----
```
Silinen satır > 150  VE  eklenen < silinen × 0.60   →  KAYIP şüphesi
Yeni/değişen dosyalarda eski fonksiyon adları korunmuş mu (parite)
Sıra: KOPYALA → YAPIŞTIR → DOĞRULA → SONRA SİL  (asla tersi)
```

Kullanım
--------
```python
from neurocore.agent.refactor import numstat_denetle, git_numstat
rapor = numstat_denetle(git_numstat("E:/PROJELER/TitoWarWeb"))
if rapor["kayip_suphesi"]:
    ...  # modele "parite kontrolü yap" mesajı
```
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.refactor")

#: Kayıp şüphesi eşikleri
SILME_ESIK = 150          # bu kadar satır silindiyse denetle
ORAN_ESIK = 0.60          # eklenen/silinen oranı bunun altındaysa şüpheli

#: Denetlenecek dosya türleri (kod)
KOD_UZANTI = (".gd", ".py", ".js", ".ts", ".rs", ".go", ".cs", ".c", ".cpp",
              ".h", ".hpp", ".java", ".kt", ".lua", ".rb", ".php", ".sh",
              ".ps1")


def git_numstat(kok: str | Path, zaman_asimi: float = 20.0) -> str:
    """
    Çalışma ağacındaki değişikliklerin `git diff --numstat` çıktısı.

    Args:
        kok: Proje kökü (git deposu)
        zaman_asimi: Alt süreç sınırı (sn)

    Returns:
        Ham çıktı (git yoksa/başarısızsa boş)
    """
    try:
        sonuc = subprocess.run(
            ["git", "-C", str(kok), "diff", "--numstat", "HEAD"],
            capture_output=True, text=True, timeout=zaman_asimi,
            encoding="utf-8", errors="replace")
        if sonuc.returncode == 0:
            return sonuc.stdout
        logger.debug(f"git numstat başarısız: {sonuc.stderr[:120]}")
    except Exception as exc:
        logger.debug(f"git numstat çalıştırılamadı: {exc}")
    return ""


def _ayristir(ham: str) -> List[Dict[str, Any]]:
    """`numstat` satırlarını sözlüklere çevirir (ikili dosyalar `-`)."""
    kayitlar: List[Dict[str, Any]] = []
    for satir in (ham or "").splitlines():
        parca = satir.split("\t")
        if len(parca) < 3:
            continue
        eklenen, silinen, yol = parca[0], parca[1], parca[2]
        try:
            e = int(eklenen)
        except ValueError:
            e = 0
        try:
            s = int(silinen)
        except ValueError:
            s = 0
        kayitlar.append({"yol": yol.strip(), "eklenen": e, "silinen": s})
    return kayitlar


def numstat_denetle(ham: str, *, silme_esik: int = SILME_ESIK,
                    oran_esik: float = ORAN_ESIK) -> Dict[str, Any]:
    """
    Değişiklikleri ölçer ve **kayıp şüphesi** olup olmadığını söyler.

    Args:
        ham: `git diff --numstat` çıktısı
        silme_esik: Bu kadar satır silinirse denetim devreye girer
        oran_esik: eklenen/silinen oranı bunun altındaysa şüpheli

    Returns:
        {"ok", "eklenen", "silinen", "oran", "kayip_suphesi", "dosyalar",
         "silen_dosyalar", "mesaj"}
    """
    kayitlar = _ayristir(ham)
    kod = [k for k in kayitlar if k["yol"].endswith(KOD_UZANTI)]
    eklenen = sum(k["eklenen"] for k in kod)
    silinen = sum(k["silinen"] for k in kod)
    oran = (eklenen / silinen) if silinen else 1.0
    supheli = silinen >= silme_esik and oran < oran_esik
    # tamamen silinen dosyalar (yeni içerik yok)
    silenler = [k["yol"] for k in kod
                if k["eklenen"] == 0 and k["silinen"] > 50]
    mesaj = ""
    if supheli:
        mesaj = (f"{silinen} satır silindi, yalnız {eklenen} satır eklendi "
                 f"(oran {oran:.2f} < {oran_esik}). PARİTE KONTROLÜ GEREKLİ: "
                 "taşınan kodun yeni yerde birebir karşılığı olduğunu doğrula "
                 "(`code.outline` ile iki dosyayı karşılaştır, `git diff` ile "
                 "silinen blokları incele).")
    elif silenler:
        mesaj = ("Tamamen silinen dosyalar: " + ", ".join(silenler[:5])
                 + " — bilinçli mi?")
    return {
        "ok": True,
        "eklenen": eklenen,
        "silinen": silinen,
        "oran": round(oran, 3),
        "kayip_suphesi": supheli,
        "dosyalar": kod[:20],
        "silen_dosyalar": silenler[:10],
        "mesaj": mesaj,
    }


def parite_kontrol(kok: str | Path, eski: str, yeni: str) -> Dict[str, Any]:
    """
    İki dosyadaki **fonksiyon adlarını** karşılaştırır (parite).

    Args:
        kok: Proje kökü
        eski: Eski dosya yolu (göreli)
        yeni: Yeni dosya yolu (göreli)

    Returns:
        {"ok", "eski_sayi", "yeni_sayi", "eksik", "fazla", "mesaj"}
    """
    # ⚠ (b) Dil bazlı sembol kalıpları: GDScript + Python + JS/TS + Rust/Go
    desenler = [
        re.compile(r"^\s*func\s+([A-Za-z_]\w*)", re.MULTILINE),          # GDScript
        re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.MULTILINE),  # Python
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
                   re.MULTILINE),                                          # JS/TS
        re.compile(r"^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*"
                   r"(?:async\s*)?\(", re.MULTILINE),                     # JS ok fonksiyon
        re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE),  # Rust
        re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", re.MULTILINE),   # Go
    ]
    sonuc: Dict[str, Any] = {"ok": True, "eski_sayi": 0, "yeni_sayi": 0,
                             "eksik": [], "fazla": [], "mesaj": ""}
    try:
        a = Path(kok) / eski
        b = Path(kok) / yeni
        def _adlar(yol: Path) -> set:
            metin = yol.read_text(encoding="utf-8", errors="replace")
            toplam: set = set()
            for d in desenler:
                toplam.update(d.findall(metin))
            return toplam

        eski_adlar = _adlar(a)
        yeni_adlar = _adlar(b)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    sonuc["eski_sayi"] = len(eski_adlar)
    sonuc["yeni_sayi"] = len(yeni_adlar)
    sonuc["eksik"] = sorted(eski_adlar - yeni_adlar)[:40]
    sonuc["fazla"] = sorted(yeni_adlar - eski_adlar)[:40]
    if sonuc["eksik"]:
        sonuc["mesaj"] = (f"{len(sonuc['eksik'])} fonksiyon yeni dosyada yok: "
                          + ", ".join(sonuc["eksik"][:8]))
    return sonuc
