"""
Model Payı Ölçümü — "yerel model % kaç kullanıldı?"
====================================================
Bu araç **çalışan uygulamanın** canlı sayaçlarını ve günlüğünü okur;
tahmin yürütmez, ölçülmüş sayıları raporlar.

Ölçülen eksenler
----------------
```
1. ÇAĞRI      : bulut LLM çağrısı vs yerel model çıkarım çağrısı
2. TOKEN      : bulut prompt+completion vs yerel eğitim/çıkarım token'ı
3. SÜRE       : bulut toplam süre vs yerel eğitim+sıralama süresi
4. İŞ         : araç çağrılarının kaçı yerelde, kaçı komşu düğümde
```

⚠ Tanım: "yerel model" = `models/neurocore` (40M, sıfırdan eğitilmiş).
Ajanın **beynini** bulut LLM yapar (plan, araç seçimi, cevap metni).
Yerel model: sıralama/seçim + eğitim + bazı çeviri işleri.

Kullanım
--------
    python tools/model_payi.py
    python tools/model_payi.py --dakika 60      # son 60 dakika
    python tools/model_payi.py --json           # yalnız JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

LOG = Path(".neurocore/logs/neurocore.log")
RAPOR = Path("_model_payi.json")

#: ⚠ Günlük satır biçimi: [2026-10-28 21:41:17] [INFO    ] [modül] mesaj
SATIR = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
                   r"\[(\w+)\s*\]\s+\[([^\]]+)\]\s+(.*)$")

#: Eğitim özeti: "Eğitim tamamlandı: 33/33 adım, kayıp 1.9663 → 1.4087,
#:                263,344 token, 91.3s (completed)"
EGITIM = re.compile(
    r"Eğitim tamamlandı:\s*(\d+)/(\d+)\s*adım,\s*kayıp\s*([\d.]+)\s*→\s*"
    r"([\d.]+),\s*([\d,]+)\s*token,\s*([\d.]+)s")

#: Yerel sıralama: "Yerel sıralama: 3 aday, en iyi #2 (fark 0.493), 488ms"
SIRALAMA = re.compile(r"Yerel sıralama:\s*(\d+)\s*aday.*?(\d+)ms")

#: Düşünme kararı: "Düşünme: 3 aday, kaynak=local, güven=orta, 18307ms"
DUSUNME = re.compile(r"Düşünme:\s*(\d+)\s*aday,\s*kaynak=(\w+).*?(\d+)ms")


@dataclass
class Olcum:
    """Ölçüm sonuçları (hepsi ölçülmüş değer)."""

    #: bulut
    bulut_cagri: int = 0
    bulut_hata: int = 0
    bulut_token: int = 0
    bulut_sure_ms: float = 0.0
    #: yerel
    yerel_siralama: int = 0
    yerel_siralama_ms: float = 0.0
    yerel_dusunme: int = 0
    yerel_dusunme_local: int = 0
    yerel_dusunme_ms: float = 0.0
    egitim_kosu: int = 0
    egitim_token: int = 0
    egitim_sure_s: float = 0.0
    egitim_kayip: List[float] = field(default_factory=list)
    #: iş
    arac_cagri: int = 0
    uzak_is: int = 0
    #: genel
    uptime_sn: float = 0.0
    pencere_dk: float = 0.0

    def yuzde(self, pay: float, payda: float) -> str:
        if payda <= 0:
            return "—"
        return f"%{100.0 * pay / payda:.1f}"


def _api(port: int, yol: str) -> Optional[Dict[str, Any]]:
    """Çalışan uygulamadan JSON okur (yoksa None)."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{yol}", timeout=10) as y:
            return json.loads(y.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError):
        return None


def _log_satirlari(dakika: float) -> List[Dict[str, str]]:
    """Günlüğün son `dakika` kadarlık bölümünü ayrıştırır."""
    if not LOG.exists():
        return []
    esik = datetime.now() - timedelta(minutes=dakika)
    out: List[Dict[str, str]] = []
    for satir in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = SATIR.match(satir)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if t < esik:
            continue
        out.append({"t": m.group(1), "seviye": m.group(2),
                    "modul": m.group(3), "mesaj": m.group(4)})
    return out


def olc(port: int, dakika: float) -> Olcum:
    """Canlı sayaçlar + günlükten ölçüm üretir."""
    o = Olcum(pencere_dk=dakika)

    # ── Bulut: canlı sayaç (uygulama açılışından beri) ──
    durum = _api(port, "/api/state")
    if durum:
        llm = durum.get("llm") or {}
        st = llm.get("stats") or {}
        o.bulut_cagri = int(st.get("calls") or 0)
        o.bulut_hata = int(st.get("errors") or 0)
        o.bulut_token = (int(st.get("prompt_tokens") or 0)
                         + int(st.get("completion_tokens") or 0))
        o.bulut_sure_ms = float(st.get("total_ms") or 0.0)
        o.uptime_sn = float(durum.get("uptime_seconds") or 0.0)
        arac = durum.get("tools") or {}
        o.arac_cagri = int(arac.get("calls") or 0)

    # ── Günlük: yerel model izleri ──
    for s in _log_satirlari(dakika):
        mesaj = s["mesaj"]
        modul = s["modul"]

        m = EGITIM.search(mesaj)
        if m:
            o.egitim_kosu += 1
            o.egitim_token += int(m.group(5).replace(",", ""))
            o.egitim_sure_s += float(m.group(6))
            o.egitim_kayip.extend([float(m.group(3)), float(m.group(4))])
            continue

        if modul.endswith("local_rank"):
            m = SIRALAMA.search(mesaj)
            if m:
                o.yerel_siralama += 1
                o.yerel_siralama_ms += float(m.group(2))
            continue

        if modul.endswith("thinking"):
            m = DUSUNME.search(mesaj)
            if m:
                o.yerel_dusunme += 1
                if m.group(2) == "local":
                    o.yerel_dusunme_local += 1
                o.yerel_dusunme_ms += float(m.group(3))
            continue

        # ⚠ Uzak düğümde çalışan iş (adım metni: "[uzak:relay:NNN]")
        if "uzak:relay" in mesaj or "[uzak:" in mesaj:
            o.uzak_is += 1

    return o


def yazdir(o: Olcum) -> None:
    """Ölçümü tablo olarak yazar."""
    yerel_token = o.egitim_token
    toplam_token = o.bulut_token + yerel_token
    yerel_sure = o.egitim_sure_s * 1000.0 + o.yerel_dusunme_ms
    toplam_sure = o.bulut_sure_ms + yerel_sure
    yerel_cagri = o.yerel_siralama

    print("=" * 74)
    print("  MODEL PAYI — ölçülmüş değerler")
    print("=" * 74)
    print(f"  pencere       : son {o.pencere_dk:.0f} dk · "
          f"uygulama açılışından beri {o.uptime_sn / 60:.1f} dk")
    print(f"  bulut modeli  : {o.bulut_cagri} çağrı · {o.bulut_hata} hata")
    print()

    print("  ── 1) ÇAĞRI ──")
    print(f"  {'bulut (DeepSeek/Gemini/Groq)':32} {o.bulut_cagri:6}")
    print(f"  {'yerel çıkarım (sıralama)':32} {yerel_cagri:6}")
    print(f"  {'yerel eğitim koşusu':32} {o.egitim_kosu:6}")
    print(f"  → yerel model payı            "
          f"{o.yuzde(yerel_cagri, o.bulut_cagri + yerel_cagri)}")
    print()

    print("  ── 2) TOKEN ──")
    print(f"  {'bulut (prompt+completion)':32} {o.bulut_token:>10,}")
    print(f"  {'yerel (eğitim)':32} {yerel_token:>10,}")
    print(f"  {'yerel (sıralama, ~tahmin)':32} {'~1.000':>10}")
    print(f"  → yerel token payı            "
          f"{o.yuzde(yerel_token, toplam_token)}")
    print()

    print("  ── 3) SÜRE ──")
    print(f"  {'bulut toplam':32} {o.bulut_sure_ms / 1000:>9.1f} sn")
    print(f"  {'yerel eğitim':32} {o.egitim_sure_s:>9.1f} sn")
    print(f"  {'yerel düşünme/sıralama':32} {o.yerel_dusunme_ms / 1000:>9.1f} sn")
    print(f"  → yerel süre payı             {o.yuzde(yerel_sure, toplam_sure)}")
    print()

    print("  ── 4) İŞ (araç yürütme) ──")
    print(f"  {'araç çağrısı':32} {o.arac_cagri:6}")
    print(f"  {'bunlardan komşu düğümde':32} {o.uzak_is:6}")
    print(f"  → yerelde yürütülen           "
          f"{o.yuzde(o.arac_cagri - o.uzak_is, max(1, o.arac_cagri))}")

    if o.egitim_kayip:
        print()
        print("  ── EĞİTİM ──")
        print(f"  kayıp {o.egitim_kayip[0]:.4f} → {o.egitim_kayip[-1]:.4f} · "
              f"{o.egitim_token:,} token · {o.egitim_sure_s:.1f} sn")

    print("=" * 74)
    print("  ⚠ Ajanın BEYNİ (plan, araç seçimi, cevap) bulut modelidir.")
    print("    Yerel model: sıralama/seçim + eğitim + çeviri.")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description="Model payı ölçümü")
    ap.add_argument("--port", type=int, default=8791,
                    help="Çalışan uygulamanın portu")
    ap.add_argument("--dakika", type=float, default=120.0,
                    help="Günlükte kaç dakika geriye bakılsın")
    ap.add_argument("--json", action="store_true", help="Yalnız JSON yaz")
    args = ap.parse_args()

    o = olc(args.port, args.dakika)
    veri = {
        "pencere_dk": o.pencere_dk, "uptime_sn": o.uptime_sn,
        "bulut": {"cagri": o.bulut_cagri, "hata": o.bulut_hata,
                  "token": o.bulut_token, "sure_ms": o.bulut_sure_ms},
        "yerel": {"siralama": o.yerel_siralama,
                  "siralama_ms": o.yerel_siralama_ms,
                  "dusunme": o.yerel_dusunme,
                  "dusunme_local": o.yerel_dusunme_local,
                  "dusunme_ms": o.yerel_dusunme_ms,
                  "egitim_kosu": o.egitim_kosu,
                  "egitim_token": o.egitim_token,
                  "egitim_sure_s": o.egitim_sure_s},
        "is": {"arac_cagri": o.arac_cagri, "uzak_is": o.uzak_is},
    }
    RAPOR.write_text(json.dumps(veri, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    if args.json:
        print(json.dumps(veri, ensure_ascii=False, indent=1))
    else:
        yazdir(o)
        print(f"\n  rapor → {RAPOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
