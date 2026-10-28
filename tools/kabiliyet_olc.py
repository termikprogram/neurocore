"""
Kabiliyet Ölçüm Seti — 10 sabit kontrol (ağ YOK)
=================================================
Amaç: yapılan değişikliğin yeteneği **gerçekten** artırıp artırmadığını
sayıyla göstermek. Her kontrol deterministiktir; LLM/SSH/HTTP çağrısı yoktur.

Kullanım:
    python tools/kabiliyet_olc.py            # tablo
    python tools/kabiliyet_olc.py --json     # ham JSON

Çıktı: konsol tablosu + `_kabiliyet.json`

Skor: geçen / 10 · her kontrol 1 puan · <1 sn
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")

RAPOR = Path("_kabiliyet.json")
TOPLAM = 10


# ─── Kontroller ─────────────────────────────────────────────────────────────

def k01_hata_taksonomisi() -> Tuple[bool, str]:
    """6 hata örneği doğru sınıflanmalı."""
    from neurocore.agent.cognition import HataTuru, siniflandir

    ornekler = {
        "Permission denied": HataTuru.IZIN,
        "HTTP 429 hız sınırı": HataTuru.AG,
        "CUDA out of memory": HataTuru.KAYNAK,
        "SyntaxError: invalid syntax": HataTuru.SOZDIZIMI,
        "TypeError: missing required argument": HataTuru.ARGUMAN,
        "FileNotFoundError: not found": HataTuru.BULUNAMADI,
    }
    yanlis = [m for m, t in ornekler.items() if siniflandir(m) != t]
    return (not yanlis), f"{len(ornekler) - len(yanlis)}/{len(ornekler)}"


def k02_kurtarma_talimati() -> Tuple[bool, str]:
    """Her hata türünün ayrı ve dolu talimatı olmalı."""
    from neurocore.agent.cognition import HataTuru, kurtarma_talimati

    metinler = {kurtarma_talimati(t) for t in HataTuru
                if t != HataTuru.YOK}
    en_kisa = min((len(kurtarma_talimati(t)) for t in HataTuru
                   if t != HataTuru.YOK), default=0)
    return (len(metinler) >= 7 and en_kisa > 20), f"{len(metinler)} farklı tür"


def k03_hedef_durumu() -> Tuple[bool, str]:
    """Tekrar eleme + üst sınır + özet kırpma."""
    from neurocore.agent.cognition import HedefDurumu

    h = HedefDurumu(hedef="görev")
    h.olgu_ekle("aynı")
    h.olgu_ekle("aynı")
    for i in range(30):
        h.olgu_ekle(f"olgu {i}")
    sinir_ok = len(h.facts) <= h.MAX_FACT and len(h.ozet(sinir=300)) <= 300
    return (len(h.facts) == h.MAX_FACT and sinir_ok), f"{len(h.facts)} olgu"


def k04_yansima_cozumu() -> Tuple[bool, str]:
    """Düz / kod bloklu / bozuk JSON ayrıştırma."""
    from neurocore.agent.cognition import Yansima

    a = Yansima.coz('{"verdict": "yeniden_planla", "missing": ["x"]}')
    b = Yansima.coz('```json\n{"verdict": "cevapla"}\n```')
    c = Yansima.coz("json yok")
    d = Yansima.coz('{"verdict": "saçmalık"}')
    ok = (a.ok and a.verdict == "yeniden_planla" and b.ok
          and b.verdict == "cevapla" and not c.ok and d.verdict == "devam")
    return ok, f"a={a.verdict} b={b.verdict} c_ok={c.ok} d={d.verdict}"


def k05_dogrulama_cozumu() -> Tuple[bool, str]:
    """Destekli/desteksiz + puan kırpma + string bool."""
    from neurocore.agent.cognition import Dogrulama

    a = Dogrulama.coz('{"supported": true, "score": 0.9}')
    b = Dogrulama.coz('{"supported": false, "score": 0.2, "unsupported": ["iddia"]}')
    c = Dogrulama.coz('{"supported": "true", "score": 7.5}')
    ok = a.supported and not b.supported and c.supported and c.score <= 1.0
    return ok, f"a={a.supported} b={b.supported} c_puan={c.score}"


def k06_hiyerarsi() -> Tuple[bool, str]:
    """Geçerli plan çözülür, liste dışı araç atılır."""
    from neurocore.agent.hierarchy import Hiyerarsi, planla
    from neurocore.agent.llm import LLMMessage  # noqa: F401

    metin = json.dumps({"goal": "ağaç geliştir", "subgoals": [
        {"title": "incele", "steps": [{"tool": "uydurma.arac"},
                                      {"tool": "asset.katalog"}],
         "check": "katalog"},
        {"title": "üret", "steps": [{"tool": "asset.uret"}], "check": "dosya"},
    ]}, ensure_ascii=False)

    class _R:
        def chat(self, *a: Any, **k: Any) -> Any:
            return type("Y", (), {"ok": True, "text": metin, "error": ""})()

    h = planla(_R(), "lowpoly ağaçları geliştir ve yeni ağaçlar üret",
               araclar=["asset.katalog", "asset.uret"])
    ok = (h.ok and len(h.alt_hedefler) == 2
          and h.araclar() == ["asset.katalog", "asset.uret"])
    return ok, f"alt hedef={len(h.alt_hedefler)} araç={h.araclar()}"


def k07_playbook() -> Tuple[bool, str]:
    """Kaydet → benzer görevde bul → kullanım say."""
    from neurocore.agent.playbooks import PlaybookStore, benzerlik

    st = PlaybookStore(Path(tempfile.mkdtemp(prefix="nc-k-")) / "pb.jsonl")
    pb = st.kaydet("lowpoly ağaç üret",
                   [{"tool": "asset.uret"}, {"tool": "asset.onizleme"}], 0.85)
    bulunan = st.bul("lowpoly ağaçları geliştir")
    st.kullanim_artir(pb.id)
    skor = benzerlik("lowpoly ağaç üret", "lowpoly ağaçları geliştir")
    ok = (pb is not None and bool(bulunan) and skor >= 0.35
          and st.istatistik()["toplam_kullanim"] == 1)
    return ok, f"benzerlik={skor:.2f} bulundu={bool(bulunan)}"


def k08_baglam_sikistirma() -> Tuple[bool, str]:
    """Uzun geçmiş özete indirilir; sistem + son mesajlar korunur."""
    from neurocore.agent.cognition import HedefDurumu
    from neurocore.agent.llm import LLMMessage
    from neurocore.agent.loop import AgentLoop

    loop = AgentLoop()
    loop._workspace = "."
    loop._config = None
    loop._event_bus = None
    loop._hedef = HedefDurumu(hedef="uzun")
    loop._hedef.olgu_ekle("kanıt 42")
    mesajlar = [LLMMessage("system", "sistem")]
    mesajlar += [LLMMessage("user", f"adım {i}") for i in range(30)]
    once = len(mesajlar)
    loop._baglami_sikistir(mesajlar)
    ok = (len(mesajlar) < once and mesajlar[0].content == "sistem"
          and "ÖZET" in mesajlar[1].content and loop._counter[
              "context_compressed"] == 1)
    return ok, f"{once} → {len(mesajlar)} mesaj"


def k09_arac_secimi() -> Tuple[bool, str]:
    """Araştırma araçları her görevde seçilebilir (çekirdek)."""
    from neurocore.agent.loop import CORE_TOOLS, select_tools

    class _Kayit:
        def names(self) -> List[str]:
            return ["web.search", "web.fetch", "knowledge.ask",
                    "code.read_file", "shell.run", "asset.uret"]

        def has(self, ad: str) -> bool:
            return ad in self.names()

    secili = select_tools("DXGI hatasını araştır", _Kayit(), max_tools=12)
    arastirma = [t for t in secili if t in ("web.search", "web.fetch",
                                           "knowledge.ask")]
    ok = len(arastirma) == 3 and all(
        a in CORE_TOOLS for a in ("web.search", "web.fetch", "knowledge.ask"))
    return ok, f"araştırma={arastirma}"


def k10_kanit_ve_olgu() -> Tuple[bool, str]:
    """Kanıt metni + tek satır olgu çıkarımı."""
    from neurocore.agent.loop import AgentRun, AgentStep, _kisa_olgu
    from neurocore.agent.tools import ToolResult

    run = AgentRun(task="t")
    run.steps = [
        AgentStep(index=0, kind="tool", tool="code.read_file",
                  result="dosya: 42 satır"),
        AgentStep(index=1, kind="plan", text="plan"),
    ]
    from neurocore.agent.loop import AgentLoop

    loop = AgentLoop()
    kanit = loop._kanit_metni(run)
    olgu = _kisa_olgu(ToolResult(ok=True, tool="code.read_file",
                                 data={"path": "a.txt"}))
    ok = ("42 satır" in kanit and "plan" not in kanit
          and "a.txt" in olgu)
    return ok, f"kanıt uzunluğu={len(kanit)} olgu={olgu[:28]}"


KONTROLLER: List[Tuple[str, str, Callable[[], Tuple[bool, str]]]] = [
    ("K01", "hata taksonomisi", k01_hata_taksonomisi),
    ("K02", "kurtarma talimatı", k02_kurtarma_talimati),
    ("K03", "hedef durumu", k03_hedef_durumu),
    ("K04", "yansıma çözümü", k04_yansima_cozumu),
    ("K05", "doğrulama çözümü", k05_dogrulama_cozumu),
    ("K06", "hiyerarşik plan", k06_hiyerarsi),
    ("K07", "playbook", k07_playbook),
    ("K08", "bağlam sıkıştırma", k08_baglam_sikistirma),
    ("K09", "araç seçimi", k09_arac_secimi),
    ("K10", "kanıt/olgu", k10_kanit_ve_olgu),
]


def main() -> int:
    print("=" * 74)
    print("  KABİLİYET ÖLÇÜMÜ — 10 kontrol · ağ yok")
    print("=" * 74)

    sonuclar: List[Dict[str, Any]] = []
    gecen = 0
    basla = time.perf_counter()
    for kod, ad, islev in KONTROLLER:
        t0 = time.perf_counter()
        try:
            ok, ayrinti = islev()
        except Exception as exc:
            ok, ayrinti = False, f"{type(exc).__name__}: {exc}"[:60]
        sure = (time.perf_counter() - t0) * 1000
        gecen += 1 if ok else 0
        isaret = "✓" if ok else "✗"
        print(f"  [{isaret}] {kod} {ad:24} {ayrinti:38} {sure:6.1f} ms")
        sonuclar.append({"kod": kod, "ad": ad, "ok": bool(ok),
                         "ayrinti": ayrinti, "ms": round(sure, 1)})

    toplam_sure = (time.perf_counter() - basla) * 1000
    skor = round(100 * gecen / TOPLAM, 1)
    print("\n" + "=" * 74)
    print(f"  SKOR: {gecen}/{TOPLAM}  (%{skor}) · toplam {toplam_sure:.0f} ms")
    print("=" * 74)

    RAPOR.write_text(json.dumps({
        "skor": skor, "gecen": gecen, "toplam": TOPLAM,
        "sure_ms": round(toplam_sure, 1), "kontroller": sonuclar,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  rapor → {RAPOR}")
    return 0 if gecen == TOPLAM else 1


if __name__ == "__main__":
    raise SystemExit(main())
