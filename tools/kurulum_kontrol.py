"""
Kurulum Kontrolü — yeni makinede çalıştırılacak tek komut
===========================================================
Ne kontrol eder?
----------------
```
1. Python sürümü (>= 3.10)
2. Depo dosyaları (neurocore/, requirements.txt)
3. Zorunlu paketler (torch, fastapi, uvicorn, psutil, …)
4. CUDA + GPU (ad, VRAM, bf16, sürücü)
5. RAM ve disk
6. SSH anahtarı + sunucuya bağlantı + sunucunun gördüğü IP
7. Araç kayıt defteri (kaç araç kuruldu)
8. (isteğe bağlı) Havuza gerçek kayıt denemesi

Kullanım
--------
    python tools/kurulum_kontrol.py
    python tools/kurulum_kontrol.py --kaydet     # gerçek kayıt denemesi

Çıkış kodu: 0 = hazır, 1 = eksik var
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

# ⚠ torch'un `pynvml` uyarısı stderr'e düşüp PowerShell betiğini
# hata gibi gösteriyordu — en başta susturulur.
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

SATIR = "=" * 74
#: (ad, geçti, ayrıntı, kritik) — kritik olmayanlar özeti **düşürmez**
SONUC: List[Tuple[str, bool, str, bool]] = []


def _kontrol(ad: str, kosul: bool, ayrinti: str = "", *,
             kritik: bool = True) -> bool:
    """
    Tek bir kontrolü kaydeder ve yazar.

    ⚠ `kritik=False`: yetenek bilgisi (ör. donanımsal bf16). Eksikliği
    kurulumu **engellemez** (1080 Ti'da bf16 emülasyonu normaldir).
    """
    SONUC.append((ad, bool(kosul), ayrinti, bool(kritik)))
    if kosul:
        isaret, renk = "✓", ""
    elif kritik:
        isaret, renk = "✗", ""
    else:
        isaret, renk = "•", ""
    print(f"  [{isaret}] {ad:34} {ayrinti}")
    return bool(kosul)


def main() -> int:
    ap = argparse.ArgumentParser(description="Kurulum kontrolü")
    ap.add_argument("--kaydet", action="store_true",
                    help="Havuza gerçek kayıt denemesi de yap")
    args = ap.parse_args()

    print(SATIR)
    print("  NEUROCORE — KURULUM KONTROLÜ")
    print(SATIR)

    # ── 1) Python ──────────────────────────────────────────────────────
    print("\n── 1) PYTHON ──")
    surum = sys.version_info
    _kontrol("Python >= 3.10", surum >= (3, 10),
             f"{surum.major}.{surum.minor}.{surum.micro}")
    _kontrol("64 bit", sys.maxsize > 2 ** 32, sys.platform)

    # ── 2) Depo ────────────────────────────────────────────────────────
    print("\n── 2) DEPO DOSYALARI ──")
    _kontrol("neurocore/ paketi", (_KOK / "neurocore" / "__init__.py").exists(),
             str(_KOK))
    _kontrol("requirements.txt", (_KOK / "requirements.txt").exists())
    _kontrol("tools/ dizini", (_KOK / "tools").is_dir())
    _kontrol("tests/ dizini", (_KOK / "tests").is_dir())

    # ── 3) Paketler ────────────────────────────────────────────────────
    print("\n── 3) PAKETLER ──")
    paketler = [
        ("torch", "torch"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
        ("psutil", "psutil"), ("yaml", "pyyaml"), ("networkx", "networkx"),
        ("PIL", "Pillow"), ("numpy", "numpy"), ("requests", "requests"),
        ("tiktoken", "tiktoken"), ("websockets", "websockets"),
    ]
    eksik: List[str] = []
    for modul, pip_adi in paketler:
        try:
            m = importlib.import_module(modul)
            s = getattr(m, "__version__", "?")
            _kontrol(f"{pip_adi}", True, str(s))
        except Exception:
            eksik.append(pip_adi)
            _kontrol(f"{pip_adi}", False, "YOK")

    # ── 4) CUDA / GPU ──────────────────────────────────────────────────
    print("\n── 4) GPU ──")
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        # ⚠ Backend ayrımı: ROCm (AMD Linux) da `torch.cuda.*` üzerinden
        # görünür; `torch.version.cuda` None olur, `torch.version.hip` dolu.
        hip = getattr(torch.version, "hip", None)
        if hip:
            backend = f"ROCm {hip} (AMD)"
        elif torch.version.cuda:
            backend = f"CUDA {torch.version.cuda} (NVIDIA)"
        else:
            backend = "CPU (GPU hızlandırma yok)"
        _kontrol("GPU backend", cuda or not hip,
                 f"torch {torch.__version__} · {backend}")
        if cuda:
            ad = torch.cuda.get_device_name(0)
            ozellik = torch.cuda.get_device_properties(0)
            toplam = ozellik.total_memory / (1024 ** 3)
            bos = (torch.cuda.mem_get_info()[0]) / (1024 ** 3)
            _kontrol("GPU", True, f"{ad}")
            _kontrol("VRAM >= 6 GB", toplam >= 6.0,
                     f"toplam {toplam:.1f} GB · boş {bos:.1f} GB")
            # ⚠ `is_bf16_supported()` Pascal'da da True döner (emülasyon).
            # Yerel (donanımsal) bf16 yalnız sm_80+ ile gelir.
            sm = (int(ozellik.major), int(ozellik.minor))
            yerel_bf16 = sm[0] >= 8
            _kontrol("bf16 (donanımsal)", yerel_bf16,
                     f"sm_{sm[0]}{sm[1]}"
                     + ("" if yerel_bf16 else " · emülasyon → fp16/fp32 seç"),
                     kritik=False)
            if toplam < 9.0:
                print("      ⚠ 8 GB sınıfı: eğitimde `--batch 8 --seq 128` "
                      "ve `--amp-yok` önerilir")
    except Exception as exc:
        _kontrol("torch/CUDA", False, f"{type(exc).__name__}: {exc}")

    # ── 5) RAM / disk ──────────────────────────────────────────────────
    print("\n── 5) RAM / DİSK ──")
    try:
        import psutil

        vm = psutil.virtual_memory()
        _kontrol("RAM >= 8 GB", vm.total / (1024 ** 3) >= 8.0,
                 f"toplam {vm.total / (1024 ** 3):.1f} GB · "
                 f"boş {vm.available / (1024 ** 3):.1f} GB")
        disk = psutil.disk_usage(str(_KOK))
        _kontrol("Disk >= 10 GB boş", disk.free / (1024 ** 3) >= 10.0,
                 f"boş {disk.free / (1024 ** 3):.1f} GB")
    except Exception as exc:
        _kontrol("psutil", False, str(exc))

    # ── 6) SSH / havuz ─────────────────────────────────────────────────
    print("\n── 6) SSH / ORTAK HAVUZ ──")
    # ⚠ Gömülü sır yok: ortam → yerel `peer.json`
    from neurocore.net.peer_pool import kume_ayarlari

    _ayar = kume_ayarlari()
    anahtar = (os.environ.get("NC_SSH_KEY") or _ayar.get("ssh_key") or "")
    anahtar_var = Path(anahtar).exists()
    _kontrol("SSH anahtarı", anahtar_var, anahtar)
    if anahtar_var:
        try:
            from neurocore.net.peer_pool import PeerPool

            havuz = PeerPool(anahtar=anahtar)
            havuz.initialize(data_path=".neurocore")
            ip = havuz.genel_ip()
            _kontrol("Sunucuya bağlantı", bool(ip),
                     f"sunucunun gördüğü IP: {ip or 'YOK'}")
            # ⚠ Küme anahtarı SUNUCUDAN gelir (SSH anahtarından türetilmez):
            # her makine kendi anahtarıyla bağlanır, sır sunucuda durur.
            tok = havuz.token()
            _kontrol("Küme anahtarı", bool(tok),
                     f"{tok[:8]}… ({len(tok)} karakter, sunucudan)")
            if tok:
                iz = havuz._ssh("stat -c '%a' ~/neurocore/cluster.token "
                                "2>/dev/null || echo yok")
                mod = (iz.get("cikti") or "").strip()
                _kontrol("Sunucu anahtar dosyası", mod == "600",
                         f"izin {mod} (600 olmalı)", kritik=False)
            if args.kaydet:
                r = havuz.register(port=0, caps=["kurulum"])
                _kontrol("Havuza kayıt", bool(r.get("ok")),
                         f"düğüm {r.get('node_id', '')[:12]} · "
                         f"havuzda {r.get('komsu', '?')} düğüm")
                havuz.birak()
            havuz.shutdown()
        except Exception as exc:
            _kontrol("Havuz", False, f"{type(exc).__name__}: {exc}")

    # ── 7) Araç kayıt defteri ──────────────────────────────────────────
    print("\n── 7) ARAÇLAR ──")
    try:
        import logging
        import warnings

        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
        from neurocore.agent.loop import CORE_TOOLS, AgentLoop  # noqa: F401
        from neurocore.agent.tools import ToolRegistry

        kayit = ToolRegistry()
        _kontrol("ToolRegistry", True, f"{len(kayit)} araç (boş)")
        _kontrol("Çekirdek araç listesi", len(CORE_TOOLS) >= 25,
                 f"{len(CORE_TOOLS)} araç")
        for gerekli in ("web.search", "web.fetch", "knowledge.ask"):
            _kontrol(f"  {gerekli}", gerekli in CORE_TOOLS)
        from neurocore.net.peer_pool import IZINLI_ARACLAR

        _kontrol("Uzaktan izinli araç", len(IZINLI_ARACLAR) >= 10,
                 f"{len(IZINLI_ARACLAR)} araç (yalnız okuma)")
    except Exception as exc:
        _kontrol("Araç katmanı", False, f"{type(exc).__name__}: {exc}")

    # ── 8) Uygulama modülü ─────────────────────────────────────────────
    print("\n── 8) UYGULAMA ──")
    try:
        from neurocore.app.server import HAS_FASTAPI, PEER_IZINLI_ARACLAR

        _kontrol("FastAPI", bool(HAS_FASTAPI))
        _kontrol("peer/job izin listesi", len(PEER_IZINLI_ARACLAR) >= 10,
                 f"{len(PEER_IZINLI_ARACLAR)} araç")
        _kontrol("__main__ girişi", (_KOK / "neurocore" / "app" /
                                     "__main__.py").exists())
    except Exception as exc:
        _kontrol("Uygulama", False, f"{type(exc).__name__}: {exc}")

    # ── ÖZET ───────────────────────────────────────────────────────────
    basarili = sum(1 for _, ok, _, _ in SONUC if ok)
    toplam = len(SONUC)
    print("\n" + SATIR)
    print(f"  SONUÇ: {basarili}/{toplam} kontrol geçti")
    if eksik:
        print(f"  Eksik paketler: {', '.join(eksik)}")
        print(f"  Kurulum: pip install -r requirements.txt")
    # ⚠ Yalnız **kritik** başarısızlıklar kurulumu engeller
    kritik = [ad for ad, ok, _, k in SONUC if not ok and k]
    if kritik:
        print(f"  ⚠ Başarısız: {', '.join(kritik[:8])}")
    print("  HAZIR" if not kritik else "  EKSİK VAR — yukarıya bak")
    print(SATIR)

    # JSON rapor (uzaktan incelenebilsin)
    try:
        Path("_kurulum_kontrol.json").write_text(
            json.dumps({"basarili": basarili, "toplam": toplam,
                        "kontroller": [{"ad": a, "ok": o, "ayrinti": b,
                                        "kritik": k}
                                       for a, o, b, k in SONUC]},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError:
        pass
    return 0 if not kritik else 1


if __name__ == "__main__":
    raise SystemExit(main())
