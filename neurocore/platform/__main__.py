"""
`python -m neurocore.platform` — platform yeteneklerini JSON yazar.

Kurulum sonrası hızlı doğrulama:

    .venv/bin/python -m neurocore.platform
    .venv\\Scripts\\python.exe -m neurocore.platform

Çıkış kodu: 0 (her durumda) — GPU yoksa `gpu_backend: "cpu"` görünür.

⚠ Windows CI'da ölçüldü: konsol cp1252 iken Türkçe notlar
  `UnicodeEncodeError: 'charmap' codec can't encode characters` üretiyordu.
  Bu yüzden stdout UTF-8'e çevrilir; olmazsa JSON ASCII-güvenli yazılır.
"""

from __future__ import annotations

import json
import sys

from .tespit import ozellikler


def _stdout_utf8() -> bool:
    """stdout/stderr'i UTF-8'e çevirir; başaramazsa False."""
    tamam = True
    for akis in (sys.stdout, sys.stderr):
        try:
            akis.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            tamam = False
    return tamam


def main() -> int:
    veri = ozellikler()
    _stdout_utf8()
    try:
        json.dump(veri, sys.stdout, ensure_ascii=False, indent=2)
    except (UnicodeEncodeError, UnicodeError):
        # Son çare: ASCII-güvenli (Türkçe karakterler \uXXXX olarak)
        json.dump(veri, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
