"""
`python -m neurocore.platform` — platform yeteneklerini JSON yazar.

Kurulum sonrası hızlı doğrulama:

    .venv/bin/python -m neurocore.platform
    .venv\\Scripts\\python.exe -m neurocore.platform

Çıkış kodu: 0 (her durumda) — GPU yoksa `gpu_backend: "cpu"` görünür.
"""

from __future__ import annotations

import json
import sys

from .tespit import ozellikler


def main() -> int:
    veri = ozellikler()
    json.dump(veri, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
