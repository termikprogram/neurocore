"""
NeuroCore Masaüstü Uygulaması — Giriş Noktası
=============================================
`python -m neurocore.app` çağrısını `neurocore.app.launcher.main`'e yönlendirir.
"""

from __future__ import annotations

import sys

from neurocore.app.launcher import main

if __name__ == "__main__":
    sys.exit(main())
