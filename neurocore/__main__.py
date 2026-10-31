"""
NeuroCore modül giriş noktası.

`python -m neurocore ...` çağrısını CLI'ya yönlendirir.
"""

from __future__ import annotations

import sys

from neurocore.cli import main

if __name__ == "__main__":
    sys.exit(main())
