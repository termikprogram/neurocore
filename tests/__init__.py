"""
NeuroCore test paketi.

Testleri çalıştırmak için::

    python -m unittest discover -s tests -v
    python -m unittest tests.test_knowledge -v
"""

from __future__ import annotations

import os
import sys
import warnings

# Testler sırasında uyarı gürültüsünü bastır (NVIDIA sürücü, torch vb.)
warnings.filterwarnings("ignore")

# Proje kökünü yola ekle (paket kurulu olmadan çalışabilsin)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

__all__ = []
