"""
Ağ katmanı — makineler arası ortak havuz.
============================================

`peer_pool` : SSH tabanlı keşif + iş paylaşımı (yerel devralma ile).
"""

from __future__ import annotations

from neurocore.net.peer_pool import Komsu, PeerPool

__all__ = ["PeerPool", "Komsu"]
