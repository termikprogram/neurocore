"""
Kaynak Yönetişimi — Parçalama Bütçesi
======================================
Bir işi **parçalara bölerek** çalıştırırken her parçanın ne kadar kaynak
kullanabileceğini hesaplar.

Neden bu modül?
    Internet Download Manager bir dosyayı parçalara böler ve boşta olan
    bant genişliğini doyurmadan kullanır. Aynı mantık CPU/GPU/RAM için
    gerekir:

        * Tek bir ağır iş kaynağı doyurur → bilgisayar kilitlenir.
        * Küçük parçalar hâlinde çalıştırmak, **boştaki kapasiteyi**
          kullanırken kullanıcıya yer bırakır.
        * Parçalar bağımsızsa **paralel** çalışabilir (IDM'in çoklu
          bağlantısı gibi).

İki kural (kullanıcı isteği):

    1. **Bütçe**  : boştakinin %90'ı kullanılır.
       CPU %50 doluysa → %45'ine kadar; sistem %90 boşsa → %81'ine kadar.
    2. **Parça**  : her parça toplam kapasitenin küçük bir dilimidir
       (varsayılan %5). Böylece **birden fazla parça paralel** çalışabilir
       ve tek parça kaynağı doyurmaz.

Örnek (kullanıcının senaryosu):

    Sistem  : 4 GB VRAM, 4 GB RAM, 4 çekirdek CPU
    Kullanım: %10 dolu
    Bütçe   : boştakinin %90'ı = %81 → 4 GB × 0,81 ≈ 3,3 GB
    Parça   : toplamın %5'i = 200 MB
    Sonuç   : 3,3 GB / 200 MB = **16 parça paralel** çalışabilir

Sistem küçüldükçe parça sayısı kendiliğinden düşer; kod değişmez.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("core.governor")

#: Kaynak adları (sabit sıra — raporlar okunur kalsın)
RESOURCES: Sequence[str] = ("cpu", "ram", "vram")


@dataclass
class ResourceSnapshot:
    """Anlık sistem durumu (0–100 yüzde + toplam kapasite)."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    vram_percent: float = 0.0
    ram_total_mb: float = 0.0
    vram_total_mb: float = 0.0
    cpu_cores: int = 1
    taken_at: float = field(default_factory=time.time)

    def percent(self, kaynak: str) -> float:
        """Kaynağın kullanım yüzdesi."""
        return float({
            "cpu": self.cpu_percent,
            "ram": self.ram_percent,
            "vram": self.vram_percent,
        }.get(kaynak, 0.0))

    def total_mb(self, kaynak: str) -> float:
        """Kaynağın toplam kapasitesi (MB; CPU için 0)."""
        if kaynak == "ram":
            return float(self.ram_total_mb)
        if kaynak == "vram":
            return float(self.vram_total_mb)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_percent": round(self.ram_percent, 1),
            "vram_percent": round(self.vram_percent, 1),
            "ram_total_mb": round(self.ram_total_mb),
            "vram_total_mb": round(self.vram_total_mb),
            "cpu_cores": self.cpu_cores,
        }

    @classmethod
    def tiny(cls) -> "ResourceSnapshot":
        """Küçük sistem profili (test ve planlama için): 4 GB RAM/VRAM."""
        return cls(ram_total_mb=4096.0, vram_total_mb=4096.0, cpu_cores=4)

    @classmethod
    def live(cls, *, cpu_cores: Optional[int] = None) -> "ResourceSnapshot":
        """
        Gerçek sistemi okur.

        `psutil` yoksa veya GPU bilgisi alınamazsa ilgili alan 0 kalır;
        yönetişim yine çalışır (o kaynağı sınırsız sayar).
        """
        cpu = ram_pct = vram_pct = 0.0
        ram_mb = vram_mb = 0.0
        cores = int(cpu_cores or 0)
        try:
            import psutil

            cpu = float(psutil.cpu_percent(interval=None))
            bellek = psutil.virtual_memory()
            ram_pct = float(bellek.percent)
            ram_mb = float(bellek.total) / 1024**2
            cores = cores or int(psutil.cpu_count(logical=True) or 1)
        except Exception as exc:
            logger.debug(f"psutil okunamadı: {exc}")
        try:
            import torch

            if torch.cuda.is_available():
                serbest, toplam = torch.cuda.mem_get_info()
                vram_mb = float(toplam) / 1024**2
                vram_pct = (toplam - serbest) / max(1.0, toplam) * 100.0
        except Exception as exc:
            logger.debug(f"CUDA okunamadı: {exc}")
        return cls(cpu_percent=cpu, ram_percent=ram_pct, vram_percent=vram_pct,
                   ram_total_mb=ram_mb, vram_total_mb=vram_mb,
                   cpu_cores=max(1, cores))


@dataclass
class GovernorPolicy:
    """Parçalama politikası."""
    #: Boştaki kaynağın ne kadarı kullanılır (kullanıcı kuralı: %90)
    budget_ratio: float = 0.90
    #: Her parçanın toplam kapasitedeki payı (IDM benzeri küçük dilim).
    #: Aynı zamanda **eşzamanlılık hedefidir**: `1 / chunk_ratio` parça
    #: paralel çalışır. Küçültmek daha çok ama daha küçük parça demektir.
    chunk_ratio: float = 0.05
    #: Tek parçanın altına düşemeyeceği taban (MB) — çok küçük parça anlamsız
    min_chunk_mb: float = 32.0
    #: CPU için taban: parça başına en az bu kadar çekirdek payı
    min_chunk_cpu_percent: float = 2.0
    #: Sert tavan: hiçbir durumda aşılmaz (kilitlenme koruması)
    hard_ceiling: float = 0.85
    #: Aynı anda en fazla kaç parça (aşırı paralellik fayda getirmez)
    max_parallel: int = 16

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "budget_ratio": self.budget_ratio,
            "chunk_ratio": self.chunk_ratio,
            "min_chunk_mb": self.min_chunk_mb,
            "min_chunk_cpu_percent": self.min_chunk_cpu_percent,
            "hard_ceiling": self.hard_ceiling,
            "max_parallel": self.max_parallel,
        }


@dataclass
class ChunkBudget:
    """
    Bir parçanın kullanabileceği kaynak.

    `parallel` alanı **kaç parçanın aynı anda** çalışabileceğini söyler;
    planlayıcı bu sayıyı eşzamanlılık sınırı olarak kullanır.
    """
    parallel: int = 1
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    vram_mb: float = 0.0
    #: Bütçeyi hangi kaynak kısıtlıyor (tanı için)
    limiting: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "parallel": self.parallel,
            "cpu_percent": round(self.cpu_percent, 2),
            "ram_mb": round(self.ram_mb, 1),
            "vram_mb": round(self.vram_mb, 1),
            "limiting": self.limiting,
            "reason": self.reason,
        }


class ResourceGovernor:
    """
    Parçalama bütçesini hesaplar.

    Kullanım:
        gov = ResourceGovernor()
        snapshot = ResourceSnapshot.live()      # veya .tiny() test için
        butce = gov.plan(snapshot)
        print(butce.parallel, butce.vram_mb)    # kaç parça, her biri ne kadar
    """

    name = "core.governor"

    def __init__(self, policy: Optional[GovernorPolicy] = None) -> None:
        self.policy = policy or GovernorPolicy()
        self._counter: Dict[str, int] = {"plans": 0, "blocked": 0, "degraded": 0}

    # ─── Hesaplama ──────────────────────────────────────────────────────

    def spendable_percent(self, used_percent: float) -> float:
        """
        Kullanılabilir kaynak yüzdesi (kullanıcı kuralı).

        `boştakinin %90'ı`, sert tavanla sınırlanır:

            CPU %50 dolu → boş %50 → %45 kullanılabilir
            CPU %10 dolu → boş %90 → %81 kullanılabilir
            CPU %95 dolu → boş %5  → %4,5 ama tavan %85 → %4,5
        """
        free = max(0.0, 100.0 - float(used_percent))
        return min(free * self.policy.budget_ratio,
                   self.policy.hard_ceiling * 100.0)

    def plan(self, snapshot: Optional[ResourceSnapshot] = None) -> ChunkBudget:
        """
        Anlık duruma göre parça bütçesini hesaplar.

        Args:
            snapshot: Sistem durumu (None = canlı oku)

        Returns:
            `ChunkBudget` — parça boyutu ve eşzamanlılık sınırı
        """
        self._counter["plans"] += 1
        anlik = snapshot or ResourceSnapshot.live()
        politika = self.policy

        pay: Dict[str, float] = {}
        for kaynak in RESOURCES:
            pay[kaynak] = self.spendable_percent(anlik.percent(kaynak))

        # Parça başına düşen dilim (toplam kapasitenin `chunk_ratio`'su)
        #
        # Tutarlılık: üç kaynakta da **aynı** pay kullanılır
        # (`pay × chunk_ratio`), çünkü tarihsel olarak `pay / chunk_ratio`
        # yazılmıştı ve CPU'yu RAM/VRAM'den farklı ölçekliyordu. Ölçüm:
        # CPU %45 bütçede parça %4,5 olmalıydı, %2,5 çıkıyordu.
        #
        # Sonuç: `parallel = pay / (pay × chunk_ratio) = 1 / chunk_ratio`.
        # Yani **parça boyu ile eşzamanlılık hedefi tutarlıdır**: parçayı
        # küçültmek paralelliği artırır, büyütmek azaltır. Planlayıcı bu
        # sayıyı hedef alarak parçaları kurar; aksi hâlde iri parçalar
        # birleşip paralelliği yok eder (ölçüldü: 8 bağımsız adım tek
        # parçaya düşüp seri çalıştı).
        parca: Dict[str, float] = {}
        parca["cpu"] = max(politika.min_chunk_cpu_percent,
                           pay["cpu"] * politika.chunk_ratio)
        for kaynak in ("ram", "vram"):
            toplam = anlik.total_mb(kaynak)
            if toplam > 0:
                parca[kaynak] = max(politika.min_chunk_mb,
                                    toplam * politika.chunk_ratio)
            else:
                parca[kaynak] = 0.0

        # Kaç parça paralel çalışabilir? En kısıtlayıcı kaynak belirler.
        adaylar: List[tuple] = []
        for kaynak in RESOURCES:
            if parca[kaynak] <= 0:
                continue
            if kaynak == "cpu":
                # CPU payı yüzde; parça yüzdesi de yüzde
                adet = int(pay["cpu"] / parca["cpu"]) if parca["cpu"] > 0 else 0
            else:
                toplam = anlik.total_mb(kaynak)
                butce_mb = toplam * (pay[kaynak] / 100.0)
                adet = int(butce_mb / parca[kaynak]) if parca[kaynak] > 0 else 0
            adaylar.append((adet, kaynak))

        if not adaylar:
            self._counter["degraded"] += 1
            return ChunkBudget(
                parallel=1, limiting="bilinmiyor",
                reason="sistem bilgisi okunamadı — tek parça ile ilerlenir")

        kısıt = min(adaylar, key=lambda x: x[0])
        paralel = max(1, min(int(kısıt[0]), politika.max_parallel))
        if paralel <= 1 and pay.get(kısıt[1], 0.0) <= 0.0:
            self._counter["blocked"] += 1

        kaynak_adlari = {"cpu": "CPU", "ram": "RAM", "vram": "VRAM"}
        gerekce = (
            f"{kaynak_adlari.get(kısıt[1], kısıt[1])} kısıtlıyor: "
            f"boştaki %{100.0 - anlik.percent(kısıt[1]):.0f} → "
            f"bütçe %{pay.get(kısıt[1], 0.0):.1f}, "
            f"parça {parca['cpu' if kısıt[1] == 'cpu' else kısıt[1]]:.0f}"
            f"{'%' if kısıt[1] == 'cpu' else 'MB'}"
        )
        return ChunkBudget(
            parallel=paralel, cpu_percent=parca["cpu"],
            ram_mb=parca["ram"], vram_mb=parca["vram"],
            limiting=kısıt[1], reason=gerekce,
        )

    def fits(self, need: Dict[str, float],
             budget: Optional[ChunkBudget] = None,
             snapshot: Optional[ResourceSnapshot] = None) -> bool:
        """
        Verilen ihtiyaç **tek parçaya** sığıyor mu?

        Args:
            need: {"cpu": yüzde, "ram_mb": mb, "vram_mb": mb}
            budget: Önceden hesaplanmış bütçe (None = hesapla)
            snapshot: Sistem durumu

        Returns:
            Sığıyorsa True
        """
        b = budget or self.plan(snapshot)
        if need.get("cpu", 0.0) > b.cpu_percent:
            return False
        if need.get("ram_mb", 0.0) > b.ram_mb:
            return False
        if need.get("vram_mb", 0.0) > b.vram_mb:
            return False
        return True

    def split_into_chunks(self, total_units: int,
                          units_per_chunk: int) -> List[tuple]:
        """
        Bir işi **sıralı parça aralıklarına** böler (IDM mantığı).

        Args:
            total_units: Toplam iş birimi (adım, örnek, bayt…)
            units_per_chunk: Parça başına birim (en az 1)

        Returns:
            `[(başlangıç, bitiş), …]` — bitiş hariç (Python dilimi gibi)
        """
        toplam = max(0, int(total_units))
        boy = max(1, int(units_per_chunk))
        return [(bas, min(bas + boy, toplam))
                for bas in range(0, toplam, boy)]

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve politika."""
        return {**self._counter, "policy": self.policy.to_dict()}


__all__ = [
    "ResourceGovernor", "GovernorPolicy", "ResourceSnapshot", "ChunkBudget",
    "RESOURCES",
]
