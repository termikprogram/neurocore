"""
Bilgi İş Akışı — Ajan Bilmediğinde Öğrenir
===========================================
Ajanın **bilgi boşluğunu fark edip kendi kendine öğrenmesini** sağlar:
kasadan sorar → yoksa öğretmenden öğrenir → kaynağa dayalı yazar → aynı
görev içinde kullanır.

Neden gerekli?
    Ölçülen boşluk: `SelfStudy` (öğretmen–öğrenci döngüsü) ve
    `LearningRecorder` vardı ama **ajana bağlı değildi**. Ajanın erişebildiği
    65 araç arasında `knowledge.*` (arama) vardı, `learn.*` **yoktu**.

    Sonuç: ajan bilmediği bir konuyla karşılaştığında ya tahmin ediyor ya
    "bilmiyorum" diyordu — **öğrenemiyordu**. Kullanıcının sorusu tam buydu:
    "bir iş verildiğinde bilmiyorsa öğrenerek yapacak mı?"

Üç araç (hepsi `read` güvenliğinde — kasa yazımı içeride):

    learn.topic      — Konuyu öğren: kasa → öğretmen → kaynak → kasa
    knowledge.ask    — Önce kasaya sor; yeterliyse öğrenmeye gerek yok
    learn.status     — Ne öğrenildi, hangi turda, hangi kaynakla

Akış (tek görev içinde):

    1. Ajan görevi alır: "Godot 4.8'de CharacterBody3D nasıl kullanılır?"
    2. `knowledge.ask` → kasa boş döner (bilmiyor)
    3. `learn.topic` → öğretmen soru üretir, **kaynak indirilir**,
       kaynağa dayalı cevap alınır, kasa yazılır
    4. Ajan öğrendiği bilgiyle görevi **aynı oturumda** tamamlar

Dürüstlük kuralları:
    * Öğrenme **kaynağa dayalıdır**; öğretmenin hafızası doğrulanmaz, adres
      verilir ve sayfa **kendimiz indiririz**.
    * Kaynak bulunamazsa tur **başarısız** sayılır, kasa kirletilmez.
    * Yerel modelin kendi cevabı asla eğitim hedefi olmaz (zehirlenme).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from neurocore.agent.tools import Safety, ToolDef
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.knowledge_work")

#: Kasadan bu kadar **alakalı** sonuç yeterli sayılır (yeniden öğrenmeye
#: gerek yok)
ENOUGH_RESULTS = 2

#: Bir sonucun "alakalı" sayılması için sorgu terimlerinin en az bu kadarı
#: sonuçta geçmelidir.
#:
#: ⚠ Ölçülen sorun: arama, sorgunun terimleri **geçmese de** yüksek skorlu
#: sonuç döndürüyordu ("zzz qqq bilinmeyen konu" → "İşletim Sistemi — IPC",
#: skor 0,729). Bu yüzden ajan "kasa biliyor" sanıp öğrenmiyordu. Skor
#: tek başına yeterli değildir; **terim örtüşmesi** de aranır.
MIN_TERM_COVERAGE = 0.30

#: Alakalılık için gereken en düşük arama skoru
MIN_SCORE = 0.35

#: Kasa aramasında özet için en büyük gövde uzunluğu
MAX_BODY_CHARS = 1200

#: Alakalılık hesabında yok sayılan kısa/genel terimler (Türkçe + İngilizce)
STOP_WORDS = frozenset({
    "nedir", "nasil", "nasıl", "nicin", "niçin", "hangi", "ile", "icin",
    "için", "bir", "bu", "ve", "veya", "ama", "daha", "en", "cok", "çok",
    "the", "and", "for", "with", "how", "what", "why", "does", "use",
    "using", "kullan", "kullanilir", "kullanılır", "uygulanir", "uygulanır",
})

#: Öğretmen cevabında **yetersizlik itirafı** arayan kalıplar.
#:
#: ⚠ Ölçülen sorun: öğretmen yanlış kaynak önerdiğinde (PyTorch sorusuna
#: NVIDIA CUDA kılavuzu) kendi cevabında "bu kaynakla cevaplanamaz" diyordu
#: ama tur **başarılı** sayılıp kasaya yazılıyordu. Sonuç: kasa değersiz
#: notla doluyor ve model ondan öğreniyor (zehirlenme).
#:
#: Bu itiraf görülürse tur **başarısız** sayılır ve kasaya **yazılmaz**.
INSUFFICIENT_MARKERS = (
    "tam olarak cevaplanamaz",
    "tam olarak cevaplayam",
    "bu kaynakla cevaplanamaz",
    "cevaplanamaz",
    "cevaplayamıyorum",
    "cevaplayamiyorum",
    "yeterli bilgi yok",
    "yeterli bilgi içermiyor",
    "kaynak bu soruyu",
    "kaynaklar bu soruyu",
    "soruyla ilgili değil",
    "soruyla alakasız",
    "soruyla alakasiz",
    "cannot answer",
    "insufficient information",
    "does not contain",
    "not relevant to the question",
)

#: Cevabın değersiz sayılacağı en kısa uzunluk
MIN_ANSWER_CHARS = 80


def _terms(text: str) -> List[str]:
    """
    Metinden **ayırt edici** terimleri çıkarır.

    Kısa kelimeler (≤3 harf) ve genel soru kelimeleri atılır; aksi hâlde
    her sonuç "alakalı" görünür.
    """
    import re

    ham = re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]{4,}", text.lower())
    return [t for t in ham if t not in STOP_WORDS]


def _coverage(query_terms: Sequence[str], text: str) -> float:
    """Sorgu terimlerinin ne kadarı metinde geçiyor (0–1)."""
    if not query_terms:
        return 0.0
    kucuk = text.lower()
    bulunan = sum(1 for t in query_terms if t in kucuk)
    return bulunan / len(query_terms)


def answer_is_useful(answer: str) -> tuple:
    """
    Öğretmen cevabı **kullanılabilir** mi?

    Neden gerekli? Ölçüldü: öğretmen yanlış kaynak önerdiğinde (PyTorch
    sorusuna NVIDIA CUDA kılavuzu) kendi cevabında "bu kaynakla
    cevaplanamaz" diyordu — ama tur **başarılı** sayılıp kasaya yazılıyordu.
    Sonuç: kasa değersiz notla dolar ve model **ondan öğrenir**.

    Args:
        answer: Öğretmen cevabı

    Returns:
        `(yararli, gerekçe)` — `yararli=False` ise kasaya yazılmamalı
    """
    metin = (answer or "").strip()
    if len(metin) < MIN_ANSWER_CHARS:
        return False, f"cevap çok kısa ({len(metin)} karakter)"
    kucuk = metin.lower()
    for kalip in INSUFFICIENT_MARKERS:
        if kalip in kucuk:
            return False, f"öğretmen yetersizlik bildirdi: '{kalip}'"
    return True, ""


@dataclass
class LearnedTopic:
    """Öğrenilen tek konunun kaydı."""
    topic: str
    from_vault: bool = False
    ok: bool = False
    question: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    note_title: str = ""
    error: str = ""
    duration_ms: float = 0.0
    #: Soruyu kim üretti: `local` (yerel model) veya `teacher`
    question_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "topic": self.topic[:200], "from_vault": self.from_vault,
            "ok": self.ok, "question": self.question[:300],
            "answer": self.answer[:MAX_BODY_CHARS],
            "sources": list(self.sources)[:8],
            "note_title": self.note_title[:200],
            "error": self.error[:200],
            "duration_ms": round(self.duration_ms, 1),
            "question_source": self.question_source,
        }


class KnowledgeWorkflow:
    """
    Ajanı öğrenme döngüsüne bağlar.

    Kullanım:
        akis = KnowledgeWorkflow()
        akis.initialize(study=self.study, knowledge=self.knowledge)
        sonuc = akis.learn("Godot'ta CharacterBody3D")
        print(sonuc.answer)
    """

    name = "agent.knowledge_work"

    def __init__(self) -> None:
        self._study: Any = None
        self._knowledge: Any = None
        self._initialized = False
        self._history: List[LearnedTopic] = []
        self._counter: Dict[str, int] = {
            "asks": 0, "vault_hits": 0, "learns": 0, "learn_failures": 0,
            "no_knowledge_source": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, study: Any = None, knowledge: Any = None) -> None:
        """
        İş akışını başlatır (idempotent).

        Args:
            study: `SelfStudy` örneği (öğretmen–öğrenci döngüsü)
            knowledge: `KnowledgeLayer` örneği (kasa)
        """
        if study is not None:
            self._study = study
        if knowledge is not None:
            self._knowledge = knowledge
        if self._initialized:
            return
        self._initialized = True
        logger.info(
            f"Bilgi iş akışı hazır — kasa: "
            f"{'var' if self._knowledge is not None else 'YOK'}, "
            f"öğretmen: {'var' if self._study is not None else 'YOK'}"
        )

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve son öğrenmeler."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "history": [k.to_dict() for k in self._history[-10:]],
            "study_ready": self._study is not None,
            "knowledge_ready": self._knowledge is not None,
        }

    # ─── Kasadan Sorma ──────────────────────────────────────────────────

    def _search_vault(self, query: str, limit: int = 5,
                      *, relevant_only: bool = True) -> List[Any]:
        """
        Kasadan arama yapar ve **alakalılık süzgecinden** geçirir.

        ⚠ Ölçülen sorun: arama altyapısı, sorgunun terimleri sonuçta
        **geçmese de** yüksek skorlu kayıt döndürüyordu:

            "zzz qqq bilinmeyen konu" → "İşletim Sistemi — IPC" (skor 0,729)
            "CharacterBody3D"        → alakasız kasa notları

        Bu yüzden ajan "kasa biliyor" sanıp **öğrenmiyordu**. Süzgeç iki
        ölçüt arar: yeterli skor **ve** yeterli terim örtüşmesi.

        Args:
            query: Arama sorgusu
            limit: En fazla kaç sonuç
            relevant_only: Alakalılık süzgeci uygulansın mı

        Returns:
            Alakalı sonuçlar (alakasızlar elenir)
        """
        if self._knowledge is None:
            return []
        arama = getattr(self._knowledge, "search", None)
        if not callable(arama):
            return []
        try:
            ham = list(arama(query, limit=max(limit * 3, 12)) or [])
        except Exception as exc:
            logger.debug(f"Kasa araması başarısız: {exc}")
            return []
        if not relevant_only:
            return ham[:limit]

        sorgu_terimleri = _terms(query)
        # Ayırt edici terim: sorgunun **en uzun** terimi (en özel olan).
        # ⚠ Ölçüldü: yalnızca ortalama örtüşmeye bakmak yetmiyordu —
        # "zzz qqq bilinmeyen konu" sorgusunda "konu" kelimesi kasa
        # notlarında geçtiği için sonuç "alakalı" sayılıyordu; oysa
        # "bilinmeyen" hiçbir notta yok. Ajan bu yüzden "biliyorum" sanıp
        # öğrenmiyordu.
        ayirt_edici = max(sorgu_terimleri, key=len) if sorgu_terimleri else ""
        suzulmus: List[Any] = []
        for sonuc in ham:
            metin = self._result_text(sonuc)
            skor = float(getattr(sonuc, "score", 0.0)
                         or (sonuc.get("score", 0.0)
                             if isinstance(sonuc, dict) else 0.0))
            ortusme = _coverage(sorgu_terimleri, metin)
            # Hem ortalama örtüşme hem **ayırt edici terim** aranır
            if ayirt_edici and ayirt_edici not in metin.lower():
                continue
            if skor >= MIN_SCORE and ortusme >= MIN_TERM_COVERAGE:
                suzulmus.append(sonuc)
            if len(suzulmus) >= limit:
                break
        if not suzulmus and ham:
            logger.debug(
                f"Kasa araması {len(ham)} sonuç verdi ama hiçbiri alakalı "
                f"değil (sorgu: {query[:60]!r}) — öğrenmeye gidilecek"
            )
        return suzulmus

    @staticmethod
    def _result_text(sonuc: Any) -> str:
        """
        Arama sonucundan okunur metin çıkarır.

        Başlık, özet **ve** gövde birleştirilir: yalnızca başlığa bakmak
        terim örtüşmesini olduğundan düşük gösterir ve alakalı sonuçları
        elerdi.
        """
        if isinstance(sonuc, dict):
            parcalar = [sonuc.get("title"), sonuc.get("excerpt"),
                        sonuc.get("snippet"), sonuc.get("body")]
        else:
            parcalar = [getattr(sonuc, "title", None),
                        getattr(sonuc, "excerpt", None),
                        getattr(sonuc, "snippet", None),
                        getattr(sonuc, "body", None)]
        metin = "\n".join(str(p) for p in parcalar if p).strip()
        return metin[:MAX_BODY_CHARS]

    def ask(self, question: str, *, allow_learn: bool = True,
            limit: int = 5) -> Dict[str, Any]:
        """
        Bilgi sorusu: **önce kasaya** bakar; yeterli değilse öğrenir.

        Args:
            question: Soru veya konu
            allow_learn: Kasa boşsa öğrenmeye izin ver
            limit: Kaç kasa sonucu okunsun

        Returns:
            {"ok", "source", "answer", "sources", "learned"}
            `source`: `vault` (kasadan) | `learned` (yeni öğrenildi) | `none`
        """
        self._counter["asks"] += 1
        soru = str(question or "").strip()
        if not soru:
            return {"ok": False, "error": "soru boş", "source": "none"}

        # ── 1) Kasadan sor ──
        sonuclar = self._search_vault(soru, limit=limit)
        if len(sonuclar) >= ENOUGH_RESULTS:
            self._counter["vault_hits"] += 1
            return {
                "ok": True,
                "source": "vault",
                "answer": "\n\n---\n\n".join(self._result_text(s)
                                             for s in sonuclar[:limit]),
                "sources": ["vault"],
                "learned": None,
                "note": "bilgi zaten kasada",
            }

        # ── 2) Kasada yok → öğren ──
        if not allow_learn:
            return {
                "ok": bool(sonuclar),
                "source": "vault" if sonuclar else "none",
                "answer": "\n\n---\n\n".join(self._result_text(s)
                                             for s in sonuclar),
                "sources": ["vault"] if sonuclar else [],
                "learned": None,
                "note": "kasa yetersiz, öğrenme kapalı",
            }

        ogrenme = self.learn(soru)
        if ogrenme.ok:
            return {
                "ok": True,
                "source": "learned",
                "answer": ogrenme.answer,
                "sources": ogrenme.sources,
                "learned": ogrenme.to_dict(),
                "note": "bilgi yeni öğrenildi ve kasaya yazıldı",
            }
        # Öğrenme başarısız: kasadaki kırıntı varsa onu ver, yoksa açıkça bildir
        return {
            "ok": bool(sonuclar),
            "source": "vault" if sonuclar else "none",
            "answer": ("\n\n---\n\n".join(self._result_text(s)
                                          for s in sonuclar)
                       if sonuclar else ""),
            "sources": ["vault"] if sonuclar else [],
            "learned": ogrenme.to_dict(),
            "error": ogrenme.error or "öğrenme başarısız",
            "note": ("öğrenme başarısız — doğrulanmış kaynak bulunamadı; "
                     "tahmin yürütmek yerine kullanıcıya bildirilmeli"),
        }

    # ─── Öğrenme ────────────────────────────────────────────────────────

    def learn(self, topic: str) -> LearnedTopic:
        """
        Bir konuyu **öğretmenden öğrenir** ve kasaya yazar.

        Akış `SelfStudy.study()` içindedir: önce yerel model soru üretir
        (üretemezse öğretmen), öğretmenin önerdiği kaynak **indirilir**,
        kaynağa dayalı cevap alınır, kasa yazılır.

        Args:
            topic: Öğrenilecek konu

        Returns:
            `LearnedTopic` — cevap, kaynaklar, başarı durumu
        """
        konu = str(topic or "").strip()
        kayit = LearnedTopic(topic=konu)
        if not konu:
            kayit.error = "konu boş"
            return kayit
        if self._study is None:
            self._counter["no_knowledge_source"] += 1
            kayit.error = ("öğretmen bağlı değil — öğrenme yapılamıyor "
                           "(API anahtarı gerekli)")
            self._history.append(kayit)
            return kayit

        basladi = time.perf_counter()
        try:
            tur = self._study.study(konu)
        except Exception as exc:
            self._counter["learn_failures"] += 1
            kayit.error = f"{type(exc).__name__}: {exc}"
            kayit.duration_ms = (time.perf_counter() - basladi) * 1000
            self._history.append(kayit)
            logger.warning(f"Öğrenme çöktü ({konu[:40]}): {exc}")
            return kayit

        kayit.duration_ms = (time.perf_counter() - basladi) * 1000
        kayit.ok = bool(getattr(tur, "ok", False))
        kayit.question = str(getattr(tur, "question", "") or "")
        kayit.answer = str(getattr(tur, "answer", "") or "")
        kayit.note_title = str(getattr(tur, "note_title", "") or "")
        kayit.error = str(getattr(tur, "error", "") or "")
        kayit.question_source = str(getattr(tur, "question_source", "") or "")
        try:
            kayit.sources = [str(s.get("url", "") if isinstance(s, dict) else s)
                             for s in (getattr(tur, "sources", None) or [])]
        except Exception:
            kayit.sources = []

        # ── Kalite kapısı ──
        #
        # ⚠ Ölçüldü: öğretmen yanlış kaynak önerdiğinde kendi cevabında
        # "bu kaynakla cevaplanamaz" diyordu ama tur başarılı sayılıp
        # kasaya yazılıyordu. Böyle bir not kasa için **zehir**dir: model
        # ondan öğrenir. İtiraf görülürse tur başarısız sayılır.
        if kayit.ok:
            yararli, gerekce = answer_is_useful(kayit.answer)
            if not yararli:
                kayit.ok = False
                kayit.error = f"cevap kullanılabilir değil: {gerekce}"
                logger.warning(
                    f"Öğrenme reddedildi ({konu[:40]}): {gerekce} "
                    f"— kasaya yazılmadı"
                )

        if kayit.ok:
            self._counter["learns"] += 1
            logger.info(
                f"Öğrenildi: {konu[:50]} — {len(kayit.sources)} kaynak, "
                f"{kayit.duration_ms:.0f}ms, soru: {kayit.question_source}"
            )
        else:
            self._counter["learn_failures"] += 1
            logger.warning(f"Öğrenilemedi ({konu[:40]}): {kayit.error[:100]}")
        self._history.append(kayit)
        return kayit

    def learned(self, limit: int = 20) -> List[LearnedTopic]:
        """Son öğrenilenler."""
        return self._history[-max(1, limit):]

    # ─── Araçlar ────────────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """
        Ajan için bilgi araçları.

        Üçü de `read` güvenliğindedir: kasaya yazım **içeride** olur ve
        doğrulanmış kaynağa dayanır. `write` güvenliği istemek, Denetimli
        modda her öğrenme için onay gerektirirdi; bu da "bilmiyorsan öğren"
        akışını kullanılamaz kılardı.
        """
        from neurocore.agent.tools import ToolParameter as P

        def ask_handler(question: str = "", allow_learn: bool = True,
                        limit: int = 5, **_: Any) -> Dict[str, Any]:
            """Kasadan sor, gerekirse öğren."""
            return self.ask(question, allow_learn=bool(allow_learn),
                            limit=int(limit))

        def learn_handler(topic: str = "", **_: Any) -> Dict[str, Any]:
            """Konuyu öğren."""
            kayit = self.learn(topic)
            return kayit.to_dict()

        def search_handler(query: str = "", limit: int = 5,
                           **_: Any) -> Dict[str, Any]:
            """
            İnternette arama yapar.

            ⚠ Ölçülen boşluk: arama altyapısı (`web.search_web`) vardı ama
            **hiçbir yerden çağrılmıyordu** (ölü kod). Ajan internete
            erişemiyordu; yalnızca öğretmenin hafızasından adres
            öneriliyordu ve yanlış adres yanlış bilgi getiriyordu. Bu araç
            o boşluğu kapatır.
            """
            from neurocore.agent.web import search_web

            sorgu = (query or "").strip()
            if not sorgu:
                return {"ok": False, "error": "sorgu boş", "results": []}
            try:
                bulgular = search_web(sorgu, limit=max(1, min(int(limit), 10)))
            except Exception as exc:
                self._counter["search_failures"] = (
                    self._counter.get("search_failures", 0) + 1)
                return {"ok": False, "error": str(exc)[:200], "results": []}
            self._counter["searches"] = self._counter.get("searches", 0) + 1
            return {"ok": bool(bulgular), "query": sorgu,
                    "count": len(bulgular), "results": bulgular}

        def fetch_handler(url: str = "", max_chars: int = 12000,
                          **_: Any) -> Dict[str, Any]:
            """Bir web sayfasını indirir ve metne çevirir."""
            from neurocore.agent.web import fetch

            adres = (url or "").strip()
            if not adres:
                return {"ok": False, "error": "adres boş"}
            try:
                sayfa = fetch(adres, timeout=25)
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:200]}
            if not sayfa.ok:
                return {"ok": False, "error": sayfa.error[:200],
                        "status": sayfa.status}
            self._counter["fetches"] = self._counter.get("fetches", 0) + 1
            sinir = max(500, min(int(max_chars), 40000))
            return {
                "ok": True,
                "url": sayfa.final_url or adres,
                "title": sayfa.title,
                "chars": len(sayfa.text),
                "text": sayfa.text[:sinir],
                "links": sayfa.links[:20],
            }

        return [
            ToolDef(
                name="knowledge.ask",
                description=(
                    "Bir bilgi sorusunu **önce bilgi kasasına** sorar; kasa "
                    "yetersizse ve `allow_learn` açıksa **öğretmenden öğrenir** "
                    "ve kasaya yazar. Bilmediğin bir konuda tahmin yürütmek "
                    "yerine bunu kullan. Dönen `source`: `vault` (zaten "
                    "biliyordu), `learned` (yeni öğrenildi), `none` (bulunamadı)."
                ),
                handler=ask_handler,
                parameters=[
                    P("question", required=True,
                      description="Soru veya konu (ör. 'Godot CharacterBody3D')"),
                    P("allow_learn", "boolean",
                      "Kasa boşsa öğrenmeye izin ver", default=True),
                    P("limit", "integer", "Kaç kasa sonucu okunsun", default=5),
                ],
                safety=Safety.READ,
                category="knowledge",
                returns="cevap, kaynak türü (vault/learned), kaynak adresleri",
                tags=["bilgi", "öğrenme", "kasa", "araştırma"],
            ),
            ToolDef(
                name="learn.topic",
                description=(
                    "Bir konuyu **sıfırdan öğrenir**: öğretmen model kaynak "
                    "adresi önerir, sayfa **indirilir ve doğrulanır**, "
                    "kaynağa dayalı cevap alınır, bilgi kasasına yazılır. "
                    "Kaynak bulunamazsa tur başarısız olur ve kasa "
                    "kirletilmez. Öğrenilen bilgi **aynı görev içinde** "
                    "kullanılabilir."
                ),
                handler=learn_handler,
                parameters=[
                    P("topic", required=True,
                      description="Öğrenilecek konu"),
                ],
                safety=Safety.READ,
                category="knowledge",
                returns="cevap, kaynaklar, soru, soruyu kim üretti",
                tags=["öğrenme", "araştırma", "kaynak"],
            ),
            ToolDef(
                name="learn.status",
                description=(
                    "Ne öğrenildiğini bildirir: son konular, kaynak sayısı, "
                    "başarı oranı, öğretmen bağlı mı. Öğrenme akışını "
                    "kullanmadan önce çağırın."
                ),
                handler=lambda **_: self.get_stats(),
                parameters=[],
                safety=Safety.READ,
                category="knowledge",
                returns="sayaçlar ve son öğrenmeler",
                tags=["öğrenme", "durum"],
            ),
            ToolDef(
                name="web.search",
                description=(
                    "**İnternette gerçek arama** yapar ve sonuç listesini "
                    "döner (başlık + adres + özet). Güncel bilgi, resmî "
                    "belge veya doğrulanabilir kaynak ararken kullan. "
                    "Adresleri **uydurma** — önce ara, sonra `web.fetch` "
                    "ile sayfayı oku. Arama başarısız olursa boş liste döner."
                ),
                handler=search_handler,
                parameters=[
                    P("query", required=True,
                      description="Arama sorgusu (ör. 'Godot 4 RPC nasıl kullanılır')"),
                    P("limit", "integer", "Kaç sonuç", default=5),
                ],
                safety=Safety.READ,
                category="web",
                returns="sonuç listesi: title, url, snippet",
                tags=["internet", "arama", "araştırma"],
            ),
            ToolDef(
                name="web.fetch",
                description=(
                    "Bir web sayfasını **indirir ve okunabilir metne** "
                    "çevirir (HTML etiketleri temizlenir). Belge, API "
                    "referansı veya makale okumak için kullan. Yerel/özel "
                    "ağ adresleri güvenlik gereği reddedilir."
                ),
                handler=fetch_handler,
                parameters=[
                    P("url", required=True, description="Tam adres (https://…)"),
                    P("max_chars", "integer", "En fazla kaç karakter metin",
                      default=12000),
                ],
                safety=Safety.READ,
                category="web",
                returns="title, url, text, links",
                tags=["internet", "okuma", "kaynak"],
            ),
        ]


__all__ = ["KnowledgeWorkflow", "LearnedTopic",
           "ENOUGH_RESULTS", "MAX_BODY_CHARS"]
