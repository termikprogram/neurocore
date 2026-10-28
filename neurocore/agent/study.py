"""
NeuroCore Kendi Kendini Geliştirme (Öğretmen–Öğrenci Döngüsü)
==============================================================
Yerel modelin, **API modellerinden öğretmen olarak** yararlanarak kendini
geliştirmesi.

Neden bu modül var?
    Yerel model (208M) bulut modellerinin genel yeteneğine ulaşamaz ama
    **projeye özel** ve **doğrulanmış** bilgiyi öğrenebilir. Eksik olan şey
    bilgi kaynağıydı: ajan yalnızca kullanıcının görevlerinden öğreniyordu.
    Bu modül üç kaynağı ekler:

        1. **Kendi hataları** — başarısız çalışmalardan ders çıkarma
        2. **Bilgi boşlukları** — kasadaki zayıf/eksik konular
        3. **Gerçek kaynaklar** — öğretmen modelin önerdiği resmî
           dokümantasyon, **indirilip doğrulanarak**

Öğrenme döngüsü::

    Soru üret (teknik: ilk ilkeler / 5 neden / karşılaştırma / hata analizi)
        │
        ├─ KASA → bu konuda ne biliyoruz? (yerel bağlam)
        ├─ KAYNAK → öğretmen model resmî adres önerir → İNDİR → doğrula
        │
        ├─ ÖĞRETMEN yanıtı → kaynağa dayalı, atıflı cevap
        ├─ ÖĞRENCİ yanıtı  → yerel model (varsa) aynı soruya cevap verir
        ├─ ELEŞTİRİ       → öğretmen farkı ve hatayı somut yazar
        │
        └─ KASA'ya yaz → "Soru/Cevap/Kaynak/Öğretmen düzeltmesi"

**Zehirlenme koruması:** Öğretmen modelin yanıtı eğitim hedefidir; yerel
modelin **kendi** yanıtı asla hedef olmaz (yalnızca karşılaştırma için
kullanılır). Bu kural `learning.py` ile aynıdır.

**Kaynak nezaketi:** Her tur kotayı ve diski ölçülü kullanır; turlar
arasında bekleme vardır ve öğretmen çağrıları `chat_with_budget` ile
yapılır (düşünme modelleri boş yanıt vermesin).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.study")


# ─── Soru Sorma Teknikleri ──────────────────────────────────────────────────

class QuestionTechnique(str, Enum):
    """
    Soru sorma tekniği.

    Her teknik farklı bir öğrenme eksiğini hedefler; çeşitlilik, tek tip
    soruların bilgiyi daraltmasını önler.
    """
    FIRST_PRINCIPLES = "ilk_ilkeler"
    FIVE_WHYS = "bes_neden"
    COMPARISON = "karsilastirma"
    ERROR_ANALYSIS = "hata_analizi"
    APPLIED = "uygulama"

    @property
    def label(self) -> str:
        """Türkçe etiket."""
        return {
            "ilk_ilkeler": "İlk ilkeler",
            "bes_neden": "Beş neden",
            "karsilastirma": "Karşılaştırma",
            "hata_analizi": "Hata analizi",
            "uygulama": "Uygulama",
        }.get(self.value, self.value)

    @property
    def instruction(self) -> str:
        """Öğretmen modele verilecek teknik yönergesi."""
        return {
            QuestionTechnique.FIRST_PRINCIPLES: (
                "Soruyu **ilk ilkelerden** kur: konunun temelindeki mekanizmayı, "
                "neden böyle tasarlandığını ve hangi varsayımlara dayandığını sor."
            ),
            QuestionTechnique.FIVE_WHYS: (
                "**Beş neden** tekniğini uygula: yüzeydeki belirtiden başlayıp "
                "kök nedene inen zincir kur."
            ),
            QuestionTechnique.COMPARISON: (
                "**Karşılaştırma** yap: alternatif yaklaşımları, ödünleşimlerini "
                "(trade-off) ve hangi durumda hangisinin seçileceğini sor."
            ),
            QuestionTechnique.ERROR_ANALYSIS: (
                "**Hata analizi** yap: bu konuda en sık yapılan hataları, "
                "belirtilerini ve nasıl teşhis edileceğini sor."
            ),
            QuestionTechnique.APPLIED: (
                "**Uygulama** odaklı sor: gerçek bir görevde bu bilgi nasıl "
                "kullanılır, hangi adımlarla uygulanır."
            ),
        }[self]


#: Tekniklerin döngüsel sırası (her turda biri kullanılır)
TECHNIQUE_CYCLE: Sequence[QuestionTechnique] = (
    QuestionTechnique.FIRST_PRINCIPLES,
    QuestionTechnique.ERROR_ANALYSIS,
    QuestionTechnique.APPLIED,
    QuestionTechnique.FIVE_WHYS,
    QuestionTechnique.COMPARISON,
)


# ─── İstemler ───────────────────────────────────────────────────────────────

QUESTION_PROMPT = """Sen bir yazılım mimarı ve öğretmensin. Yerel bir yapay zekâ modelinin
öğrenmesi için **tek bir soru** üret.

Bağlam (yerel modelin bilgi alanı):
{context}

Öğrenilecek konu: {topic}
Soru sorma tekniği: {technique}
{technique_instruction}

Kurallar:
- Soru **tek** ve **net** olsun; birden fazla soru sorma.
- Cevabı kısa bir paragrafla verilebilecek kadar daralt ama yüzeysel olma.
- Türkçe yaz.
- Yalnızca soruyu yaz, başka açıklama ekleme."""

#: Yerel modele soru **ürettiren** istem (soru sormayı öğrenme adımı).
#:
#: Neden yerel model? Kullanıcının istediği tam olarak bu: model önce *neyi
#: bilmediğini* formüle etmeyi öğrensin. Öğretmenin sorusu her zaman daha
#: iyi olur ama o zaman model soru sormayı hiç öğrenmez.
LOCAL_QUESTION_PROMPT = """Sen bir yazılım geliştiricisisin ve bir konuyu öğrenmeye çalışıyorsun.

Konu: {topic}
Soru sorma tekniği: {technique}
{technique_instruction}

Bu konuda **öğrenmek için soracağın tek bir soruyu** yaz.
Bilmediğin noktayı net biçimde hedefle.

Kurallar:
- Tek bir soru yaz, birden fazla sorma.
- Soru işareti ile bitir.
- Yalnızca soruyu yaz; açıklama, giriş cümlesi veya liste ekleme.
- Türkçe yaz."""

#: Yerel modelin **kendi cevabını ve takıldığı noktayı** yazması için istem.
LOCAL_ANSWER_PROMPT = """Aşağıdaki soruyu elinden geldiğince cevapla.

SORU: {question}

Kurallar:
- Kısa ve net yaz (en fazla 6 cümle).
- Emin olmadığın noktayı **açıkça** belirt: "buradan emin değilim".
- Uydurma; bilmiyorsan "bilmiyorum" yaz.
- Türkçe yaz."""

ANSWER_PROMPT = """Aşağıdaki teknik soruyu, **verilen kaynaklara dayanarak** cevapla.

SORU: {question}

KAYNAKLAR:
{sources}

Kurallar:
1. Yalnızca kaynaklarda geçen bilgiye dayan. Kaynakta olmayan bir şeyi
   yazarsan bunu açıkça belirt.
2. Her önemli iddia için kaynak numarası ver: [1], [2] gibi.
3. Somut ol: kod örneği, fonksiyon adı, dosya yolu, adım numarası kullan.
4. En sonda **"Sık yapılan hatalar"** başlığı altında 2-4 madde yaz.
5. Türkçe yaz, markdown kullan.

Cevap:"""

CRITIQUE_PROMPT = """Bir yapay zekâ modelinin cevabını değerlendiriyorsun.

SORU: {question}

DOĞRU CEVAP (kaynaklara dayalı):
{reference}

MODELİN CEVABI:
{student}

Görevin: modelin cevabındaki **eksikleri ve hataları** somut olarak yaz.

Kurallar:
- Yalnızca **gerçekten** yanlış veya eksik olan noktaları belirt; uydurma.
- Her madde şu biçimde olsun: `- YANLIŞ/EKSİK: … | DOĞRUSU: …`
- Model doğru söylemişse `- (hata yok)` yaz.
- En fazla 5 madde. Türkçe yaz."""


# ─── Kayıt ──────────────────────────────────────────────────────────────────

@dataclass
class StudyRound:
    """Tek bir öğrenme turu."""
    at: float
    topic: str = ""
    technique: str = ""
    question: str = ""
    answer: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)
    student_answer: str = ""
    critique: str = ""
    note_title: str = ""
    #: Soruyu **kim** üretti: `local` (yerel model) veya `teacher`.
    #: Yerel model üretim yapamıyorsa öğretmen sorar; bu alan "soru sormayı
    #: öğrenme" ilerlemesini ölçmeyi sağlar.
    question_source: str = ""
    ok: bool = False
    error: str = ""
    duration_ms: float = 0.0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "at": self.at, "topic": self.topic, "technique": self.technique,
            "question": self.question, "question_source": self.question_source,
            "answer": self.answer,
            "sources": [item.get("url", "") for item in self.sources],
            "student_answer": self.student_answer, "critique": self.critique,
            "note_title": self.note_title, "ok": self.ok, "error": self.error,
            "duration_ms": round(self.duration_ms, 1), "tokens": self.tokens,
        }


# ─── Kendi Kendini Geliştirme ───────────────────────────────────────────────

class SelfStudy:
    """
    Öğretmen–öğrenci döngüsüyle kendi kendini geliştirme.

    Kullanım:
        study = SelfStudy()
        study.initialize(llm=router, knowledge=knowledge_layer,
                         workspace="E:/proje", event_bus=bus)
        result = study.study()              # tek tur
        study.start_worker(interval=900)    # arka planda
        print(study.status())

    Args:
        topics: Öncelikli konular (boş = kasadan ve hatalardan türetilir)
        max_sources: Tur başına indirilecek en fazla kaynak
        note_kind: Kasaya yazılacak not türü
    """

    name = "agent.study"

    def __init__(self, *, topics: Optional[List[str]] = None,
                 max_sources: int = 3, note_kind: str = "reference") -> None:
        self._topics: List[str] = list(topics or [])
        self.max_sources = max(1, int(max_sources))
        self.note_kind = note_kind

        self._llm: Any = None
        self._knowledge: Any = None
        self._recorder: Any = None
        self._workspace: str = ""
        self._event_bus: Any = None
        self._initialized = False

        self._rounds: List[StudyRound] = []
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._studying = False
        self._technique_index = 0

        self._counter = {
            "rounds": 0, "ok": 0, "failed": 0, "notes": 0,
            "sources_fetched": 0, "tokens": 0, "critiques": 0,
            #: Kaç turda **yerel model** soruyu üretti (soru sormayı öğrenme)
            "local_questions": 0,
            #: Öğretmenin hiçbir sağlayıcıdan yanıt alamadığı tur sayısı
            "teacher_failures": 0,
            #: Sağlayıcı yedeklemesiyle kurtarılan çağrı sayısı
            "teacher_switches": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, llm: Any = None, knowledge: Any = None,
                   recorder: Any = None, workspace: str = "",
                   event_bus: Any = None,
                   topics: Optional[List[str]] = None) -> None:
        """Döngüyü başlatır (bağımlılıklar enjekte edilir)."""
        if self._initialized:
            return
        self._llm = llm
        self._knowledge = knowledge
        self._recorder = recorder
        self._workspace = workspace or ""
        self._event_bus = event_bus
        if topics:
            self._topics = list(topics)
        self._initialized = True
        logger.info(
            f"Kendi kendini geliştirme hazır — öğretmen: "
            f"{self._teacher_name()}, konu: {len(self._topics)}"
        )

    def shutdown(self) -> None:
        """Arka plan işçisini durdurur."""
        self.stop_worker()
        self._initialized = False
        logger.info(
            f"Kendi kendini geliştirme kapatıldı ({self._counter['rounds']} tur, "
            f"{self._counter['notes']} not)"
        )

    def _teacher_name(self) -> str:
        """Öğretmen modelin adı (günlük ve arayüz için)."""
        if self._llm is None:
            return "yok"
        try:
            status = self._llm.status()
            return f"{status.get('active') or '?'} ({status.get('active_model') or '?'})"
        except Exception:
            return "?"

    # ─── Konu Seçimi ────────────────────────────────────────────────────

    def _next_topic(self) -> str:
        """
        Sıradaki öğrenme konusunu belirler.

        Öncelik sırası:
            1. Kullanıcının verdiği konular (henüz işlenmemiş olanlar)
            2. Ajanın **başarısız** çalışmalarından çıkarılan konular
            3. Kasadaki konu başlıkları (bilgiyi derinleştirmek için)

        Returns:
            Konu metni (boş = konu yok)
        """
        # Yalnızca **başarıyla** işlenen konular atlanır. Başarısız tur
        # (kaynak bulunamadı, öğretmen hatası) konuyu tüketmemelidir; aksi
        # hâlde kullanıcının verdiği konu sessizce kaybolur.
        done = {item.topic for item in self._rounds if item.ok}
        for topic in self._topics:
            if topic not in done:
                return topic

        # Başarısız çalışmalardan ders çıkar
        if self._recorder is not None:
            try:
                for record in self._recorder.records(limit=40):
                    if record.get("ok"):
                        continue
                    task = str(record.get("task") or "").strip()
                    if not task:
                        continue
                    topic = (f"Şu görev neden başarısız oldu ve nasıl "
                             f"düzeltilir: {task[:180]}")
                    if topic not in done:
                        return topic
            except Exception as exc:
                logger.debug(f"Başarısız çalışmalar okunamadı: {exc}")

        # Kasadaki başlıklar → **yeni konu keşfi için tohum**
        #
        # ⚠ Ölçülen sorun: eski kod kasada olan başlığı "{başlık} konusunu
        # derinleştir" diye tekrar soruyordu. Yani **kendi bildiğini** tekrar
        # öğreniyordu; kendi kendine **yeni konu keşfetmiyordu**. Kasa
        # büyüdükçe bu döngü kendi kuyruğunu yerdi.
        tohumlar: List[str] = []
        if self._knowledge is not None:
            try:
                recent = (self._knowledge.recent(limit=15)
                          if hasattr(self._knowledge, "recent") else [])
                for note in recent or []:
                    title = str((note or {}).get("title") or "").strip()
                    if title:
                        tohumlar.append(title)
            except Exception as exc:
                logger.debug(f"Kasa başlıkları okunamadı: {exc}")

        konu = self._generate_topic(tohumlar)
        if konu and konu not in done:
            return konu

        # Son çare: bilinen bir başlığı derinleştir (yine de öğrenme olur)
        for tohum in tohumlar:
            konu = f"{tohum} konusunu derinleştir"
            if konu not in done:
                return konu
        return ""

    def _generate_topic(self, tohumlar: Sequence[str]) -> str:
        """
        **Yeni** bir öğrenme konusu üretir (öğretmen modelden).

        Neden gerekli? Sabit konu listesi tükenince veya boşsa, özerk
        öğrenme durmamalı. Öğretmen modele "henüz bilmediğim, öğrenmeye
        değer ne var?" diye sorulur; cevap **kasa içeriğine dayandırılır**
        ki konu alakasız olmasın.

        Args:
            tohumlar: Kasada bulunan başlıklar (bağlam için)

        Returns:
            Yeni konu metni (üretilemezse boş dize)
        """
        if self._llm is None:
            return ""
        bilinen = "; ".join(t[:80] for t in tohumlar[:12]) or "(kasa boş)"
        istem = (
            "Bir bilgi tabanını genişletmek için **tek bir yeni** öğrenme "
            "konusu öner.\n\n"
            f"Zaten bilinen başlıklar: {bilinen}\n\n"
            "Kurallar:\n"
            "- Yukarıdakilerden FARKLI, henüz kapsanmamış bir konu seç\n"
            "- Teknik, somut ve doğrulanabilir olsun (belgelenebilir)\n"
            "- Yalnızca konuyu yaz; açıklama, madde imi, tırnak ekleme\n"
            "- En fazla 12 kelime\n"
        )
        try:
            from neurocore.agent.llm import LLMMessage

            istek = [LLMMessage("user", istem)]
            if hasattr(self._llm, "chat_with_budget"):
                yanit = self._llm.chat_with_budget(istek, temperature=0.8)
            else:
                yanit = self._llm.chat(istek, max_tokens=200, temperature=0.8)
        except Exception as exc:
            logger.debug(f"Konu üretilemedi: {exc}")
            return ""
        if not yanit.ok:
            return ""

        # Temizle: madde imi, tırnak, numara, fazla satır
        metin = (yanit.text or "").strip().splitlines()[0] if yanit.text else ""
        metin = metin.strip().lstrip("-*•0123456789.) ").strip().strip('"\'')
        if len(metin) < 6 or len(metin) > 160:
            return ""
        self._counter["topics_generated"] = (
            self._counter.get("topics_generated", 0) + 1)
        logger.info(f"Yeni konu üretildi: {metin[:70]}")
        return metin

    def _context(self) -> str:
        """Öğretmen modele verilecek yerel bağlam (kısa tutulur)."""
        parts: List[str] = []
        if self._workspace:
            parts.append(f"Proje: {self._workspace}")
        if self._knowledge is not None:
            try:
                stats = self._knowledge.get_stats() if hasattr(self._knowledge, "get_stats") else {}
                if stats:
                    parts.append(
                        f"Kasadaki not sayısı: {stats.get('notes', stats.get('total', '?'))}"
                    )
            except Exception:
                pass
        if self._rounds:
            parts.append(f"İşlenen konu sayısı: {len(self._rounds)}")
        return "\n".join(parts) or "Bağlam yok."

    # ─── Tek Tur ────────────────────────────────────────────────────────

    def study(self, topic: str = "") -> StudyRound:
        """
        **Tek bir öğrenme turu** yürütür.

        Args:
            topic: Konu (boş = otomatik seçilir)

        Returns:
            `StudyRound`
        """
        if self._llm is None:
            return StudyRound(at=time.time(), topic=topic or "?",
                              technique="", question="",
                              error="öğretmen model yok")

        started = time.perf_counter()
        topic = topic or self._next_topic()
        if not topic:
            round_ = StudyRound(at=time.time(), topic="?", technique="",
                                question="", error="öğrenilecek konu bulunamadı")
            return round_

        technique = TECHNIQUE_CYCLE[self._technique_index % len(TECHNIQUE_CYCLE)]
        self._technique_index += 1
        round_ = StudyRound(at=time.time(), topic=topic, technique=technique.value)
        self._emit("study.round_started", {
            "topic": topic[:140], "technique": technique.value,
            "technique_label": technique.label,
        })

        try:
            # 1) Soru — **önce yerel model sorar** (soru sormayı öğrensin),
            #    üretemezse öğretmen soru üretir.
            round_.question = self._student_question(topic, technique)
            round_.question_source = "local" if round_.question else "teacher"
            if round_.question_source == "local":
                self._counter["local_questions"] += 1
            if not round_.question:
                round_.question = self._make_question(topic, technique)
            if not round_.question:
                round_.error = "soru üretilemedi"
                return self._finish(round_, started)
            self._emit("study.question_ready", {
                "topic": topic[:120], "technique": technique.value,
                "source": round_.question_source,
                "question": round_.question[:200],
            })

            # 2) Kaynakları topla (öğretmen önerisi + gerçek indirme)
            round_.sources = self._gather_sources(round_.question)
            if not round_.sources:
                round_.error = "doğrulanmış kaynak bulunamadı"
                return self._finish(round_, started)

            # 3) Öğretmen cevabı (kaynağa dayalı) — yedekli çağrı
            round_.answer = self._teacher_answer(round_.question, round_.sources)
            if not round_.answer:
                round_.error = ("öğretmen yanıt veremedi — denenen sağlayıcılar "
                                "tükendi")
                return self._finish(round_, started)

            # 4) Öğrenci (yerel model) cevabı **ve** emin olmadığı nokta
            round_.student_answer = self._student_reflection(round_.question)

            # 5) Eleştiri: yerel modelin eksiğini somut yaz
            if round_.student_answer:
                round_.critique = self._critique(round_)
                if round_.critique:
                    self._counter["critiques"] += 1

            # 6) Kasaya yaz (öğretmen bilgisi = eğitim hedefi)
            round_.note_title = self._write_note(round_)
            round_.ok = True
            self._counter["ok"] += 1

        except Exception as exc:
            round_.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Öğrenme turu çöktü: {exc}", exc_info=True)

        return self._finish(round_, started)

    def _finish(self, round_: StudyRound, started: float) -> StudyRound:
        """Turu kaydeder ve olay yayınlar."""
        round_.duration_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self._rounds.append(round_)
            self._counter["rounds"] += 1
            if not round_.ok:
                self._counter["failed"] += 1
        self._emit("study.round_finished", {
            "topic": round_.topic[:140], "ok": round_.ok,
            "technique": round_.technique, "error": round_.error,
            "duration_ms": round_.duration_ms,
            "sources": len(round_.sources),
            "note_title": round_.note_title,
            "question": round_.question[:200],
        })
        logger.info(
            f"Öğrenme turu: {round_.topic[:50]!r} · "
            f"{'başarılı' if round_.ok else 'başarısız: ' + round_.error[:60]} · "
            f"{len(round_.sources)} kaynak · {round_.duration_ms / 1000:.1f} sn"
        )
        return round_

    # ─── Adımlar ────────────────────────────────────────────────────────

    def _call_teacher(self, prompt: str, *, max_tokens: int = 2500) -> str:
        """
        Öğretmen modele çağrı yapar; **başarısız olursa diğer modelleri dener**.

        Neden gerekli? Birden fazla anahtar tanımlı olabilir (DeepSeek, Gemini,
        Groq). Biri kotasını doldurduğunda (`429`) veya geçici olarak
        kullanılamadığında (`503`) tur **çökmemeli**; sıradaki sağlayıcı
        denenir. Böylece öğrenme, tek bir sağlayıcının kotasına bağlı kalmaz.

        Args:
            prompt: Öğretmene gönderilecek istem
            max_tokens: En büyük yanıt uzunluğu

        Returns:
            Yanıt metni (hiçbiri çalışmazsa boş dize)
        """
        from neurocore.agent.llm import LLMMessage

        request = [LLMMessage("user", prompt)]
        attempted: List[str] = []
        # Deneme sayısı: tanımlı sağlayıcı kadar (en fazla 4)
        for attempt in range(4):
            current = self._teacher_name()
            response = self._ask(request, max_tokens=max_tokens)
            if response is not None and response.ok and (response.text or "").strip():
                self._counter["tokens"] += response.total_tokens
                if attempted:
                    logger.info(
                        f"Öğretmen yedeklemesi başarılı: "
                        f"{' → '.join(attempted)} → {current}"
                    )
                return (response.text or "").strip()

            reason = response.error if response is not None else "yanıt yok"
            attempted.append(f"{current}[{(reason or '?')[:40]}]")
            logger.warning(
                f"Öğretmen çağrısı başarısız ({current}): {(reason or '?')[:100]}"
            )

            # Sıradaki sağlayıcıya geç
            if not self._switch_teacher():
                break
            self._counter["teacher_switches"] += 1
            self._emit("study.teacher_switched", {
                "to": self._teacher_name(), "reason": (reason or "")[:160],
                "attempted": attempted,
            })

        logger.warning(
            f"Öğretmen yanıt vermedi — denenenler: {' → '.join(attempted) or 'yok'}"
        )
        self._counter["teacher_failures"] += 1
        return ""

    def _ask(self, request: Sequence[Any], *, max_tokens: int) -> Any:
        """Aktif öğretmene tek çağrı yapar (boş yanıt korumasıyla)."""
        try:
            if hasattr(self._llm, "chat_with_budget"):
                # Düşünme modelleri küçük bütçede boş döner; bütçe artırılır
                return self._llm.chat_with_budget(
                    request, token_ladder=(1500, 4000, 8192), temperature=0.3
                )
            return self._llm.chat(request, max_tokens=max_tokens, temperature=0.3)
        except Exception as exc:
            logger.debug(f"Öğretmen çağrısı çöktü: {exc}")
            return None

    def _switch_teacher(self) -> bool:
        """
        Sıradaki **kullanılabilir** sağlayıcıya geçer.

        Returns:
            Geçiş yapıldıysa True; başka sağlayıcı yoksa False
        """
        if self._llm is None or not hasattr(self._llm, "failover"):
            return False
        try:
            return bool(self._llm.failover("öğretmen model yanıt vermedi"))
        except Exception as exc:
            logger.debug(f"Öğretmen değiştirilemedi: {exc}")
            return False

    # ─── Yerel Modelin Katılımı ─────────────────────────────────────────

    def _ask_local(self, prompt: str, *, max_tokens: int = 700) -> str:
        """
        **Yerel modele** sorar (soru üretmesi veya cevaplaması için).

        Yerel model üretim yapamıyorsa (henüz 800 adıma ulaşmadıysa) boş döner
        ve döngü öğretmen üretimli soruya düşer — böylece öğrenme durmaz ama
        yerel model hazır olduğunda **kendi sorusunu sormaya** başlar.

        Returns:
            Yerel modelin yanıtı (boş = üretemedi)
        """
        engine = getattr(self._recorder, "_engine", None) if self._recorder else None
        if engine is None or getattr(engine, "model", None) is None:
            return ""
        try:
            if not engine.model.can_generate:
                return ""
        except Exception:
            return ""
        try:
            method = getattr(engine, "query", None)
            if not callable(method):
                return ""
            output = method(prompt)
            if isinstance(output, dict):
                text = str(output.get("answer") or output.get("text") or "")
            elif isinstance(output, str):
                text = output
            else:
                text = ""
            return text.strip()[:max_tokens * 4]
        except Exception as exc:
            logger.debug(f"Yerel model yanıt veremedi: {exc}")
            return ""

    def _student_question(self, topic: str, technique: QuestionTechnique) -> str:
        """
        **Yerel modelin kendi sorusunu** üretmesini ister.

        Bu, "soru sormayı öğrenme" adımıdır: model önce neyi bilmediğini
        formüle eder, sonra öğretmen bu soruyu kaynakla yanıtlar.

        Sağlık kontrolü: çok kısa, soru işareti olmayan veya tekrar eden
        çıktılar **reddedilir** (yerel model henüz üretim yapamıyor olabilir);
        bu durumda öğretmen üretimli soruya düşülür.

        Returns:
            Geçerli soru metni veya boş dize
        """
        prompt = LOCAL_QUESTION_PROMPT.format(
            topic=topic, technique=technique.label,
            technique_instruction=technique.instruction,
        )
        text = self._ask_local(prompt)
        if not text:
            return ""

        # İlk anlamlı satırı al (model başlık/etiket ekleyebilir)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        question = ""
        for line in lines[:6]:
            cleaned = re.sub(r"^(soru|question)\s*[:\-]\s*", "", line,
                             flags=re.IGNORECASE).strip()
            if len(cleaned) >= 20:
                question = cleaned
                break
        if len(question) < 20:
            logger.debug("Yerel soru çok kısa — öğretmen sorusuna düşülüyor")
            return ""
        # Soru işareti şart: model bazen cevap veya açıklama yazar. Soru
        # biçiminde olmayan uzun metni kabul etmek, öğretmene **soru yerine
        # cevap** göndermeye yol açardı.
        if "?" not in question:
            logger.debug("Yerel çıktı soru biçiminde değil (soru işareti yok) — reddedildi")
            return ""
        return question[:600]

    def _student_reflection(self, question: str) -> str:
        """
        Yerel modelin soruya **kendi cevabını ve bilmediğini** yazması.

        Öğretmenin eleştirisi bu metne dayanır: model "nerede takıldığını"
        yazarsa geri bildirim **hedefli** olur ("şu noktayı bilmiyorsun")
        yerine genel bir değerlendirme gelmez.
        """
        prompt = LOCAL_ANSWER_PROMPT.format(question=question)
        return self._ask_local(prompt)

    def _make_question(self, topic: str, technique: QuestionTechnique) -> str:
        """Öğretmen modelden tek bir soru üretmesini ister."""
        prompt = QUESTION_PROMPT.format(
            context=self._context(), topic=topic,
            technique=technique.label,
            technique_instruction=technique.instruction,
        )
        text = self._call_teacher(prompt, max_tokens=1200)
        if not text:
            return ""
        # Model bazen başlık/etiket ekler; ilk anlamlı paragrafı al
        text = re.sub(r"^(soru|question)\s*[:\-]\s*", "", text.strip(), flags=re.IGNORECASE)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        question = lines[0] if lines else ""
        if len(question) < 12 and len(lines) > 1:
            question = " ".join(lines[:2])
        return question[:600]

    def _gather_sources(self, question: str) -> List[Dict[str, str]]:
        """
        Soruyu yanıtlayan **gerçek** kaynakları toplar.

        İki kanal kullanılır, **sırayla**:

            1. **Gerçek arama** (`search_web`) — DuckDuckGo'dan gelen
               gerçek sonuçlar. En güvenilir kanaldır çünkü adresler
               öğretmenin hafızasından **değil**, canlı dizinden gelir.
            2. **Öğretmen önerisi** (`propose_sources`) — yalnızca arama
               sonuç vermezse başvurulur.

        ⚠ Ölçülen sorun: eski kod **yalnızca** öğretmen önerisini
        kullanıyordu ve `search_web` **hiç çağrılmıyordu** (ölü kod).
        Öğretmen hafızasından adres uydurduğunda yanlış sayfa indiriliyordu:

            Soru : "PyTorch DataLoader num_workers nasıl kullanılır?"
            Öneri: docs.nvidia.com/cuda/cuda-programming-guide/  ← ALAKASIZ
            Sonuç: "Verilen kaynak bu soruyu tam olarak cevaplayamaz"

        Gerçek arama ile aynı soru doğru adrese gider. Ayrıca her adres
        **indirilir** ve doğrulanır — uydurma adres kasaya girmez.
        """
        from neurocore.agent.web import fetch, propose_sources, search_web

        sources: List[Dict[str, str]] = []
        gorulen: set = set()

        def _ekle(url: str) -> bool:
            """Adresi indirir; başarılıysa listeye ekler."""
            if not url or url in gorulen:
                return False
            gorulen.add(url)
            page = fetch(url, timeout=25)
            if not page.ok:
                logger.info(
                    f"Kaynak indirilemedi: {url[:70]} — {page.error[:50]}")
                return False
            text = page.text[:9000]
            if len(text) < 300:
                # Neredeyse boş sayfa (JS ile çizilen site) — işe yaramaz
                return False
            sources.append({
                "url": page.final_url or url,
                "title": page.title or url,
                "text": text,
            })
            self._counter["sources_fetched"] += 1
            return True

        # ── 1) Gerçek arama (birincil kanal) ──
        try:
            bulgular = search_web(question, limit=self.max_sources + 2)
        except Exception as exc:
            logger.debug(f"Arama başarısız: {exc}")
            bulgular = []
        for bulgu in bulgular:
            if len(sources) >= self.max_sources:
                break
            _ekle(str(bulgu.get("url") or ""))

        # ── 2) Arama yetmezse öğretmen önerisi (yedek kanal) ──
        if len(sources) < self.max_sources:
            try:
                urls = propose_sources(question, self._llm,
                                       limit=self.max_sources + 1)
            except Exception as exc:
                logger.debug(f"Kaynak önerisi başarısız: {exc}")
                urls = []
            for url in urls:
                if len(sources) >= self.max_sources:
                    break
                _ekle(url)

        logger.info(
            f"Kaynak: {len(sources)} doğrulandı "
            f"(arama {len(bulgular)}, öneri yedek)"
        )
        return sources

    def _teacher_answer(self, question: str,
                        sources: List[Dict[str, str]]) -> str:
        """Kaynağa dayalı öğretmen cevabı."""
        blocks: List[str] = []
        for index, source in enumerate(sources, 1):
            blocks.append(
                f"[{index}] {source.get('title', '')}\n"
                f"Adres: {source.get('url', '')}\n"
                f"{source.get('text', '')[:6000]}"
            )
        prompt = ANSWER_PROMPT.format(question=question, sources="\n\n".join(blocks))
        return self._call_teacher(prompt, max_tokens=4000)

    def _critique(self, round_: StudyRound) -> str:
        """Yerel modelin cevabındaki eksikleri somut olarak yazdırır."""
        prompt = CRITIQUE_PROMPT.format(
            question=round_.question,
            reference=round_.answer[:5000],
            student=round_.student_answer[:2500],
        )
        return self._call_teacher(prompt, max_tokens=1500)

    def _write_note(self, round_: StudyRound) -> str:
        """
        Öğrenilen bilgiyi kasaya yazar.

        Not, gelecekte hem ajanın bağlamına girer hem de yerel modelin eğitim
        verisine dönüşür. Başlık `Öğrenildi:` ön ekiyle yazılır ki
        kendiliğinden öğrenilen bilgi ayırt edilebilsin.
        """
        if self._knowledge is None:
            return ""
        title = "Öğrenildi: " + " ".join(round_.topic.split())[:70]
        lines = [
            "## Soru", round_.question, "",
            "## Cevap", round_.answer, "",
        ]
        if round_.critique and "(hata yok)" not in round_.critique:
            lines += ["## Yerel modelin eksikleri", round_.critique, ""]
        if round_.sources:
            lines.append("## Kaynaklar")
            for index, source in enumerate(round_.sources, 1):
                lines.append(f"{index}. [{source.get('title', '')}]({source.get('url', '')})")
            lines.append("")
        lines += [
            "## Künye",
            f"- Teknik: {round_.technique}",
            f"- Öğretmen: {self._teacher_name()}",
            f"- Tarih: {time.strftime('%Y-%m-%d %H:%M', time.localtime(round_.at))}",
        ]
        body = "\n".join(lines)
        try:
            self._knowledge.write(
                title, body, kind=self.note_kind,
                tags=["öğrenildi", "araştırma", round_.technique],
                confidence=0.75, importance=0.7,
                sources=[item.get("url", "") for item in round_.sources],
                dedupe=True,
            )
            self._counter["notes"] += 1
            self._emit("study.note_written", {"title": title, "topic": round_.topic[:120]})
            return title
        except Exception as exc:
            logger.warning(f"Öğrenme notu yazılamadı: {exc}")
            return ""

    # ─── Arka Plan ──────────────────────────────────────────────────────

    def start_worker(self, interval: float = 900.0) -> bool:
        """
        Arka plan öğrenme işçisini başlatır.

        Args:
            interval: İki tur arası en az süre (saniye). Varsayılan 15 dakika:
            ücretsiz API kotalarını tüketmemek ve diski şişirmemek için.

        Returns:
            İşçi başlatıldıysa True
        """
        if self._worker is not None and self._worker.is_alive():
            return True
        if self._llm is None:
            logger.warning("Öğretmen model yok — kendi kendine öğrenme başlatılamadı")
            return False
        self._stop_worker.clear()

        def loop() -> None:
            """Tur döngüsü."""
            while not self._stop_worker.wait(interval):
                if self._studying:
                    continue
                try:
                    self._studying = True
                    self.study()
                except Exception as exc:
                    logger.debug(f"Öğrenme işçisi hatası: {exc}")
                finally:
                    self._studying = False

        self._worker = threading.Thread(target=loop, name="NeuroCoreSelfStudy",
                                        daemon=True)
        self._worker.start()
        logger.info(f"Kendi kendine öğrenme başladı (aralık {interval:.0f} sn)")
        return True

    def stop_worker(self) -> None:
        """Arka plan işçisini durdurur."""
        self._stop_worker.set()
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    @property
    def is_studying(self) -> bool:
        """Şu anda tur çalışıyor mu?"""
        return self._studying

    # ─── Durum ──────────────────────────────────────────────────────────

    def rounds(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Son öğrenme turları."""
        with self._lock:
            items = list(self._rounds)
        return [item.to_dict() for item in items[-limit:]]

    def status(self) -> Dict[str, Any]:
        """Arayüz için durum raporu."""
        with self._lock:
            rounds = list(self._rounds)
        return {
            "teacher": self._teacher_name(),
            "topics": self._topics[:20],
            "studying": self._studying,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "stats": {
                **self._counter,
                "topics_done": len({item.topic for item in rounds}),
            },
            "recent": [item.to_dict() for item in rounds[-10:]],
        }

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        return dict(self._counter)

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<SelfStudy rounds={self._counter['rounds']} "
                f"notes={self._counter['notes']} teacher={self._teacher_name()}>")


__all__ = [
    "SelfStudy", "StudyRound", "QuestionTechnique", "TECHNIQUE_CYCLE",
    "QUESTION_PROMPT", "ANSWER_PROMPT", "CRITIQUE_PROMPT",
]
