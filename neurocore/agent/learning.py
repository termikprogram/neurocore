"""
NeuroCore Öğrenme Kaydedicisi
==============================
Ajanın yaptığı işi **öğrenmeye** çeviren köprü.

Neden bu modül var?
    Ajan katmanı (`neurocore.agent`) ile öğrenme çekirdeği
    (`neurocore.core.engine`) birbirinden bağımsız geliştirildi. Sonuç: ajan
    saatlerce çalışsa bile yerel model **tek adım bile ilerlemiyordu** —
    deneyim tamponu boş kalıyor, kasa büyümüyor, eğitim hiç çalışmıyordu.
    Bu modül o kopukluğu kapatır.

Üç öğrenme kanalı::

    Ajan görevi
        │
        ├─ 1. DENEYİM  → data/learning/experiences.jsonl  (+ evolution.observe)
        ├─ 2. KASA     → knowledge_base/  (yalnızca başarılı, öğretici çözümler)
        └─ 3. EĞİTİM   → models/neurocore/  (kaynak uygun olduğunda, arka planda)

**ZEHİRLİ DÖNGÜ YASAĞI (en kritik tasarım kuralı):**
    Ajanın kararları **bulut modelinden** (Gemini/Groq/DeepSeek) gelir. Yerel
    model yalnızca sizin görevlerinizden ve kasadan öğrenir. Yerel modelin
    ürettiği metin **asla** ajanın istemine geri verilmez. Aksi hâlde model
    kendi çıktısıyla eğitilir (model çökmesi / "model collapse") ve kalite
    hızla düşer. Bu kural `_build_prompt` yorumunda ve testlerde korunur.

**Kaynak nezaketi:**
    Eğitim yalnızca `NeuroCoreEngine.resources_available()` kapısı açıkken
    başlar; yani bilgisayarı boşta olan kaynağın %90'ından fazlasını kullanmaz
    ve siz çalışırken duraklar.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.learning")


# ─── Ödül Hesabı ────────────────────────────────────────────────────────────

#: Ödül aralığı — politika öğrenmesi ve değerlendirme için ortak ölçek
REWARD_MIN = -1.0
REWARD_MAX = 1.0


def compute_reward(run: Any, *, user_feedback: Optional[float] = None,
                   had_error: bool = False, self_recovered: bool = False) -> float:
    """
    Bir ajan çalışmasını **ödül sinyaline** çevirir.

    Neden ödül? Eğitim verisinin hangi kısmının iyi, hangisinin kötü örnek
    olduğunu bilmek gerekir. Ödül, "bunu tekrar yap" ile "bundan kaçın"
    arasındaki farkı sayısallaştırır.

    Puanlama:
        * Kullanıcı geri bildirimi varsa ağırlıklı olarak o kullanılır.
        * Başarılı + kendi hatasını düzeltmiş → **0.85** (en değerli örnek)
        * Başarılı, araç kullanmış          → **0.7** (gerçek iş yapılmış)
        * Başarılı, araçsız (sohbet)        → **0.3**
        * Başarısız ama ilerleme var        → **0.1**
        * Başarısız                         → **−0.6**
        * İptal / onay reddi                → **−0.3** (ceza değil, sinyal)
    """
    if user_feedback is not None:
        return max(REWARD_MIN, min(REWARD_MAX, float(user_feedback)))

    ok = bool(getattr(run, "ok", False))
    tools = list(getattr(run, "tools_used", []) or [])
    files = list(getattr(run, "files_changed", []) or [])
    steps = len(getattr(run, "steps", []) or [])
    reason = str(getattr(run, "reason", "") or "")
    cancelled = bool(getattr(run, "cancelled", False))

    if cancelled or "onay reddedildi" in reason or "iptal" in reason:
        return -0.3

    if ok:
        # Gerçek iş yapıldı mı? Sıralama önemli: **kendi kendini düzeltme**
        # en değerli örnektir (hata → çözüm çifti), dosya değişikliğinden de
        # üstündür. Bu yüzden önce o denetlenir.
        if self_recovered:
            return 0.85
        if files:
            return 0.8
        if tools:
            return 0.7
        return 0.3

    # Başarısız: kısmi ilerleme varsa hafif pozitif
    if had_error and steps > 1:
        return 0.1
    return -0.6


# ─── Deneyim Kaydı ──────────────────────────────────────────────────────────

@dataclass
class LearningRecord:
    """Öğrenme için saklanan tek bir ajan çalışması."""
    at: float
    task: str
    answer: str = ""
    ok: bool = False
    reward: float = 0.0
    mode: str = ""
    provider: str = ""
    model: str = ""
    tools_used: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    steps: int = 0
    duration_ms: float = 0.0
    reason: str = ""
    error: str = ""
    workspace: str = ""
    #: Araç çağrılarının kısa izi: (araç, başarılı, argüman özeti)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    #: Eğitimde kullanıldı mı?
    used_in_training: bool = False
    trained_at: float = 0.0
    #: Evrim katmanına (dolayısıyla eğitim verisine) aktarıldı mı?
    #:
    #: Motor tembel yüklendiği için ilk kayıtlar sırasında evrim katmanı
    #: yoktur; bu bayrak, motor yüklendiğinde birikenlerin **bir kez**
    #: aktarılmasını sağlar.
    observed: bool = False

    @property
    def instructive(self) -> bool:
        """
        Bu kayıt eğitim için **öğretici** mi?

        Yalnızca gerçek iş yapılmış çalışmalar modele bir şey öğretir.
        Tek satırlık sohbet yanıtları veri kümesini seyreltir.
        """
        return (self.ok and self.reward >= 0.6
                and (len(self.tools_used) >= 1 or bool(self.files_changed)))

    def prompt_text(self) -> str:
        """Eğitim çiftinin **girdi** tarafı."""
        return self.task.strip()

    def answer_text(self) -> str:
        """
        Eğitim çiftinin **hedef** tarafı.

        Yalnızca bulut modelinin ürettiği yanıt kullanılır; yerel modelin
        çıktısı asla buraya yazılmaz (zehirlenme koruması).
        """
        return (self.answer or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "at": self.at, "task": self.task, "answer": self.answer,
            "ok": self.ok, "reward": round(self.reward, 3), "mode": self.mode,
            # `instructive` türetilmiş bir alandır ama arayüz ve eğitim
            # kararları için gerekir; bu yüzden dışa aktarılır.
            "instructive": self.instructive,
            "provider": self.provider, "model": self.model,
            "tools_used": self.tools_used, "files_changed": self.files_changed,
            "commands_run": self.commands_run, "steps": self.steps,
            "duration_ms": round(self.duration_ms, 1), "reason": self.reason,
            "error": self.error, "workspace": self.workspace,
            "tool_trace": self.tool_trace,
            "used_in_training": self.used_in_training,
            "trained_at": self.trained_at,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningRecord":
        """Sözlükten kayıt üretir (eksik/fazla alanlar tolere edilir)."""
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


def record_from_run(run: Any, *, workspace: str = "",
                    user_feedback: Optional[float] = None) -> LearningRecord:
    """
    Bir `AgentRun` nesnesini öğrenme kaydına çevirir.

    Args:
        run: `agent.loop.AgentRun`
        workspace: Çalışma alanı yolu
        user_feedback: Kullanıcı puanı (−1..1); verilirse ödülü belirler

    Returns:
        `LearningRecord`
    """
    steps = list(getattr(run, "steps", []) or [])
    trace: List[Dict[str, Any]] = []
    had_error = False
    recovered = False
    for step in steps:
        if getattr(step, "kind", "") != "tool":
            continue
        ok = bool(getattr(step, "ok", False))
        if not ok:
            had_error = True
        elif had_error:
            # Hata aldıktan sonra başarılı adım → kendi kendini düzeltme
            recovered = True
        trace.append({
            "tool": getattr(step, "tool", ""),
            "ok": ok,
            "args": _brief(getattr(step, "args", {}) or {}),
        })

    # Kendi kendini düzeltme yalnızca çalışma **başarılıysa** değerlidir
    self_recovered = recovered and bool(getattr(run, "ok", False))

    return LearningRecord(
        at=time.time(),
        task=str(getattr(run, "task", "") or "").strip(),
        answer=str(getattr(run, "answer", "") or ""),
        ok=bool(getattr(run, "ok", False)),
        reward=compute_reward(run, user_feedback=user_feedback,
                              had_error=had_error, self_recovered=self_recovered),
        mode=str(getattr(run, "mode", "") or ""),
        provider=str(getattr(run, "provider", "") or ""),
        model=str(getattr(run, "model", "") or ""),
        tools_used=list(getattr(run, "tools_used", []) or []),
        files_changed=list(getattr(run, "files_changed", []) or []),
        commands_run=list(getattr(run, "commands_run", []) or []),
        steps=len(steps),
        duration_ms=float(getattr(run, "duration_ms", 0.0) or 0.0),
        reason=str(getattr(run, "reason", "") or ""),
        error=str(getattr(run, "error", "") or ""),
        workspace=workspace,
        tool_trace=trace,
    )


def _brief(args: Dict[str, Any], limit: int = 3) -> str:
    """Argümanların kısa özeti (deneyim kaydı şişmesin)."""
    parts: List[str] = []
    for key, value in list(args.items())[:limit]:
        text = str(value).replace("\n", "⏎")
        parts.append(f"{key}={text[:40]}")
    return ", ".join(parts)[:160]


# ─── Öğrenme Kaydedicisi ────────────────────────────────────────────────────

class LearningRecorder:
    """
    Ajan çalışmalarını deneyime, kasaya ve eğitime bağlar.

    Kullanım:
        recorder = LearningRecorder()
        recorder.initialize(workspace="E:/proje", data_path=".neurocore")

        # Her ajan çalışmasından sonra:
        record = recorder.record(run)
        print(record.reward, record.instructive)

        # Kullanıcı puanı (isteğe bağlı ama çok değerli):
        recorder.feedback(record.at, +1.0)

        # Arka plan eğitimi (kaynak uygunsa kendiliğinden):
        recorder.start_worker()
        print(recorder.status())

    Args:
        min_new_experiences: Eğitim için gereken en az **yeni** kayıt sayısı
        min_interval_seconds: İki eğitim arası en az süre (saniye)
        max_minutes_per_train: Tek eğitim turunun süre sınırı
        write_notes: Başarılı çözümler kasaya not olarak yazılsın mı?
    """

    name = "agent.learning"

    def __init__(self, *, min_new_experiences: int = 5,
                 min_interval_seconds: float = 900.0,
                 max_minutes_per_train: float = 4.0,
                 write_notes: bool = True) -> None:
        self.min_new_experiences = max(1, int(min_new_experiences))
        self.min_interval_seconds = max(60.0, float(min_interval_seconds))
        self.max_minutes_per_train = max(0.5, float(max_minutes_per_train))
        self.write_notes = bool(write_notes)

        self._records: List[LearningRecord] = []
        self._lock = threading.RLock()
        self._data_path: Optional[Path] = None
        self._journal_path: Optional[Path] = None
        self._workspace: str = ""
        self._event_bus: Any = None
        self._initialized = False

        # Motor **tembel** yüklenir: 208M parametreli modeli açılışta yüklemek
        # 6 saniye sürer ve GPU belleği ister; kullanıcı yalnızca sohbet
        # edecekse buna gerek yoktur.
        self._engine: Any = None
        self._engine_lock = threading.RLock()
        self._engine_error: str = ""
        #: Hafif kasa (model gerektirmeyen not yazımı için, tembel)
        self._knowledge: Any = None

        self._worker: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._training = False
        self._last_training: float = 0.0
        self._last_training_result: Dict[str, Any] = {}
        self._pending_since_training = 0

        self._counter = {
            "recorded": 0, "notes_written": 0, "notes_skipped": 0,
            "feedback": 0, "training_runs": 0, "training_steps": 0,
            "training_skipped": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, workspace: str = "", data_path: str = "",
                   event_bus: Any = None) -> None:
        """Kaydediciyi başlatır ve geçmiş kayıtları yükler."""
        if self._initialized:
            return
        self._workspace = workspace or ""
        self._event_bus = event_bus

        base = Path(data_path) if data_path else Path(workspace or ".") / ".neurocore"
        self._data_path = base / "learning"
        try:
            self._data_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Öğrenme dizini oluşturulamadı: {exc}")
            self._data_path = Path.cwd() / "learning"
            self._data_path.mkdir(parents=True, exist_ok=True)
        self._journal_path = self._data_path / "experiences.jsonl"

        self._load()
        self._initialized = True
        instructive = sum(1 for record in self._records if record.instructive)
        logger.info(
            f"Öğrenme kaydedicisi hazır — {len(self._records)} kayıt "
            f"({instructive} öğretici), günlük: {self._journal_path.name}"
        )

    def shutdown(self, save_engine: bool = True) -> None:
        """Arka plan işçisini durdurur ve motoru kapatır."""
        self.stop_worker()
        with self._engine_lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as exc:
                logger.debug(f"Motor kapatılamadı: {exc}")
        if self._knowledge is not None:
            try:
                self._knowledge.shutdown()
            except Exception as exc:
                logger.debug(f"Kasa kapatılamadı: {exc}")
            self._knowledge = None
        self._initialized = False
        logger.info(
            f"Öğrenme kaydedicisi kapatıldı ({self._counter['recorded']} kayıt, "
            f"{self._counter['training_runs']} eğitim turu)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        with self._lock:
            records = list(self._records)
        return {
            **self._counter,
            "initialized": self._initialized,
            "records": len(records),
            "instructive": sum(1 for record in records if record.instructive),
            "used_in_training": sum(1 for record in records if record.used_in_training),
            "pending": self._pending_since_training,
            "training": self._training,
            "last_training": self._last_training,
            "journal": str(self._journal_path or ""),
            "engine_loaded": self._engine is not None,
            "engine_error": self._engine_error,
        }

    # ─── Kayıt ──────────────────────────────────────────────────────────

    def record(self, run: Any, *, user_feedback: Optional[float] = None) -> LearningRecord:
        """
        Bir ajan çalışmasını kaydeder.

        Üç şey yapar: günlüğe yazar, evrim katmanına bildirir (varsa) ve
        başarılıysa kasaya not ekler.

        Args:
            run: `agent.loop.AgentRun`
            user_feedback: Kullanıcı puanı (−1..1)

        Returns:
            Oluşan `LearningRecord`
        """
        record = record_from_run(run, workspace=self._workspace,
                                 user_feedback=user_feedback)
        if not record.task:
            return record

        with self._lock:
            self._records.append(record)
            self._pending_since_training += 1
            self._counter["recorded"] += 1

        self._append_journal(record)
        self._observe_evolution(record)
        if self.write_notes and record.instructive:
            self._write_note(record)

        self._emit("learning.recorded", {
            "task": record.task[:160], "reward": record.reward,
            "ok": record.ok, "instructive": record.instructive,
            "tools": record.tools_used,
        })
        logger.info(
            f"Deneyim kaydedildi: ödül {record.reward:+.2f} · "
            f"{'başarılı' if record.ok else 'başarısız'} · "
            f"{len(record.tools_used)} araç · "
            f"{'öğretici' if record.instructive else 'öğretici değil'}"
        )
        return record

    def feedback(self, when: float, reward: float,
                 note: str = "") -> Dict[str, Any]:
        """
        Bir kayda **kullanıcı puanı** verir.

        Bu, en değerli öğrenme sinyalidir: kullanıcı "bu iyiydi" veya "bu
        yanlıştı" dediğinde model doğrudan düzeltilir.

        Args:
            when: Kaydın `at` zaman damgası (tolerans: ±2 saniye)
            reward: −1..1 arası puan
            note: Kısa gerekçe

        Returns:
            {"ok", "record"?, "error"?}
        """
        reward = max(REWARD_MIN, min(REWARD_MAX, float(reward)))
        with self._lock:
            target = None
            for record in reversed(self._records):
                if abs(record.at - when) <= 2.0:
                    target = record
                    break
            if target is None and self._records:
                target = self._records[-1]
            if target is None:
                return {"ok": False, "error": "geri bildirim verilecek kayıt yok"}
            target.reward = reward
            self._counter["feedback"] += 1

        # Evrim katmanına da ilet (politika öğrenmesi için)
        if self._engine is not None:
            try:
                self._engine.evolution.feedback(str(target.at), reward, note=note)
            except Exception as exc:
                logger.debug(f"Evrim geri bildirimi iletilemedi: {exc}")

        self._rewrite_journal()
        self._emit("learning.feedback", {
            "task": target.task[:120], "reward": reward, "note": note,
        })
        logger.info(f"Kullanıcı geri bildirimi: {reward:+.2f} — {target.task[:60]}")
        return {"ok": True, "record": target.to_dict()}

    # ─── Kasa ───────────────────────────────────────────────────────────

    def knowledge_layer(self) -> Any:
        """
        Kasa katmanını döndürür (tembel kurulur, model gerektirmez).

        Kendi kendine öğrenme (`agent.study`) kasaya not yazmak için bunu
        kullanır. Motor yüklüyse onun kasası tercih edilir; değilse hafif bir
        kasa kurulur — böylece 208M parametreli model yüklenmeden de not
        yazılabilir.
        """
        if self._engine is not None:
            knowledge = getattr(self._engine, "knowledge", None)
            if knowledge is not None:
                return knowledge
        if self._knowledge is None:
            try:
                from neurocore.knowledge import KnowledgeLayer

                layer = KnowledgeLayer()
                layer.initialize(event_bus=self._event_bus)
                self._knowledge = layer
            except Exception as exc:
                logger.warning(f"Hafif kasa kurulamadı: {exc}")
                return None
        return self._knowledge

    def _write_note(self, record: LearningRecord) -> None:
        """
        Başarılı çözümü kasaya **bilgi notu** olarak yazar.

        Not, gelecekte ajanın bağlamına girer ve benzer görevlerde yol
        gösterir. Yalnızca gerçek iş yapılmış çalışmalar yazılır.

        Neden hafif kasa? Motordan (`NeuroCoreEngine`) yazmak için motorun
        yüklü olması gerekir; ama motor 208M parametreli modeli GPU'ya yükler
        ve eğitim sırasında zaten açıktır. İki ağır örnek çakışır. Bu yüzden
        not yazımı **model gerektirmeyen** hafif kasa üzerinden yapılır.
        """
        title = self._note_title(record)
        body = self._note_body(record)

        engine = self._engine
        if engine is not None:
            try:
                # Not: `NeuroCoreEngine.learn()` imzası `source` (tekil) ve
                # `importance`/`sources` **almaz**; uydurma parametre göndermek
                # TypeError verir.
                engine.learn(
                    body, title=title, kind="skill",
                    tags=["öğrenilen", "ajan"] + [name.split(".")[0]
                                                  for name in record.tools_used[:3]],
                    confidence=max(0.3, min(0.95, (record.reward + 1) / 2)),
                    source=f"agent://run/{int(record.at)}",
                )
                self._counter["notes_written"] += 1
                self._emit("learning.note_written", {"title": title})
                return
            except Exception as exc:
                logger.debug(f"Motor üzerinden not yazılamadı, hafif kasaya düşülüyor: {exc}")

        # Hafif kasa yolu (model gerektirmez)
        try:
            from neurocore.knowledge import KnowledgeLayer

            if self._knowledge is None:
                knowledge = KnowledgeLayer()
                knowledge.initialize(event_bus=self._event_bus)
                self._knowledge = knowledge
            self._knowledge.write(
                title, body, kind="skill",
                tags=["öğrenilen", "ajan"] + [name.split(".")[0]
                                              for name in record.tools_used[:3]],
                confidence=max(0.3, min(0.95, (record.reward + 1) / 2)),
                importance=0.6,
                sources=[f"agent://run/{int(record.at)}"],
                dedupe=True,
            )
            self._counter["notes_written"] += 1
            self._emit("learning.note_written", {"title": title})
            logger.info(f"Öğrenme notu kasaya yazıldı: {title[:60]}")
        except Exception as exc:
            self._counter["notes_skipped"] += 1
            logger.debug(f"Öğrenme notu yazılamadı: {exc}")

    @staticmethod
    def _note_title(record: LearningRecord) -> str:
        """Not başlığı: görevin kısaltılmış hâli."""
        task = " ".join(record.task.split())
        if len(task) > 70:
            task = task[:67].rsplit(" ", 1)[0] + "…"
        return f"Çözüm: {task}"

    @staticmethod
    def _note_body(record: LearningRecord) -> str:
        """Not gövdesi: ne istendi, ne yapıldı, nasıl doğrulandı."""
        lines = [
            "## Görev",
            record.task,
            "",
            "## Uygulanan adımlar",
        ]
        for item in record.tool_trace[:12]:
            mark = "✓" if item.get("ok") else "✗"
            args = item.get("args", "")
            lines.append(f"- {mark} `{item.get('tool')}`"
                         + (f" — {args}" if args else ""))
        if record.files_changed:
            lines.extend(["", "## Değişen dosyalar"])
            lines.extend(f"- `{name}`" for name in record.files_changed[:12])
        if record.commands_run:
            lines.extend(["", "## Çalıştırılan komutlar"])
            lines.extend(f"- `{cmd[:110]}`" for cmd in record.commands_run[:8])
        if record.answer:
            lines.extend(["", "## Sonuç", _trim(record.answer, 1200)])
        lines.extend([
            "", "## Kayıt",
            f"- Sağlayıcı: {record.provider or 'yerel'} ({record.model or '—'})",
            f"- Adım: {record.steps} · süre: {record.duration_ms / 1000:.1f} sn",
            f"- Ödül: {record.reward:+.2f}",
        ])
        return "\n".join(lines)

    # ─── Evrim Katmanı ──────────────────────────────────────────────────

    def _observe_evolution(self, record: LearningRecord) -> None:
        """
        Deneyimi evrim katmanına bildirir (varsa).

        Evrim katmanı kendi vektör tamponunu ve müfredatını yönetir; eğitim
        verisi **oradan** üretilir (`engine.train` →
        `evolution.curriculum.build` → `build_dataset_from_experiences`).

        ⚠ Bu yüzden motor yüklü değilse deneyim **kaybolmamalıdır**: motor
        tembel yüklendiği için (208M parametre) ilk kayıtlar sırasında henüz
        yoktur. Motor sonradan yüklendiğinde biriken kayıtlar toplu olarak
        gönderilir — aksi hâlde ölçüldüğü gibi eğitim veri kümesi **boş**
        kalır ve model yalnızca kasadan öğrenir:

            `Kasadan 32 not ile eğitim verisi üretildi (35 örnek)`
            (deneyimden söz edilmiyor → sohbetler eğitime girmiyor)
        """
        engine = self._engine
        if engine is None:
            self._counter["observations_pending"] = int(
                self._counter.get("observations_pending", 0)) + 1
            return
        self._flush_observations(engine)

    def _flush_observations(self, engine: Any) -> int:
        """
        Motor yüklendiğinde **biriken** deneyimleri evrim katmanına aktarır.

        Yalnızca henüz gönderilmemiş kayıtlar aktarılır (her kayıt için
        `observed` bayrağı tutulur), böylece tekrar gönderim olmaz.

        Returns:
            Aktarılan kayıt sayısı
        """
        with self._lock:
            pending = [record for record in self._records
                       if not getattr(record, "observed", False)]
        if not pending:
            return 0
        sent = 0
        for record in pending:
            try:
                engine.evolution.observe(
                    query=record.task,
                    answer=record.answer[:2000],
                    context=record.files_changed[:6] + record.commands_run[:4],
                    latency_ms=record.duration_ms,
                    uncertainty=max(0.0, min(1.0, (1.0 - record.reward) / 2)),
                    strategy=record.mode or "agent",
                    tags=["ajan"] + [name.split(".")[0]
                                     for name in record.tools_used[:4]],
                )
                record.observed = True      # type: ignore[attr-defined]
                sent += 1
            except Exception as exc:
                logger.debug(f"Deneyim aktarılamadı: {exc}")
                break
        if sent:
            self._counter["observations"] = int(
                self._counter.get("observations", 0)) + sent
            self._counter["observations_pending"] = 0
            logger.info(
                f"{sent} deneyim evrim katmanına aktarıldı — eğitim verisine "
                "girecek"
            )
            self._emit("learning.observations_flushed", {"count": sent})
            self._rewrite_journal()
        return sent

    # ─── Günlük (JSONL) ─────────────────────────────────────────────────

    def _append_journal(self, record: LearningRecord) -> None:
        """Kaydı JSONL günlüğüne ekler (atomik satır yazımı)."""
        if self._journal_path is None:
            return
        try:
            with self._journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Deneyim günlüğe yazılamadı: {exc}")

    def _rewrite_journal(self) -> None:
        """Geri bildirim sonrası günlüğü yeniden yazar."""
        if self._journal_path is None:
            return
        try:
            temporary = self._journal_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for record in self._records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            temporary.replace(self._journal_path)
        except OSError as exc:
            logger.warning(f"Deneyim günlüğü güncellenemedi: {exc}")

    def _load(self) -> None:
        """Geçmiş kayıtları okur."""
        if self._journal_path is None or not self._journal_path.exists():
            return
        loaded = 0
        try:
            with self._journal_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._records.append(
                            LearningRecord.from_dict(json.loads(line))
                        )
                        loaded += 1
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError as exc:
            logger.warning(f"Deneyim günlüğü okunamadı: {exc}")
        # Sınırsız büyümeyi önle (son 5000 kayıt yeterli)
        if len(self._records) > 5000:
            self._records = self._records[-5000:]
        if loaded:
            # ── Bekleyen işi **diskten** hesapla ──
            # Sayaç bellekte tutulur ve yeniden başlatmada sıfırlanıyordu;
            # bu yüzden 10+ eğitici kayıt birikse bile "yeterli yeni deneyim
            # yok (0/5)" görünüyor ve eğitim **hiç başlamıyordu**. Bekleyen
            # iş, eğitimde kullanılmamış **öğretici** kayıtların sayısıdır ve
            # diskten hesaplanabilir.
            self._pending_since_training = sum(
                1 for record in self._records
                if record.instructive and not record.used_in_training
            )
            logger.info(
                f"Günlükten {loaded} kayıt yüklendi — {self._pending_since_training} "
                "öğretici kayıt eğitim bekliyor"
            )

    def records(self, limit: int = 50,
                only_instructive: bool = False) -> List[Dict[str, Any]]:
        """Son kayıtlar."""
        with self._lock:
            items = list(self._records)
        if only_instructive:
            items = [item for item in items if item.instructive]
        return [item.to_dict() for item in items[-limit:]]

    # ─── Eğitim ─────────────────────────────────────────────────────────

    def _ensure_engine(self) -> Tuple[bool, str]:
        """
        Öğrenme motorunu **tembel** yükler.

        Motor 208M parametreli modeli yükler (~6 saniye, GPU belleği).
        Kullanıcı yalnızca sohbet edecekse bunu açılışta yapmak gereksizdir.
        """
        with self._engine_lock:
            if self._engine is not None:
                return True, ""
            if self._engine_error and not self._engine_error.startswith("geçici"):
                return False, self._engine_error
            try:
                from neurocore import NeuroCoreEngine

                engine = NeuroCoreEngine()
                engine.initialize()
                self._engine = engine
                self._engine_error = ""
                logger.info("Öğrenme motoru yüklendi (tembel)")
                self._emit("learning.engine_ready", {
                    "state": str(getattr(engine.state, "value", engine.state)),
                })
                return True, ""
            except Exception as exc:
                self._engine_error = f"geçici: {type(exc).__name__}: {exc}"
                logger.warning(f"Öğrenme motoru yüklenemedi: {exc}")
                return False, self._engine_error

    def training_readiness(self) -> Dict[str, Any]:
        """
        Eğitim başlatılabilir mi? Gerekçesiyle birlikte bildirir.

        Returns:
            {"ready", "reason", "pending", "wait_seconds", "resources_ok"}
        """
        # Bekleyen iş **kayıtlardan** hesaplanır (artımlı sayaç kayabilir)
        with self._lock:
            pending = sum(
                1 for record in self._records
                if record.instructive and not record.used_in_training
            )
        elapsed = time.time() - self._last_training
        wait = max(0.0, self.min_interval_seconds - elapsed)

        if self._training:
            return {"ready": False, "reason": "eğitim zaten çalışıyor",
                    "pending": pending, "wait_seconds": 0.0, "resources_ok": True}
        if pending < self.min_new_experiences:
            return {
                "ready": False,
                "reason": (f"yeterli yeni deneyim yok ({pending}/"
                           f"{self.min_new_experiences})"),
                "pending": pending, "wait_seconds": wait, "resources_ok": True,
            }
        if wait > 0:
            return {"ready": False,
                    "reason": f"iki eğitim arası bekleme ({wait:.0f} sn)",
                    "pending": pending, "wait_seconds": wait, "resources_ok": True}

        ok, reason = self._ensure_engine()
        if not ok:
            return {"ready": False, "reason": f"motor yüklenemedi: {reason}",
                    "pending": pending, "wait_seconds": 0.0, "resources_ok": False}

        try:
            available, why = self._engine.resources_available(for_heavy_work=True)
        except Exception as exc:
            available, why = False, f"kaynak denetimi başarısız: {exc}"
        if not available:
            return {"ready": False, "reason": f"kaynaklar uygun değil: {why}",
                    "pending": pending, "wait_seconds": 0.0, "resources_ok": False}

        return {"ready": True, "reason": "hazır", "pending": pending,
                "wait_seconds": 0.0, "resources_ok": True}

    def train_now(self, *, max_minutes: Optional[float] = None,
                  steps: Optional[int] = None,
                  force: bool = False) -> Dict[str, Any]:
        """
        Yerel modeli eğitir (kasa + deneyim).

        Args:
            max_minutes: Süre sınırı (None = varsayılan)
            steps: Adım sayısı (None = otomatik plan)
            force: Kaynak/deneyim kapılarını atla (kullanıcı açıkça istediyse)

        Returns:
            Eğitim sonucu (JSON güvenli)
        """
        if self._training:
            return {"ok": False, "error": "eğitim zaten çalışıyor"}

        if not force:
            readiness = self.training_readiness()
            if not readiness["ready"]:
                self._counter["training_skipped"] += 1
                self._emit("learning.training_skipped", readiness)
                return {"ok": False, "skipped": True,
                        "error": readiness["reason"], **readiness}

        ok, reason = self._ensure_engine()
        if not ok:
            return {"ok": False, "error": f"motor yüklenemedi: {reason}"}

        with self._lock:
            self._training = True
        # Motor yeni yüklendiyse biriken deneyimleri **şimdi** aktar: eğitim
        # verisi evrim katmanının tamponundan üretilir, orası boşsa model
        # yalnızca kasadan öğrenir (sohbetler eğitime girmez).
        self._flush_observations(self._engine)
        started = time.time()
        self._emit("learning.training_started", {
            "steps": steps, "max_minutes": max_minutes or self.max_minutes_per_train,
            "experiences": len(self._records),
        })
        logger.info("Eğitim başlıyor (kasa + deneyim)…")

        try:
            result = self._engine.train(
                steps=steps,
                from_vault=True,
                from_experiences=True,
                max_minutes=max_minutes or self.max_minutes_per_train,
            )
        except Exception as exc:
            logger.error(f"Eğitim çöktü: {exc}", exc_info=True)
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self._lock:
                self._training = False
            self._last_training = time.time()

        duration = time.time() - started
        # Eğitimde kullanılan kayıtları işaretle
        with self._lock:
            for record in self._records:
                if not record.used_in_training and record.instructive:
                    record.used_in_training = True
                    record.trained_at = self._last_training
            self._pending_since_training = 0

        if not isinstance(result, dict):
            result = {"ok": False, "error": f"beklenmeyen sonuç tipi: {type(result).__name__}"}
        if not result:
            # Boş sözlük = sessiz başarısızlık; kullanıcıya bildirilmelidir.
            result = {"ok": False,
                      "error": "eğitim sonuç döndürmedi (eğitim çekirdeği sessizce durdu)"}

        if result.get("ok"):
            self._counter["training_runs"] += 1
            self._counter["training_steps"] += int(result.get("steps_completed", 0) or 0)
        self._last_training_result = {
            "at": self._last_training, "duration_seconds": round(duration, 1),
            **{key: value for key, value in result.items() if key != "history"},
        }
        self._rewrite_journal()
        self._emit("learning.training_finished", self._last_training_result)
        if result.get("ok"):
            logger.info(
                f"Eğitim tamamlandı: {result.get('steps_completed', 0)} adım, "
                f"{duration:.0f} sn, kayıp "
                f"{result.get('initial_loss', 0):.3f} → "
                f"{result.get('final_loss', 0):.3f}"
            )
        else:
            logger.warning(f"Eğitim başarısız: {result.get('error')}")
        return self._last_training_result

    def start_worker(self, interval: float = 120.0) -> bool:
        """
        Arka plan öğrenme işçisini başlatır.

        İşçi belirli aralıklarla `training_readiness()` denetler; kaynaklar
        uygunsa ve yeterli yeni deneyim birikmişse eğitimi **kendiliğinden**
        başlatır. Kaynak nezaketi kapısı her turda yeniden denetlenir, bu
        yüzden kullanıcı bilgisayarı kullanırken eğitim duraklar.

        Args:
            interval: Denetim aralığı (saniye)

        Returns:
            İşçi başlatıldıysa True
        """
        if self._worker is not None and self._worker.is_alive():
            return True
        self._stop_worker.clear()

        def loop() -> None:
            """Denetim döngüsü."""
            while not self._stop_worker.wait(interval):
                try:
                    readiness = self.training_readiness()
                    if readiness["ready"]:
                        self.train_now()
                except Exception as exc:
                    logger.debug(f"Öğrenme işçisi hatası: {exc}")

        self._worker = threading.Thread(target=loop, name="NeuroCoreLearning",
                                        daemon=True)
        self._worker.start()
        logger.info(f"Öğrenme işçisi başladı (denetim aralığı {interval:.0f} sn)")
        return True

    def stop_worker(self) -> None:
        """Arka plan işçisini durdurur."""
        self._stop_worker.set()
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    @property
    def is_training(self) -> bool:
        """Şu anda eğitim çalışıyor mu?"""
        return self._training

    def status(self) -> Dict[str, Any]:
        """Arayüz için tam durum raporu."""
        stats = self.get_stats()
        with self._lock:
            records = list(self._records)
        # ── Sayaçları **gerçek durumdan** hesapla ──
        # `_pending_since_training` artımlı bir sayaçtır; yeniden başlatma ve
        # anlık görüntüleme sırasında kayabilir. Arayüz ve eğitim kararı
        # doğrudan kayıtlara bakmalı.
        pending = sum(
            1 for record in records
            if record.instructive and not record.used_in_training
        )
        stats["pending"] = pending
        stats["observations_pending"] = sum(
            1 for record in records if not record.observed
        )
        stats["instructive"] = sum(1 for record in records if record.instructive)
        stats["used_in_training"] = sum(
            1 for record in records if record.used_in_training
        )
        readiness = self.training_readiness()
        readiness["pending"] = pending
        if not readiness["ready"] and pending < self.min_new_experiences:
            readiness["reason"] = (
                f"yeterli yeni deneyim yok ({pending}/"
                f"{self.min_new_experiences}) — öğretici bir görev çalıştırın"
            )
        model: Dict[str, Any] = {}
        if self._engine is not None:
            try:
                model = {
                    "parameters": self._engine.model.parameter_count(),
                    "training_steps": self._engine.model.growth.total_training_steps,
                    "tokens_seen": self._engine.model.growth.total_tokens_seen,
                    "experts": self._engine.model.growth.num_experts,
                    "generation": self._engine.model.growth.generation,
                    "can_generate": self._engine.model.can_generate,
                    "min_generation_steps": getattr(
                        self._engine.model, "MIN_GENERATION_STEPS", 800),
                }
            except Exception as exc:
                model = {"error": str(exc)}
        return {
            "stats": stats,
            "readiness": readiness,
            "model": model,
            "last_training": self._last_training_result,
            #: Eğitim için gereken en az **yeni** öğretici kayıt (arayüz
            #: "3/5" gibi bir ilerleme göstergesi çizer)
            "min_new_experiences": self.min_new_experiences,
            "min_interval_seconds": self.min_interval_seconds,
        }

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<LearningRecorder records={len(self._records)} "
                f"pending={self._pending_since_training} "
                f"training={self._training}>")


def _trim(text: str, limit: int) -> str:
    """Metni sınırlar (not gövdesi şişmesin)."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


__all__ = [
    "LearningRecorder",
    "LearningRecord",
    "compute_reward",
    "record_from_run",
    "REWARD_MIN",
    "REWARD_MAX",
]
