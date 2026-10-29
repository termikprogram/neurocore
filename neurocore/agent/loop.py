"""
NeuroCore Ajan Döngüsü
=======================
"Planla → uygula → gözlemle → tekrarla" döngüsünün tam uygulaması.

Neden bu modül var?
    Araç kayıt defteri (`agent.tools`) tek bir çağrıyı yönetir; kodlama ve
    kabuk katmanları tek bir işi yapar. Ama gerçek görev — "bu projeye karanlık
    mod ekle", "Godot sahnesini düzelt", "testleri geçir" — onlarca adım
    gerektirir. Bu modül o adımları **modelden gelen kararlara** göre sıralar,
    sonuçları modele geri besler ve işi bitirir.

İki çalışma modu:
    * **Araç çağrısı modu** (`supports_tools=True`) — model şema tabanlı
      `tool_calls` üretir. Güvenilir yol.
    * **JSON istemi modu** — model araç çağrısını desteklemiyorsa, yanıtından
      ```json {"tool": "...", "args": {...}} ``` bloğu ayrıştırılır.
    * **Kural tabanlı mod** — hiçbir dil modeli yoksa, yaygın istekleri
      (proje haritası, sürümler, testler, dosya okuma) doğrudan araçlara
      eşleyen gömülü bir yönlendirici devreye girer. Sistem böylece **hiçbir
      API anahtarı olmadan da** işe yarar.

Onay akışı:
    Politika bir çağrıyı engellerse döngü `approval_callback`'i çağırır. Bu
    geri çağrım arayüze bir soru yayar ve kullanıcı yanıtını bekler
    (`None` dönerse akış kesintisiz sürer — ör. komut satırı modu).

Durdurma koşulları:
    * model araç çağırmadan metin yanıtı verdi → bitti
    * adım sınırı aşıldı
    * süre sınırı aşıldı
    * kullanıcı iptal etti
    * aynı araç + aynı argüman üst üste tekrarlandı (döngü tespiti)
"""

from __future__ import annotations

from pathlib import Path

import difflib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neurocore.agent.playbooks import PlaybookStore
from neurocore.agent.cognition import (
    CognitionLayer,
    HataTuru,
    HedefDurumu,
    kurtarma_talimati,
)
from neurocore.agent.llm import LLMMessage, LLMResponse, LLMRouter
from neurocore.agent.tools import (
    SAFETY_LABELS,
    AutonomyLevel,
    Safety,
    ToolContext,
    ToolDef,
    ToolRegistry,
    ToolResult,
    to_wire_name,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.loop")


#: Hiyerarşik planlama için en kısa görev uzunluğu
HIER_MIN_UZUNLUK = 80

#: ⚠ Güvenlik tavanı: adım sınırı kaldırıldığında bile kaçınılmaz üst sınır.
#: 400 adım ≈ 400 araç çağrısı; süre sınırı (600 sn) genelde önce dolar.
MUTLAK_ADIM = 400

#: Bağlam sıkıştırma eşiği: bu kadar mesajdan sonra özetlenir
SIKISTIRMA_ESIK = 24

#: Sıkıştırmada korunacak son mesaj sayısı
SIKISTIRMA_TUT = 10


def _kisa_args(args: Any) -> str:
    """Araç argümanlarını tek satırlık kısa metne indirger."""
    if not isinstance(args, dict):
        return ""
    parcalar = [f"{k}={str(v)[:32]}" for k, v in list(args.items())[:3]]
    return ", ".join(parcalar)[:120]


def _kisa_olgu(result: ToolResult) -> str:
    """
    Araç sonucundan **tek satırlık** doğrulanmış bilgi çıkarır.

    Hedef durumuna girer; bağlamı şişirmeden kanıt taşır.
    """
    veri = result.data
    if isinstance(veri, dict):
        for anahtar in ("path", "file", "url", "command", "stdout", "text",
                        "message", "title", "result"):
            if veri.get(anahtar):
                return f"{result.tool}: {str(veri[anahtar])[:120]}"
        return f"{result.tool}: {len(veri)} alan döndü"
    if isinstance(veri, (list, tuple)):
        return f"{result.tool}: {len(veri)} öğe döndü"
    if veri is not None:
        return f"{result.tool}: {str(veri)[:120]}"
    return f"{result.tool}: tamam"


# ─── Veri Yapıları ──────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """Döngüdeki tek bir adım (şeffaflık için arayüze gönderilir)."""
    index: int
    kind: str                       # thinking | tool | approval | answer | error | plan
    text: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    ok: bool = True
    duration_ms: float = 0.0
    at: float = field(default_factory=time.time)

    def to_dict(self, max_result: int = 4000) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "index": self.index, "kind": self.kind, "text": self.text[:4000],
            "ok": self.ok, "duration_ms": round(self.duration_ms, 1),
            "at": self.at,
        }
        if self.tool:
            data["tool"] = self.tool
            data["args"] = self.args
            data["result"] = self.result[:max_result]
        return data


@dataclass
class AgentRun:
    """Tek bir ajan çalışmasının sonucu."""
    task: str = ""
    answer: str = ""
    ok: bool = True
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    mode: str = ""                  # tools | prompt | rules
    provider: str = ""
    model: str = ""
    reason: str = ""                # neden durdu
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    cancelled: bool = False
    approvals: int = 0
    denials: int = 0

    def to_dict(self, include_steps: bool = True) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "task": self.task, "answer": self.answer, "ok": self.ok,
            "mode": self.mode, "provider": self.provider, "model": self.model,
            "reason": self.reason,
            "duration_ms": round(self.duration_ms, 1),
            "steps_count": len(self.steps),
            "tools_used": self.tools_used,
            "files_changed": self.files_changed,
            "commands_run": self.commands_run,
            "tokens": {"prompt": self.prompt_tokens,
                       "completion": self.completion_tokens,
                       "total": self.prompt_tokens + self.completion_tokens},
            "error": self.error,
            "cancelled": self.cancelled,
            "approvals": self.approvals, "denials": self.denials,
        }
        if include_steps:
            data["steps"] = [step.to_dict() for step in self.steps]
        return data

    def render(self, max_steps: int = 40) -> str:
        """İnsan okur özet."""
        lines = [f"Görev: {self.task}", f"Mod: {self.mode or 'bilinmiyor'}"
                 + (f" ({self.provider}/{self.model})" if self.provider else "")]
        for step in self.steps[:max_steps]:
            if step.kind == "tool":
                mark = "✓" if step.ok else "✗"
                lines.append(f"  {mark} {step.tool}({_brief_args(step.args)}) "
                             f"— {step.duration_ms:.0f}ms")
            elif step.kind == "approval":
                lines.append(f"  ⚿ onay: {step.text}")
            elif step.kind == "error":
                lines.append(f"  ! {step.text[:160]}")
        if self.files_changed:
            lines.append(f"Değişen dosyalar: {', '.join(self.files_changed[:12])}")
        lines.append("")
        lines.append(self.answer or "(yanıt üretilmedi)")
        return "\n".join(lines)


# ─── İstemler ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sen NeuroCore'sun: kullanıcının Windows bilgisayarında çalışan, kod yazabilen, \
komut çalıştırabilen ve uygulamaları (Godot, Blender) kontrol edebilen bir yazılım ajanısın.

## Dil (⚠ ölçüldü: istem yalnız Türkçeydi → İngilizce kullanıcı Türkçe yanıt alıyordu)
Kullanıcı hangi dilde yazdıysa **o dilde** yanıtla. Türkçe soruya Türkçe,
İngilizce soruya İngilizce. Kod, komut ve dosya adları dilden bağımsızdır;
onları **olduğu gibi** bırak. Dil belirsizse (ör. tek sözcük) son kullanılan
dili sürdür.

## Temel kurallar
1. **Önce bak, sonra değiştir.** Bir dosyayı düzenlemeden önce oku. Bir projeyi \
anlamadan önce `code.workspace_map` ile haritala.
2. **Kanıtla.** Bir değişiklik yaptıysan çalıştığını göster: `code.check_syntax`, \
`code.run_tests` veya `code.run_command`.
3. **Küçük adımlar.** Tek seferde tek mantıklı iş yap; sonucu gör, sonra devam et.
4. **Cerrahi düzenleme.** `code.edit_file` kullan; `old_text` dosyada birebir \
bulunmalı (girinti ve satır sonları dahil). Gerekirse önce dosyayı okuyup tam \
metni kopyala.
5. **Hata alırsan teşhis et.** `code.diagnose_error` veya `shell.diagnose` ile \
hata çıktısını yorumla; aynı komutu körü körüne tekrarlama.
6. **Durum koru.** Sıralı işlerde (kurulum, derleme) `shell.session_run` kullan; \
`cd` orada kalıcıdır.
7. **Uydurma.** Bilmediğin bir API veya dosya içeriği hakkında tahmin yürütme; \
okumak için araç çağır.

## Çalışma alanı
Çalışma alanın kullanıcının açtığı projedir. Çalışma alanı dışına yazma \
reddedilir. Yıkıcı komutlar (silme, format, kayıt defteri) engellenir veya \
onay ister.

## Yanıt biçimi
İş bitince **Türkçe**, kısa ve somut bir özet ver: ne yaptın, hangi dosyalar \
değişti, nasıl doğruladın, kalan bir sorun var mı. Kod bloğu yalnızca gerektiğinde.

Kullanıcı Türkçe yazıyorsa Türkçe yanıt ver."""

PLAN_PROMPT = """Aşağıdaki görevi tamamlamak için kısa bir plan yap.

Görev: {task}

Çalışma alanı bilgisi:
{workspace}

Planı **3-6 madde** olarak yaz. Her madde tek bir eylem olsun. Sonra hemen ilk \
adımı uygula. Plan dışında açıklama yazma."""


# ─── Kural Tabanlı Yönlendirici ─────────────────────────────────────────────

@dataclass
class RulePattern:
    """Kural tabanlı modda bir istek kalıbı → araç çağrısı."""
    patterns: List[str]
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    sequence: List[Dict[str, Any]] = field(default_factory=list)


#: Dil modeli yokken kullanılan gömülü yönlendirici.
#:
#: Dürüst sınır: bu mod **serbest akıl yürütme yapamaz**. Yalnızca yaygın,
#: açıkça ifade edilmiş istekleri araçlara eşler. Karmaşık görevler için bir
#: dil modeli gerekir; bu mod sistemin tamamen kullanılamaz olmasını önler.
RULE_PATTERNS: List[RulePattern] = [
    RulePattern(
        ["proje haritası", "projeyi haritala", "proje yapısı", "workspace map",
         "neler var", "proje nedir", "yapıyı çıkar"],
        "code.workspace_map", {"max_files": 200},
        "proje yapısını haritalar"),
    RulePattern(
        ["sürümler", "sürüm nedir", "hangi araçlar", "araç zinciri", "versiyon",
         "toolchain", "neler kurulu"],
        "shell.versions", {},
        "kurulu araçların sürümlerini listeler"),
    RulePattern(
        ["testleri çalıştır", "test çalıştır", "testleri koş", "run tests",
         "testleri geçir"],
        "code.run_tests", {},
        "testleri çalıştırır"),
    RulePattern(
        ["git durumu", "git status", "neler değişti", "değişiklikler"],
        "code.git_status", {},
        "git durumunu okur"),
    RulePattern(
        ["git farkı", "diff", "farkı göster"],
        "code.git_diff", {},
        "git farkını okur"),
    RulePattern(
        ["bekle", "durum", "hazır mısın", "ne yapabilirsin", "yetenekler"],
        "", {}, "yetenek özeti", sequence=[]),
]


class RuleRouter:
    """
    Dil modeli yokken çalışan gömülü yönlendirici.

    Kullanıcının isteğini kalıplarla eşler ve araç çağrıları üretir. `args`
    içinde `{{path}}` gibi yer tutucular varsa istekten çıkarılır.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def route(self, task: str) -> List[Dict[str, Any]]:
        """Görevi araç çağrılarına eşler."""
        lowered = (task or "").lower()
        calls: List[Dict[str, Any]] = []

        # Yol/klasör okuma: "X dosyasını oku", "X.py içeriği"
        path = _extract_path(task)
        if path:
            calls.append({"tool": "code.read_file",
                          "args": {"path": path, "max_lines": 300}})

        # Arama: "X nerede kullanılıyor", "X ara"
        symbol = _extract_symbol(task)
        if symbol and any(word in lowered for word in
                          ("ara", "bul", "nerede", "kullanıl", "search", "find")):
            calls.append({"tool": "code.find_symbol", "args": {"name": symbol}})

        for rule in RULE_PATTERNS:
            if not rule.tool:
                calls.append({"tool": "", "args": {}, "description": rule.description})
                continue
            if any(pattern in lowered for pattern in rule.patterns):
                if self._registry.has(rule.tool):
                    arguments = dict(rule.args)
                    if path and "path" in arguments:
                        arguments["path"] = path
                    calls.append({"tool": rule.tool, "args": arguments,
                                  "description": rule.description})

        # Komut çalıştırma: "şu komutu çalıştır: ..."
        command = _extract_command(task)
        if command and self._registry.has("shell.run"):
            calls.append({"tool": "shell.run",
                          "args": {"command": command, "timeout": 120.0}})

        # Tekrar edenleri temizle
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for call in calls:
            key = (call.get("tool", ""), json.dumps(call.get("args", {}), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        return unique

    def describe_capabilities(self) -> str:
        """Dil modeli yokken kullanıcıya ne yapabildiğimizi anlatır."""
        tools = self._registry.list_tools()
        by_category: Dict[str, List[str]] = {}
        for tool in tools:
            by_category.setdefault(tool["category"], []).append(tool["name"])
        lines = [
            "**Dil modeli yapılandırılmadı.** Şu anda kural tabanlı modda "
            "çalışıyorum: serbest akıl yürütemem, ama şu işleri yapabilirim:",
            "",
        ]
        labels = {
            "code": "Kod", "shell": "Kabuk", "vision": "Görsel doğrulama",
            "app": "Uygulama kontrolü", "knowledge": "Bilgi kasası",
            "godot": "Godot", "blender": "Blender",
        }
        for category, names in sorted(by_category.items()):
            label = labels.get(category, category)
            lines.append(f"- **{label}** ({len(names)} araç): "
                         f"{', '.join('`' + n + '`' for n in sorted(names)[:6])}"
                         + (" …" if len(names) > 6 else ""))
        lines.extend([
            "",
            "Serbest akıl yürütme ve karmaşık çok adımlı görevler için "
            "**Ayarlar → Dil Modeli** bölümünden bir API anahtarı girin "
            "(DeepSeek, OpenAI, Groq, OpenRouter) veya yerel bir Ollama/LM Studio "
            "sunucusu başlatın.",
        ])
        return "\n".join(lines)


# ─── Araç Seçimi ────────────────────────────────────────────────────────────

#: Her görevde gönderilen temel araçlar (hiçbir görevde eksik olmamalı).
CORE_TOOLS: Tuple[str, ...] = (
    "code.workspace_map",
    "code.read_file",
    "code.list_directory",
    "code.search_text",
    "code.find_files",
    "code.edit_file",
    "code.create_file",
    "code.run_command",
    "code.run_tests",
    "code.diagnose_error",
    "shell.run",
    "shell.session_run",
    "shell.diagnose",
    # ⚠ tur 14 (Hedef C): araştırma araçları — istediği dilin/kütüphanenin
    # güncel sürümünü ve API'sini internetten bulup okuyabilir.
    "doc.kutuphane_ara",
    "doc.api_imzasi_al",
    "doc.surum_karsilastir",
    # ⚠ tur 23: Godot 4 API bağlam enjektörü (retrieval).
    # Ölçüldü: bağlamsız model 3 uydurma API üretiyor
    # (`Seed`, `SetNoiseScale`, `AddNoise`); bağlamlı 0.
    "doc.godot_api_baglam",
    # ⚠ tur 41: Godot 4 KOD ONARIM döngüsü — üretilen kodu gerçek
    # derleyiciyle doğrular, hatayı bağlama geri besler, yeniden üretir.
    # Ölçüldü: 0/4 → 4/4 derleme (ortalama tur 1,2).
    "code.godot_onar",
    # ⚠ tur 10: Blender VARLIK ÜRETİMİ — kalite kademesi + kategori.
    # Ölçüldü: 6 kategori · 12 ürün · 3 kalite kademesi.
    "asset.uret",
    # ⚠ tur 10: Blender ANİMASYON — 5 tarif (rol tabanlı).
    # Ölçüldü: 4 varlık × 5 animasyon = 20/20 dolu action.
    "asset.animasyon",
    # ⚠ tur 10: Varlık kataloğu — kategori/meta sorgu.
    "asset.katalog",
    # ⚠ tur 11: DOKU — ara, puanla, seçime sun, UYGULA.
    # Ölçüldü: 7 prosedürel (7/7 derlendi) + yerel görüntü kütüphanesi.
    "asset.doku",
    # ⚠ tur 13: ANİMASYON GEÇİŞLERİ — action'lar arası smoothstep blend
    # + NLA yerleşimi. Ölçüldü: 7 geçiş · 12 NLA izi.
    "asset.gecis",
    # ⚠ tur 14: VARLIK ÖNİZLEME — 4 açıdan render (görsel doğrulama).
    # Ölçüldü: 4/4 açı · ~3 sn · EEVEE · 512x512 PNG.
    "asset.onizleme",
    # ⚠ tur 18: ANİMASYON DOĞRULAMA — hareket var mı, kaç kemik?
    # Ölçüldü: 23 animasyon · NORMAL 16 · GEÇİŞ DÜZ 7 · HATA 0.
    "asset.anim_onizleme",
    # ══ ⚠ Aşama 5: ARAŞTIRMA — artık **çekirdek** araçlar ═══════════════
    #
    # Ölçüldü: bu üçü kayıt defterinde **vardı** (`launcher.py:278`
    # `KnowledgeWorkflow.tools()`), ama `CORE_TOOLS`'ta olmadıkları için
    # yalnızca "araştır/web/internet" anahtar sözcüğü geçen görevlerde
    # listeye giriyorlardı. Sonuç: ajan **kendiliğinden** araştırmıyordu.
    "web.search",      # çevrimiçi arama
    "web.fetch",       # sayfa içeriğini çek
    "knowledge.ask",   # kasa + belge bilgisine sor
)

#: Anahtar sözcük → araç kategorisi/ön eki eşlemesi.
#:
#: Neden gerekli? 63 aracın **tamamını** her istekte göndermek:
#:   1. Sağlayıcının istek boyutu sınırını aşar (Groq ücretsiz katmanında
#:      `HTTP 413 Request too large`),
#:   2. Ücretsiz katmanların dakikalık token sınırını hızla tüketir
#:      (`HTTP 429`),
#:   3. Modelin doğru aracı seçmesini **zorlaştırır** (63 seçenek arasından
#:      seçim yapmak, 15 seçenek arasından seçmekten belirgin biçimde zordur).
#: Bu yüzden göreve uygun araçlar seçilip geri kalanı gönderilmez.
TOOL_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    # ⚠ Araştırma ipucu **başta** olmalı: "araştır" içeren bir görevde
    # `godot.`/`code.` araçları önce eklenirse 24'lük bütçe dolar ve web
    # araçları listeye hiç girmez. Ölçüldü: "Godot ... FARKI araştır"
    # görevinde ajan `web.search`/`web.fetch` araçlarını **göremedi** ve
    # interneti hiç kullanmadı (16 adımda shell/godot ile idare etti).
    (("araştır", "arastir", "internet", "web", "online", "çevrimiçi",
      "güncel bilgi", "guncel bilgi", "kaynak bul", "resmî", "resmi",
      "doküman", "dokuman", "makale", "fetch", "url", "site"),
     ("web.", "knowledge.", "learn.")),
    (("godot", "gdscript", "sahne", "scene", ".tscn", "oyun motoru", "tcnen"),
     ("godot.",)),
    (("blender", "3b", "model", "render", "mesh", "malzeme", "küp", "kure",
      "sahne kur", ".blend", "gltf", "fbx", "animasyon",
      # ⚠ Ölçülen hata: arazi/prosedürel işler bu listede yoktu, bu yüzden
      # `blender.*` ailesi hiç seçilmiyordu. Ölçüldü: "terrain, grid, noise,
      # erozyon" içeren bir görevde ajan `blender.run_script` aracını
      # **göremedi** ve işi yapamayacağını bildirdi (oysa araç kayıtlıydı).
      "terrain", "arazi", "grid", "izgara", "noise", "gürültü", "gurultu",
      "erozyon", "erosion", "yükseklik", "yukseklik", "heightmap", "dağ",
      "dag", "tepe", "vadi", "nehir", "su", "yol", "çim", "cim", "ağaç",
      "agac", "çiçek", "cicek", "bitki", "lowpoly", "low-poly", "prosedürel",
      "procedural", "vertex color", "shade flat", "peyzaj", "manzara",
      "biome", "biyom", "katman", "texture", "doku"),
     ("blender.",)),
    (("ekran", "görüntü", "pencere", "arayüz", "ui", "buton", "görsel",
      "screenshot", "ocr", "modal"),
     ("vision.",)),
    # ⚠ tur 14 (Hedef C): araştırma araçları — kütüphane/sürüm/API arama.
    # ⚠ Ölçülen ders (tur 8): `TOOL_HINTS` eşlemesi olmadan araç
    # **kayıtlı olsa bile seçilmiyor** ("terrain" görevi `blender.*`
    # aracını göremedi).
    (("kütüphane", "kutuphane", "api", "imza", "signature",
      "sürüm", "surum", "versiyon", "version", "doküman", "dokuman",
      "docs", "referans", "documentation", "hangi sürüm",
      "nasıl kullanılır", "kullanımı", "fonksiyonu", "metodu",
      "library", "framework", "paket", "module", "modül"),
     ("doc.",)),
    # ⚠ tur 23: Godot 4 API bağlamı — ölçüldü ki bu eşleme olmadan
    # araç **kayıtlı olsa bile seçilmiyor** (tur 8 dersi).
    (("godot", "gdscript", "gd betik", "sahne", "node3d", "node2d",
      "karakter gövdesi", "characterbody", "malzeme", "material",
      "gürültü", "noise", "tween", "animasyon", "ara sahne", "tilemap",
      "godot 4", "godot4", "class_name", "@export", "@onready",
      "fastnoiselite", "standardmaterial3d", "meshinstance",
      "are3d", "area3d", "rigidbody", "kamerası", "camera3d"),
     ("doc.godot_api_baglam",)),
    # ⚠ tur 41: kod onarım döngüsü — derleme hatalarını düzeltir.
    (("derleme hatası", "derleme hatasi", "parse error", "derlenmiyor",
      "sözdizimi hatası", "sozdizimi hatasi", "onar", "düzelt",
      "duzelt", "hata veriyor", "çalışmıyor", "calismiyor",
      "godot hata", "gdscript hata", "syntax error"),
     ("code.godot_onar",)),
    # ⚠ tur 10: Blender varlık üretimi — kategori + kalite.
    (("blender", "kaya oluştur", "kaya olustur", "ağaç oluştur",
      "agac olustur", "mesh oluştur", "mesh olustur", "model oluştur",
      "model olustur", "low-poly", "low poly", "lowpoly", "kaliteli",
      "varlık üret", "varlik uret", "asset", "3d model", "sahne kur",
      "malzeme oluştur", "material", "doku ekle", "texture",
      # ⚠ tur 10: yaratık/karakter sözcükleri (ölçüldü: kaçıyordu)
      "timsah", "canavar", "monster", "yaratık", "yaratik", "ejder",
      "karakter oluştur", "karakter yap", "insan", "humanoid",
      "sandık", "sandik", "fıçı", "fici", "masa", "kılıç", "kilic",
      "duvar", "kapı", "kapi", "sütun", "sutun", "zemin", "çalı", "cali",
      "çim", "cim", "yarat", "üret", "uret", "oluştur", "olustur",
      # ⚠ tur 26 ÖLÇÜMÜ: 17 yeni istekte 5 kaçak bulundu.
      # `kayadan bir kule yap` → [] (asset.uret gelmeliydi)
      "kule", "kayadan", "taş yap", "tas yap", "yap"),
     ("asset.uret",)),
    # ⚠ tur 10: Animasyon.
    (("animasyon", "anime et", "anim", "yürüyüş", "yuruyus", "koşma",
      "kosma", "saldırı", "saldiri", "ölüm anim", "olum anim", "bekleme",
      "idle", "rig", "armature", "kemik", "iskelet",
      # ⚠ tur 26 ÖLÇÜMÜ: `karakteri zıplat` / `insanı tırmandır` kaçtı
      "zıplat", "ziplat", "zıplama", "ziplama", "tırmandır",
      "tirmandir", "tırmanma", "tirmanma", "yüzdür", "yuzdur", "yüzme",
      "yuzme", "diril", "diriltme", "hasar al", "hasar"),
     ("asset.animasyon",)),
    # ⚠ tur 18: Animasyon doğrulama — hareket ölçümü.
    (("animasyon doğrula", "hareket var mı", "animasyon çalışıyor",
      "animasyon kontrol", "kemik hareket", "anim test",
      "animasyon bozuk mu", "hareket ölç",
      # ⚠ tur 26 ÖLÇÜMÜ: `animasyonlar çalışıyor mu` kaçtı
      # ⚠ Sıra ÖNEMLİ: bu blok `asset.animasyon`'dan ÖNCE gelmeli
      "animasyonlar çalışıyor", "animasyon çalışıyor mu",
      "animasyonlar doğru", "hareket ediyor mu", "anim doğru"),
     ("asset.anim_onizleme",)),
    # ⚠ tur 14: Önizleme — render / görsel doğrulama.
    (("önizleme", "onizleme", "render", "görüntüle", "goruntule",
      "görsel", "gorsel", "resim", "png", "nasıl görünüyor",
      "nasil gorunuyor", "preview", "ekran görüntüsü",
      # ⚠ tur 26 ÖLÇÜMÜ: `modeli önizle` kaçtı (yalnız "önizleme" vardı)
      "önizle", "onizle", "göster", "goster"),
     ("asset.onizleme",)),
    # ⚠ tur 26 ÖLÇÜMÜ: `varlığın UV'sini aç` ve `varlığı diske kaydet`
    # kaçtı → UV ve kaydet anahtarları eklendi.
    # ⚠ UV işi `asset.uret` içinde otomatik yapılır.
    (("uv", "uv aç", "uv ac", "uvsini", "dikiş", "dikis", "seam",
      "uv açma", "uv acma", "uv ayarla", "unfold", "unwrap"),
     ("asset.uret",)),
    (("kaydet", "kayıt", "diske yaz", "kataloğa ekle", "kataloga ekle",
      "meta veri", "katalogla"),
     ("asset.katalog",)),
    # ⚠ tur 13: Geçiş — action'lar arası blend + NLA.
    (("geçiş", "gecis", "blend", "animasyon geçişi", "nla",
      "animasyon bağla", "animasyon zincirle", "yumuşak geçiş",
      "animasyon sırala", "transition"),
     ("asset.gecis",)),
    # ⚠ tur 11: Doku — ara / seç / uygula.
    (("doku", "texture", "kaplama", "texture ekle", "doku ekle",
      "doku uygula", "doku ara", "doku bul", "kaplama ekle",
      "pürüzlülük haritası", "normal map", "uv doku"),
     ("asset.doku",)),
    # ⚠ tur 10: Varlık kataloğu.
    (("varlık katalog", "varlik katalog", "katalog", "hangi varlık",
      "varlıkları listele", "varliklari listele", "üretilen modeller",
      "uretilen modeller", "asset list", "kaç varlık", "kac varlik"),
     ("asset.katalog",)),
    (("test", "pytest", "unittest", "doğrula", "kontrol et"),
     ("code.run_tests", "code.check_syntax", "code.diagnose_error")),
    (("komut", "çalıştır", "powershell", "terminal", "kabuk", "kur", "derle",
      "build", "install", "git"),
     ("shell.", "code.run_command")),
    (("ara", "bul", "nerede", "kullanıl", "search", "sembol", "fonksiyon",
      "sınıf", "tanım"),
     ("code.search_text", "code.find_symbol", "code.list_symbols")),
    (("düzenle", "değiştir", "yaz", "ekle", "oluştur", "düzelt", "refactor",
      "edit", "fix"),
     ("code.edit_file", "code.create_file", "code.revert", "code.check_syntax")),
)


#: Sağlayıcı başına istek bütçesi.
#:
#: Neden gerekli? Ücretsiz katmanların **dakikalık token sınırı** vardır ve
#: bu sınır, araç tanımlarının toplamıyla kolayca aşılır. Ölçülen değer:
#: Groq ücretsiz katmanı `x-ratelimit-limit-tokens: 8000` (dakikada). Tek bir
#: istekte 16 araç şeması (~2400 token) + `max_tokens: 8192` göndermek sınırı
#: aşar ve `HTTP 429` alınır. Bu yüzden hem araç sayısı hem `max_tokens`
#: sağlayıcıya göre ayarlanır.
PROVIDER_BUDGETS: Dict[str, Dict[str, int]] = {
    "groq": {"max_tools": 12, "max_tokens": 2048},
    "gemini": {"max_tools": 20, "max_tokens": 8192},
    "deepseek": {"max_tools": 24, "max_tokens": 8192},
    "openai": {"max_tools": 24, "max_tokens": 8192},
    "openrouter": {"max_tools": 20, "max_tokens": 4096},
    "together": {"max_tools": 20, "max_tokens": 4096},
}

#: Bilinmeyen/yerel sağlayıcılar için varsayılan bütçe
DEFAULT_BUDGET: Dict[str, int] = {"max_tools": 22, "max_tokens": 8192}


#: Göreve gönderilecek varsayılan araç bütçesi.
#:
#: ⚠ ÖLÇÜLDÜ: `CORE_TOOLS` **28** araç ama bütçe **22** idi → her görevde
#: en az 6 çekirdek araç listeye **hiç girmiyordu** (ölçüm: `asset.animasyon`
#: 4 farklı görevde 0/4 seçildi). Bütçe çekirdeği kapsayacak şekilde
#: 34'e çıkarıldı (sağlayıcı sınırı ~63 araç; 413 riski yok).
VARSAYILAN_ARAC_BUTCESI = 40


def select_tools(task: str, registry: ToolRegistry,
                 max_tools: int = VARSAYILAN_ARAC_BUTCESI) -> List[str]:
    """
    Göreve uygun araç adlarını seçer.

    Sıralama önemlidir: **önce göreve özel araçlar**, sonra temel araçlar.
    Aksi hâlde liste sonunda `max_tools` ile kesilirken göreve özel araçlar
    (vision, Godot, Blender) düşer ve ajan o yeteneği hiç göremez.

    Args:
        task: Kullanıcının isteği
        registry: Araç kayıt defteri
        max_tools: Gönderilecek en fazla araç sayısı

    Returns:
        Araç adları listesi (öncelik sırasına göre)
    """
    lowered = (task or "").lower()
    selected: List[str] = []

    def add(name: str) -> None:
        if name not in selected and registry.has(name):
            selected.append(name)

    # 1) Göreve özel araçlar (kesilmemeleri için başa konur)
    for keywords, prefixes in TOOL_HINTS:
        if not any(keyword in lowered for keyword in keywords):
            continue
        for name in registry.names():
            if any(name.startswith(prefix) or name == prefix for prefix in prefixes):
                add(name)

    # 2) Temel araçlar (her görevde gerekli)
    for name in CORE_TOOLS:
        add(name)

    # 3) **Yürütme araçları** — göreve özel ailede önce bunlar gelmeli
    #
    # ⚠ Ölçülen sorun: `registry.names()` alfabetik döner. Bir ailede
    # bütçeden fazla araç varsa, alfabetik olarak **sonda** kalan yürütme
    # araçları listeden düşer. Ölçüldü: Blender görevinde 13 `blender.*`
    # aracından `blender.render`, `blender.run_script` ve
    # `blender.validate_script` **listeye hiç girmedi**; ajan "elimde
    # run_script yok" diyerek işi yapamadı ve dürüstçe reddetti.
    #
    # Bu yüzden yürütme/doğrulama/render araçları **öncelikli** eklenir.
    if len(selected) < max_tools:
        for name in registry.names():
            if len(selected) >= max_tools:
                break
            if name.endswith((".run_script", ".validate_script", ".render",
                              ".run_expression", ".run_command", ".python",
                              ".build_scene", ".health")):
                add(name)

    # 4) Kalan yeri kod/kabuk ailesinden tamamla
    if len(selected) < max_tools:
        for name in registry.names():
            if len(selected) >= max_tools:
                break
            if name.startswith(("code.", "shell.")):
                add(name)

    # 5) Hâlâ yer varsa diğer kategorilerden ekle
    if len(selected) < max_tools:
        for name in registry.names():
            if len(selected) >= max_tools:
                break
            add(name)

    # ⚠ MCP araçları her zaman listeye girer (sunucu başına 1 araç, az sayıda):
    # ölçüldü — `mcp.sahte.kullan` kayıtlı olmasına rağmen seçime girmediği
    # için ajan "bende böyle bir araç yok" dedi.
    # ⚠ Bütçeden bağımsız: MCP sunucusu başına 1 araç (az sayıda) ve
    # ajan bunları görmeli. Ölçüldü: bütçe 40 doluyken hiç eklenmiyordu.
    for name in registry.names():
        if name.startswith("mcp."):
            add(name)

    # ⚠ KESME KURALI: çekirdek araçlar **asla** düşmez.
    # Ölçüldü: düz `selected[:max_tools]` kırpması çekirdeğin sonundaki
    # `asset.animasyon` / `asset.onizleme` gibi araçları atıyordu.
    # ⚠ MCP araçları da çekirdek gibi korunur (kesilirse ajan göremez)
    cekirdek = ([n for n in CORE_TOOLS if registry.has(n)]
                + [n for n in registry.names() if n.startswith("mcp.")])
    ozel = [n for n in selected if n not in cekirdek]
    butce = max(int(max_tools), len(cekirdek))
    yer = max(0, butce - len(cekirdek))
    # Göreve özel araçlar önce gelir (kesilme riski yok), çekirdek sonra
    return (ozel[:yer] + cekirdek)[:butce]




class AgentLoop:
    """
    Planla → uygula → gözlemle döngüsü.

    Kullanım:
        loop = AgentLoop()
        loop.initialize(registry, router, workspace="E:/proje",
                        on_step=lambda step: print(step.text),
                        approval_callback=ask_user)

        run = loop.run("Bu projedeki testleri çalıştır ve hataları düzelt")
        print(run.render())

    Args:
        on_step: Her adımda çağrılır (arayüze canlı akış için)
        approval_callback: (tool, args, safety, reason) -> (bool, bool)
            döndürür: (onaylandı, hatırla)
    """

    name = "agent.loop"

    #: Aynı araç+argüman kaç kez tekrarlanırsa döngü kabul edilir
    REPEAT_LIMIT = 3

    #: Model art arda kaç kez boş yanıt verirse çalışma durdurulur
    MAX_EMPTY_RESPONSES = 3

    #: Hız sınırında (429) en fazla kaç kez yeniden denenecek
    MAX_RATE_LIMIT_RETRIES = 3

    #: Her yeniden deneme öncesi beklenecek taban süre (saniye); katları alınır
    RATE_LIMIT_WAIT_SECONDS = 20.0

    #: ⚠ Düşünme döngüsünde kaç **farklı** plan adayı üretilecek.
    #: Ölçüldü: yerel model akıcı metin üretemiyor ama **seçebiliyor**
    #: (Blender kodunda %70, yaklaşımda %50; rastgele %25). Bu yüzden plan
    #: yazdırmak yerine adaylar arasından **seçtiriyoruz**.
    PLAN_CANDIDATES = 3

    #: ⚠ Otomatik plan eşiği: bu uzunluktan itibaren düşünme döngüsü
    #: kendiliğinden devreye girer (kısa sorular için gereksiz maliyet).
    AUTO_PLAN_CHARS = 60

    def __init__(self) -> None:
        self._registry: Optional[ToolRegistry] = None
        self._router: Optional[LLMRouter] = None
        self._rules: Optional[RuleRouter] = None
        self._workspace: str = ""
        self._config: Any = None
        self._event_bus: Any = None
        self._initialized = False
        #: ⚠ Düşünme döngüsü (bulut üretir → yerel seçer → belirsizse buluta
        #: düşer). `None` ise plan tek çağrıyla alınır (eski davranış).
        self._thinker: Any = None
        #: ⚠ Ortak havuz (komşu düğümler). `None` ise iş paylaşımı kapalı.
        self._peer: Any = None
        #: ⚠ Bilişsel katman (hata taksonomisi · yansıma · doğrulama).
        #: `None` = katman kapalı (eski davranış korunur).
        self._cog: Optional[CognitionLayer] = None
        #: Çalışma başına hedef durumu (olgular, denemeler, açık sorular)
        self._hedef: Optional[HedefDurumu] = None
        #: ⚠ Puzzle: iç içe fazlar + walkthrough'lar (agent/puzzle.py)
        self._puzzle: Any = None
        #: ⚠ Kontrol noktası: kesilen işi kaldığı yerden sürdürmek için
        self._kontrol: Any = None
        self._kontrol_yol: Any = None
        self._kontrol_arsiv: Any = None
        #: ⚠ A) Walkthrough cevabı bekleniyor mu (çalışma durur)
        self._puzzle_bekle = False
        #: B) Dinamik genişletme için planlayıcı (tembel kurulur)
        self._puzzle_planlayici: Any = None
        #: Puzzle planlaması açık mı (NC_PUZZLE=0 kapatır)
        self._puzzle_aktif = True
        #: ⚠ Playbook deposu (başarılı iş akışları → prosedür)
        self._playbooks: Optional[PlaybookStore] = None
        #: Hiyerarşik planlama açık mı (uzun görevlerde)
        self._hier_aktif = True
        #: Bu çalışmada kullanılan playbook (kullanım sayacı için)
        self._kullanilan_playbook = ""
        #: Son plan turunun düşünme metadatası (denetim izi için)
        self._last_plan_meta: Dict[str, Any] = {}

        self._on_step: Optional[Callable[[AgentStep], None]] = None
        self._on_run: Optional[Callable[[AgentRun], None]] = None
        self._approval: Optional[Callable[..., Any]] = None
        self._cancel = threading.Event()
        self._cancel_hooks: List[Callable[[], Any]] = []
        self._lock = threading.RLock()

        self._counter = {
            "runs": 0, "steps": 0, "tool_calls": 0, "approvals": 0,
            "denials": 0, "errors": 0, "cancelled": 0, "rule_runs": 0,
            # ⚠ ortak havuz sayaçları (adım 3)
            "peer_calls": 0, "peer_ok": 0, "peer_fallback": 0,
            # ⚠ bilişsel katman sayaçları
            "cog_reflect": 0, "cog_replan": 0, "cog_verify": 0,
            "cog_correct": 0, "cog_recovery": 0,
            # ⚠ playbook / hiyerarşi / sıkıştırma
            "playbook_hit": 0, "playbook_saved": 0,
            "hier_plan": 0, "hier_fail": 0, "context_compressed": 0,
            "faz_raporu": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, registry: ToolRegistry, router: Optional[LLMRouter] = None,
                   workspace: str = "", config: Any = None, event_bus: Any = None,
                   on_step: Optional[Callable[[AgentStep], None]] = None,
                   approval_callback: Optional[Callable[..., Any]] = None) -> None:
        """Döngüyü başlatır (idempotent)."""
        if self._initialized:
            return
        self._registry = registry
        self._router = router
        self._rules = RuleRouter(registry) if registry is not None else None
        self._workspace = workspace
        self._config = config
        self._event_bus = event_bus
        self._on_step = on_step
        self._approval = approval_callback

        # ⚠ Bilişsel katman: varsayılan AÇIK; `NC_COG=0` ile kapatılır.
        # Sınırlar: en fazla 2 yansıma, en fazla 1 düzeltme.
        import os as _os

        if (_os.environ.get("NC_COG") or "1").strip() not in ("0", "false", "off"):
            try:
                self._cog = CognitionLayer(
                    max_reflect=int(_os.environ.get("NC_COG_REFLECT") or 2),
                    verify=(_os.environ.get("NC_COG_VERIFY") or "1").strip()
                    not in ("0", "false", "off"),
                    reflect_every=int(_os.environ.get("NC_COG_EVERY") or 8),
                )
                self._cog.initialize(router=self._router)
            except Exception as exc:
                logger.warning(f"Bilişsel katman kurulamadı: {exc}")
                self._cog = None

        # ⚠ Playbook deposu: <workspace>/.neurocore/playbooks.jsonl
        # ⚠ Puzzle: iç içe fazlar + walkthrough (NC_PUZZLE=0 kapatır)
        # ⚠ Kontrol noktası yolları (kesilen işi sürdürmek için)
        try:
            from pathlib import Path as _PK

            _kok = _PK(workspace or ".") / ".neurocore"
            self._kontrol_yol = _kok / "kontrol.json"
            self._kontrol_arsiv = _kok / "kontrol_gecmis.jsonl"
            self._puzzle_yol = _kok / "puzzle.json"
            self._proje_yol = _kok / "proje.json"
        except Exception:
            self._kontrol_yol = None
            self._kontrol_arsiv = None

        self._puzzle_aktif = (
            _os.environ.get("NC_PUZZLE") or "1").strip() not in (
                "0", "false", "off")
        self._hier_aktif = (_os.environ.get("NC_HIER") or "1").strip() not in (
            "0", "false", "off")
        try:
            from pathlib import Path as _P

            kok = _P(workspace or ".") / ".neurocore"
            self._playbooks = PlaybookStore(kok / "playbooks.jsonl")
        except Exception as exc:
            logger.debug(f"Playbook deposu kurulamadı: {exc}")
            self._playbooks = None
        self._initialized = True
        logger.info(
            f"Ajan döngüsü hazır — {len(registry) if registry else 0} araç, "
            f"mod: {'araç çağrısı' if (router and router.supports_tools) else 'kural tabanlı'}"
        )

    def shutdown(self) -> None:
        """Döngüyü durdurur."""
        self.cancel()
        self._initialized = False
        logger.info(f"Ajan döngüsü kapatıldı ({self._counter['runs']} çalışma)")

    def attach_thinker(self, thinker: Any) -> None:
        """
        Düşünme döngüsünü bağlar (bulut üretir → yerel model seçer).

        ⚠ Neden ayrı metot? `initialize()` idempotent ve çağrı sırası
        launcher'da **döngü → köprü → düşünme** biçiminde. Düşünme döngüsü
        sonradan kurulabildiği için ayrı bağlanır. `None` verilirse eski
        davranış (tek çağrılık plan) korunur.
        """
        self._thinker = thinker
        logger.info(
            "Ajan döngüsüne düşünme eklendi"
            if thinker is not None else
            "Düşünme kaldırıldı — plan tek çağrıyla alınacak"
        )

    def attach_peer(self, havuz: Any) -> None:
        """
        Ortak havuzu bağlar: izinli araçlar **önce komşuda** denenir.

        ⚠ Yalnız okuma araçları (`IZINLI_ARACLAR`) uzakta çalıştırılır;
        yanıt gelmezse çağrı **yerelde** yapılır (iş kaybolmaz).

        Args:
            havuz: `PeerPool` benzeri nesne (`komsu_dene` metodu olmalı)
        """
        self._peer = havuz
        logger.info(
            "Ajan döngüsüne ortak havuz bağlandı"
            if havuz is not None else
            "Ortak havuz kaldırıldı — tüm işler yerel"
        )

    def _count_plan_meta(self) -> Dict[str, Any]:
        """Son plan turunun düşünme metadatası (boş olabilir)."""
        return dict(self._last_plan_meta)

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "workspace": self._workspace,
            "mode": self.mode,
            "tools": len(self._registry) if self._registry else 0,
            "cancelled_flag": self._cancel.is_set(),
        }

    @property
    def mode(self) -> str:
        """Aktif çalışma modu."""
        if self._router is None or not self._router.available:
            return "rules"
        return "tools" if self._router.supports_tools else "prompt"

    def cancel(self) -> None:
        """
        Çalışan görevi iptal eder.

        Yalnızca bayrak kaldırmak **yetmez**: o anda çalışan bir komut
        (ör. 600 saniyelik bir sunucu) iptalden sonra da devam ederdi. Bu
        yüzden iptal geri çağrımları da tetiklenir — kayıtlı geri çağrımlar
        çalışan alt süreçleri öldürür.
        """
        self._cancel.set()
        killed = 0
        for hook in list(self._cancel_hooks):
            try:
                outcome = hook()
                if isinstance(outcome, int):
                    killed += outcome
            except Exception as exc:
                logger.debug(f"İptal geri çağrımı hata verdi: {exc}")
        self._emit("agent.cancel_requested", {"killed_processes": killed})

    def add_cancel_hook(self, hook: Callable[[], Any]) -> None:
        """
        İptal edildiğinde çağrılacak geri çağrım ekler.

        Uygulama bunu kullanarak kabuk katmanını bağlar: "Durdur"
        basıldığında çalışan komutlar gerçekten sonlandırılır.
        """
        if hook not in self._cancel_hooks:
            self._cancel_hooks.append(hook)

    def reset_cancel(self) -> None:
        """İptal bayrağını temizler."""
        self._cancel.clear()

    def set_approval_callback(self, callback: Optional[Callable[..., Any]]) -> None:
        """Onay geri çağrımını değiştirir."""
        self._approval = callback

    def set_step_listener(self, listener: Optional[Callable[[AgentStep], None]]) -> None:
        """Adım dinleyicisini değiştirir."""
        self._on_step = listener

    def set_run_listener(self, listener: Optional[Callable[["AgentRun"], None]]) -> None:
        """
        Çalışma bittiğinde çağrılacak dinleyiciyi ayarlar.

        Öğrenme kaydedicisi bunu kullanır: her çalışma deneyime, kasaya ve
        eğitime dönüşür. Dinleyici hatası çalışmayı **etkilemez**.
        """
        self._on_run = listener

    # ─── Ana Giriş ──────────────────────────────────────────────────────

    #: ⚠ KULLANICI KARARI: görevler **fazlara** bölünüyor ve ajan her fazda
    #: durumu bildirip devam ediyor; sabit adım tavanı işi erken kesiyordu.
    #: Artık varsayılan **sınırsız** (0); güvenlik için yalnız `MUTLAK_ADIM`
    #: ve süre sınırı kalır. Tekrardan korunma REPEAT_LIMIT ile sürer.
    VARSAYILAN_ADIM = 0

    #: Kaçıncı araç adımında faz raporu verilir (0 = kapalı)
    FAZ_ADIM = 12

    def run(self, task: str, *, max_steps: int = VARSAYILAN_ADIM,
            max_seconds: float = 600.0,
            history: Optional[Sequence[LLMMessage]] = None,
            project_context: str = "", plan_first: bool = False) -> AgentRun:
        """
        Görevi çalıştırır.

        Args:
            task: Kullanıcının isteği (doğal dil)
            max_steps: Maksimum araç adımı
            max_seconds: Maksimum süre
            history: Önceki konuşma (bağlam için)
            project_context: Çalışma alanı özeti (isteme eklenir)
            plan_first: Önce açık bir plan ürettir (karmaşık görevler)

        Returns:
            `AgentRun`
        """
        if not self._initialized:
            raise RuntimeError("Ajan döngüsü başlatılmadı — initialize() çağırın")

        self.reset_cancel()
        # Kabuk katmanının iptal bayrağını da temizle: önceki görevden kalan
        # "iptal edildi" durumu yeni görevi anında öldürmemeli.
        for hook_reset in list(self._cancel_hooks):
            owner = getattr(hook_reset, "__self__", None)
            reset = getattr(owner, "reset_cancel", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
        started = time.perf_counter()
        run = AgentRun(task=task, mode=self.mode)
        self._counter["runs"] += 1
        # Arayüz adım sayacını bu değerle çizer ("adım 7 / 24")
        self._emit("agent.run_started", {
            "task": task[:200], "mode": run.mode,
            "max_steps": max_steps, "max_seconds": max_seconds,
        })

        if self._router is not None and self._router.available:
            run.provider = self._router.status().get("active", "")
            run.model = self._router.status().get("active_model", "")

        try:
            # ⚠ Otomatik plan (Aşama 5): düşünme döngüsü bağlıysa ve görev
            # yeterince uzunsa planlama **kendiliğinden** devreye girer.
            # Kısa sorularda ek LLM çağrısı gereksiz maliyet olduğu için
            # eşik `AUTO_PLAN_CHARS`.
            if (not plan_first and self._thinker is not None
                    and self.mode != "rules"
                    and len(task.strip()) >= self.AUTO_PLAN_CHARS):
                plan_first = True
                self._emit("agent.auto_plan", {
                    "task": task[:120], "reason": "uzun_gorev_dusunme",
                    "chars": len(task.strip()),
                })

            if self.mode == "rules":
                self._counter["rule_runs"] += 1
                self._run_rules(task, run)
            else:
                # ⚠ Kontrol noktası: devam mı, yeni çalışma mı?
                try:
                    self._kontrol_hazirla(task, run)
                except Exception as exc:
                    logger.debug(f"Kontrol noktası hazırlanamadı: {exc}")

                self._run_model(task, run, max_steps=max_steps,
                                 max_seconds=max_seconds, history=history,
                                 project_context=project_context,
                                 plan_first=plan_first)
        except Exception as exc:
            run.ok = False
            run.error = f"{type(exc).__name__}: {exc}"
            self._counter["errors"] += 1
            self._add_step(run, AgentStep(
                index=len(run.steps), kind="error",
                text=f"beklenmeyen hata: {exc}", ok=False,
            ))
            # ⚠ Hata/kesinti: kaldığı yeri kaydet (devam edilebilsin)
            try:
                self._kontrol_bitir(run, tamamlandi=False,
                                    sebep=f"hata: {type(exc).__name__}")
            except Exception:
                pass
            logger.error(f"Ajan çalışması çöktü: {exc}", exc_info=True)

        run.duration_ms = (time.perf_counter() - started) * 1000
        run.cancelled = self._cancel.is_set()
        if not run.reason:
            run.reason = "tamamlandı" if run.ok else "hata"
        self._emit("agent.run_finished", {
            "task": task[:200], "ok": run.ok, "mode": run.mode,
            "steps": len(run.steps), "duration_ms": run.duration_ms,
            "reason": run.reason,
        })
        # Çalışma dinleyicisi (öğrenme kaydı) — hatası çalışmayı etkilemez.
        if self._on_run is not None:
            try:
                self._on_run(run)
            except Exception as exc:
                logger.debug(f"Çalışma dinleyicisi hata verdi: {exc}")
        logger.info(
            f"Ajan çalışması bitti: {len(run.steps)} adım, "
            f"{run.duration_ms / 1000:.1f}s, mod={run.mode}, neden={run.reason}"
        )
        return run

    # ─── Kural Tabanlı Yol ──────────────────────────────────────────────

    def _run_rules(self, task: str, run: AgentRun) -> None:
        """Dil modeli yokken kalıp eşleme ile araçları çalıştırır."""
        assert self._rules is not None and self._registry is not None

        calls = self._rules.route(task)
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="plan",
            text=f"kural tabanlı yönlendirme: {len(calls)} işlem belirlendi",
        ))

        if not calls or (len(calls) == 1 and not calls[0].get("tool")):
            run.answer = self._rules.describe_capabilities()
            run.reason = "yetenek özeti"
            self._add_step(run, AgentStep(
                index=len(run.steps), kind="answer", text=run.answer,
            ))
            return

        results: List[str] = []
        for call in calls:
            if self._cancel.is_set():
                run.ok = False
                run.reason = "iptal edildi"
                return
            tool_name = str(call.get("tool", ""))
            if not tool_name:
                continue
            result = self._execute(tool_name, dict(call.get("args", {})), run)
            if result is None:
                run.ok = False
                run.reason = "onay reddedildi"
                run.answer = (
                    f"`{tool_name}` çağrısı için onay verilmedi, bu yüzden işlemi "
                    "durdurdum. Onay verirseniz aynı isteği yeniden gönderin."
                )
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="answer", text=run.answer,
                ))
                return
            results.append(self._summarize_result(result))

        # Sonuçları okunur biçimde sun
        run.answer = "\n\n".join(results[:6]) if results else (
            "İsteği somut bir işleme çeviremedim. Proje haritası çıkarma, dosya "
            "okuma, kod arama, test çalıştırma veya araç sürümlerini listeleme "
            "gibi net bir istek yazabilir misiniz?"
        )
        run.reason = "kural tabanlı tamamlandı"
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="answer", text=run.answer[:2000],
        ))

    def _summarize_results(self, results: Sequence[str]) -> str:
        """Birden çok araç sonucunu birleştirir."""
        return "\n\n".join(results[:6])

    def _summarize_result(self, result: ToolResult) -> str:
        """
        Tek bir araç sonucunu **okunur** metne çevirir.

        Neden `ToolResult.render()` kullanılmıyor? Çünkü `render()` metni
        `[OK] araç.adı: {json}` biçiminde paketler ve bu paketi tekrar
        ayrıştırmak kırılgandır (araç adı ile JSON arasındaki iki nokta, JSON
        içindeki iki noktalarla karışır — Windows yolları yüzünden bu her
        seferinde olur). Bunun yerine yapısal veri doğrudan biçimlendirilir.
        """
        tool_name = result.tool

        if result.blocked:
            return (f"**{tool_name}**\n\n> Bu araç çalıştırılmadı: {result.reason}\n\n"
                    "Gerekçe politikadan geliyor (özerklik seviyesi veya yıkıcı "
                    "komut koruması). Kullanıcıdan onay isteyin veya farklı bir "
                    "yaklaşım seçin.")
        if not result.ok:
            lines = [f"**{tool_name}** — hata", "",
                     f"```\n{str(result.error)[:1500]}\n```"]
            if result.reason:
                lines.append(f"\nİpucu: {result.reason}")
            return "\n".join(lines)

        data = result.data
        if data is None:
            return f"**{tool_name}** — tamamlandı"
        if not isinstance(data, dict):
            return f"**{tool_name}**\n\n```\n{str(data)[:2500]}\n```"

        lines = [f"**{tool_name}**", ""]
        lines.extend(self._format_tool_data(tool_name, data))
        return "\n".join(lines)

    def _format_tool_data(self, tool_name: str, data: Dict[str, Any]) -> List[str]:
        """
        Araç çıktısını **insan ve model için okunur** biçime çevirir.

        Genel bir JSON dökümü yerine araca özgü biçimlendirme yapılır: bir
        komut çıktısı kod bloğu, bir test sonucu özet satırı, bir proje
        haritası madde listesi olarak gösterilir. Bu, hem kullanıcının
        okuyabilmesi hem de modelin bir sonraki adımda doğru karar vermesi
        için gereklidir.
        """
        lines: List[str] = []

        # ── Komut çıktıları ──
        if "stdout" in data and "returncode" in data:
            mark = "✓" if data.get("ok") else "✗"
            meta = f"çıkış kodu {data.get('returncode')}"
            if data.get("duration_ms"):
                meta += f", {float(data['duration_ms']):.0f}ms"
            if data.get("timed_out"):
                meta += ", **zaman aşımı**"
            if data.get("blocked"):
                meta += ", **engellendi**"
            lines.append(f"{mark} `{str(data.get('command', ''))[:200]}` — {meta}")
            for key, label in (("stdout", "çıktı"), ("stderr", "hata çıktısı")):
                value = str(data.get(key, "") or "").strip()
                if not value:
                    continue
                if len(value) > 2500:
                    value = value[:2500] + f"\n… (+{len(value) - 2500} karakter)"
                lines.append(f"\n**{label}:**\n```\n{value}\n```")
            for extra in ("diagnosis", "analysis", "fix"):
                block = data.get(extra)
                if not isinstance(block, dict):
                    continue
                summary = block.get("summary") or block.get("suggestion")
                if summary:
                    lines.append(f"\n**teşhis:** {summary}")
                alternative = block.get("alternative_command")
                if alternative:
                    lines.append(f"**önerilen komut:** `{alternative}`")
            return lines

        # ── Test sonuçları ──
        if "framework" in data and "summary" in data:
            total = sum(int(data.get(key, 0) or 0)
                        for key in ("passed", "failed", "errors", "skipped"))
            clean = data.get("ok") and not data.get("failed") and not data.get("errors")
            mark = "✓" if clean else "✗"
            summary = data.get("summary") or (
                f"{total} test bulundu ama özet çıkarılamadı" if total
                else "özet çıkarılamadı — çıktının sonuna bakın"
            )
            lines.append(f"{mark} **{summary}**")
            lines.append(f"- çerçeve: `{data.get('framework')}`")
            lines.append(f"- komut: `{str(data.get('command', ''))[:170]}`")
            if data.get("duration"):
                lines.append(f"- süre: {data['duration']}")
            failures = data.get("failures") or []
            if failures:
                lines.append(f"- başarısız testler ({len(failures)}):")
                lines.extend(f"  - `{name}`" for name in failures[:15])
            tail = str(data.get("stdout_tail") or "").strip() \
                or str(data.get("stderr_tail") or "").strip()
            if tail and not clean:
                snippet = tail[-2000:]
                lines.append(f"\n**çıktı (son kısım):**\n```\n{snippet}\n```")
            return lines

        # ── Dosya okuma ──
        if "content" in data and "total_lines" in data:
            lines.append(f"- dosya: `{data.get('path')}` ({data.get('language')}), "
                         f"{data.get('total_lines')} satır")
            if data.get("note"):
                lines.append(f"- {data['note']}")
            content = str(data.get("content", ""))
            if len(content) > 3000:
                content = content[:3000] + "\n… (kısaltıldı)"
            lines.append(f"\n```{data.get('language', '')}\n{content}\n```")
            return lines

        # ── Düzenleme sonucu ──
        if "replaced" in data and "path" in data:
            mark = "✓" if data.get("ok") else "✗"
            lines.append(f"{mark} `{data.get('path')}` — {data.get('replaced')} değişiklik")
            if data.get("syntax_ok") is False:
                lines.append(f"- **sözdizimi hatası:** {data.get('syntax_error')}")
            elif data.get("syntax_ok") is True:
                lines.append("- sözdizimi doğrulandı")
            if data.get("error"):
                lines.append(f"- hata: {data['error']}")
            diff = str(data.get("diff", ""))
            if diff:
                lines.append(f"\n```diff\n{diff[:2500]}\n```")
            return lines

        # ── Proje haritası ──
        if "project_type" in data:
            languages = data.get("languages") or {}
            lines.append(f"- proje türü: **{data.get('project_type')}** "
                         f"({data.get('project_label', '')})")
            lines.append(f"- dosya: {data.get('file_count')} ({data.get('size_mb')} MB)")
            if languages:
                lines.append("- diller: " + ", ".join(
                    f"{key} ({value})" for key, value in list(languages.items())[:8]))
            directories = data.get("top_directories") or {}
            if directories:
                lines.append("- üst dizinler: " + ", ".join(
                    f"`{key}` ({value})" for key, value in list(directories.items())[:10]))
            for key, label in (("key_files", "anahtar dosyalar"),
                               ("entry_points", "giriş noktaları")):
                values = data.get(key) or []
                if values:
                    lines.append(f"- {label}: "
                                 + ", ".join(f"`{item}`" for item in values[:10]))
            files = data.get("files") or []
            if files:
                lines.append(f"- ilk {min(len(files), 25)} dosya:")
                lines.extend(f"  - `{item.get('relative')}` ({item.get('language')})"
                             for item in files[:25])
            return lines

        # ── Sembol listesi ──
        if "symbols" in data:
            lines.append(f"- {data.get('count')} sembol")
            kinds = data.get("by_kind") or {}
            if kinds:
                lines.append("- türler: " + ", ".join(
                    f"{key} ({value})" for key, value in list(kinds.items())[:8]))
            for symbol in (data.get("symbols") or [])[:40]:
                badge = " *(yaklaşık)*" if symbol.get("approximate") else ""
                lines.append(f"  - `{symbol.get('kind')}` **{symbol.get('qualified')}** "
                             f"satır {symbol.get('line')}{badge}")
            return lines

        # ── Arama sonuçları ──
        if "hits" in data:
            lines.append(f"- {data.get('count')} eşleşme "
                         f"({data.get('files_searched')} dosya tarandı)")
            for hit in (data.get("hits") or [])[:40]:
                lines.append(f"  - `{hit.get('file')}:{hit.get('line')}` "
                             f"{str(hit.get('text', ''))[:120]}")
            return lines

        # ── Dizin listesi / dosya bulma ──
        if "items" in data or "matches" in data:
            entries = data.get("items") or data.get("matches") or []
            lines.append(f"- {len(entries)} girdi")
            for item in entries[:50]:
                suffix = f" ({item.get('language')})" if item.get("language") else ""
                kind = " *(dizin)*" if item.get("type") == "dir" else ""
                lines.append(f"  - `{item.get('relative')}`{suffix}{kind}")
            return lines

        # ── Hata teşhisi ──
        if "errors" in data and "advice" in data:
            lines.append(f"- {data.get('advice')}")
            for error in (data.get("errors") or [])[:12]:
                lines.append(f"  - **{error.get('type')}**: "
                             f"{str(error.get('message', ''))[:150]}")
                if error.get("hint"):
                    lines.append(f"    → {error['hint']}")
            return lines

        # ── Sürüm listesi ──
        if data and all(isinstance(value, dict) and "available" in value
                        for value in data.values()):
            available = {key: value for key, value in data.items() if value.get("available")}
            missing = [key for key, value in data.items() if not value.get("available")]
            lines.append(f"- kurulu: {len(available)} araç")
            for key, value in available.items():
                lines.append(f"  - **{key}**: {str(value.get('version', ''))[:110]}")
            if missing:
                lines.append(f"- kurulu değil: {', '.join(missing)}")
            return lines

        # ── Genel düşüş ──
        for key, value in list(data.items())[:20]:
            if value in (None, "", 0, False, [], {}):
                continue
            if isinstance(value, (dict, list)):
                preview = json.dumps(value, ensure_ascii=False, default=str)
                if len(preview) > 240:
                    preview = preview[:240] + "…"
                lines.append(f"- {key}: `{preview}`")
            else:
                lines.append(f"- {key}: {str(value)[:240]}")
        return lines or ["- (çıktı yok)"]

    # ─── Model Tabanlı Yol ──────────────────────────────────────────────

    def _run_model(self, task: str, run: AgentRun, *, max_steps: int,
                   max_seconds: float, history: Optional[Sequence[LLMMessage]],
                   project_context: str, plan_first: bool) -> None:
        """Model ile planla-uygula-gözlemle döngüsünü yürütür."""
        assert self._router is not None and self._registry is not None

        # ── Sağlayıcı sağlık denetimi ve otomatik yedekleme ──
        # Ücretsiz katman kotası dolduğunda (`HTTP 429`) kullanıcının elle
        # geçiş yapmasını beklemek yerine çalışan başka bir sağlayıcıya
        # geçilir. Bu yalnızca **çalışma başında** yapılır: çalışma ortasında
        # sağlayıcı değiştirmek araç çağrısı geçmişini geçersiz kılar
        # (Gemini'nin düşünme imzası gibi).
        if self._router is not None and hasattr(self._router, "provider_quota_exhausted"):
            try:
                if self._router.provider_quota_exhausted() and \
                        hasattr(self._router, "failover"):
                    new_provider = self._router.failover("önceki sağlayıcı kotası doldu")
                    if new_provider:
                        self._emit("agent.provider_switched", {
                            "to": new_provider, "task": task[:120],
                        })
            except Exception as exc:
                logger.debug(f"Sağlayıcı yedekleme denetimi başarısız: {exc}")

        messages: List[LLMMessage] = [LLMMessage("system", SYSTEM_PROMPT)]

        # ── Araç seçimi ve bütçe ──
        # 63 aracın tamamını göndermek sağlayıcı sınırlarını aşar
        # (`HTTP 413`), ücretsiz katman kotalarını tüketir (`HTTP 429`) ve
        # modelin doğru aracı seçmesini zorlaştırır. Bu yüzden göreve uygun
        # bir alt küme seçilir; sayı ve token sınırı sağlayıcıya göre ayarlanır.
        provider_name = ""
        if self._router is not None:
            active = self._router.active
            provider_name = getattr(active, "service", "") or getattr(active, "name", "")
        budget = dict(PROVIDER_BUDGETS.get(provider_name, DEFAULT_BUDGET))
        max_tools = int(budget["max_tools"])
        selected_tools = select_tools(task, self._registry, max_tools=max_tools)
        # `use_tools` context kurulmadan **önce** belirlenmeli: aşağıdaki
        # metin/şema tutarlılık bloğu buna bağlıdır.
        use_tools = self._router.supports_tools

        context_parts: List[str] = []
        if self._workspace:
            context_parts.append(f"Çalışma alanı: {self._workspace}")
        if project_context:
            context_parts.append(f"Proje özeti:\n{project_context[:2000]}")

        # ⚠ TUTARLILIK KURALI: Modele **metin olarak** tanıtılan araç listesi,
        # şeması gönderilen araç listesiyle **birebir aynı** olmalıdır.
        #
        # Aksi hâlde model, metinden gördüğü ama şeması gönderilmemiş bir aracı
        # çağırır ve sağlayıcı reddeder:
        #     `HTTP 400 Tool call validation failed: attempted to call tool
        #      'code.workspace_map' which was not in request.tools`  (Groq)
        # Ya da tersi olur: 63 aracın tamamı gönderilirse istek boyutu sınırı
        # aşılır (`HTTP 413`) ve ücretsiz katman kotası tükenir (`HTTP 429`).
        #
        # Ayrıca adlar **API-güvenli** biçimde duyurulur (`code_create_file`):
        # DeepSeek `^[a-zA-Z0-9_-]+$` şart koşar ve nokta içeren adı reddeder.
        advertised = ", ".join(to_wire_name(item) for item in selected_tools)
        if use_tools:
            context_parts.append(
                f"Kullanılabilir araçlar ({len(selected_tools)} adet, adı "
                f"aynen kullan): {advertised}\n"
                "Araç adları `kategori_isim` biçimindedir. Yalnızca bu "
                "listede olan araçları çağır."
            )
        else:
            # Şema desteği yok (JSON istemi modu): açıklamalı liste şarttır.
            context_parts.append(self._registry.describe_for_prompt(max_tools=60))
        messages.append(LLMMessage("system", "\n\n".join(context_parts)))

        for message in (history or [])[-6:]:
            messages.append(message)

        messages.append(LLMMessage("user", task))

        schemas = self._registry.schemas(names=selected_tools) if use_tools else None
        if schemas:
            logger.info(
                f"Ajan {len(schemas)} araç ile başlıyor "
                f"(sağlayıcı: {provider_name or 'bilinmiyor'}, "
                f"max_tokens: {budget['max_tokens']})"
            )

        if plan_first:
            # ⚠ `_ask_plan` artık `(plan, meta)` döndürür. Meta, düşünme
            # döngüsünün **denetim izidir** (kaç aday, kim seçti, güven).
            plan, pmeta = self._ask_plan(task, messages, project_context)
            if pmeta:
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="thinking",
                    text=(f"düşünme: {pmeta.get('candidates', 0)} aday · "
                          f"kaynak={pmeta.get('source', '?')} · "
                          f"güven={pmeta.get('confidence', '?')} · "
                          f"marj={pmeta.get('margin')}"
                          + (f" · {pmeta['note']}" if pmeta.get("note") else "")),
                    ok=not pmeta.get("note"),
                ))
                self._last_plan_meta = pmeta
            if plan:
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="plan", text=plan,
                ))
                messages.append(LLMMessage("assistant", plan))
                messages.append(LLMMessage(
                    "user", "Planı uygula. Hemen ilk araç çağrısıyla başla.",
                ))

        # ⚠ PROJE KARTI: bu klasörde daha önce çalışıldıysa tanıt
        try:
            self._proje_mesaji(messages, run)
        except Exception as exc:
            logger.debug(f"Proje kartı eklenemedi: {exc}")

        # ⚠ 0) KALDIĞIN YER: yarıda kalmış çalışma varsa özeti ver
        try:
            self._kontrol_mesaji(messages, run)
        except Exception as exc:
            logger.debug(f"Devam mesajı atlandı: {exc}")

        # ⚠ 1) PLAYBOOK: benzer iş daha önce çözüldüyse ipucu ver
        self._kullanilan_playbook = ""
        try:
            self._playbook_ipucu(task, run, messages)
        except Exception as exc:
            logger.debug(f"Playbook ipucu atlandı: {exc}")

        # ⚠ 1.5) PUZZLE: iç içe fazlar + ilk walkthrough
        try:
            self._puzzle_kur(task, run, messages)
        except Exception as exc:
            logger.debug(f"Puzzle atlandı: {exc}")

        # ⚠ 2) HİYERARŞİK PLAN: uzun görevlerde hedef → alt hedef → adım
        try:
            self._hiyerarsi_ekle(task, run, messages)
        except Exception as exc:
            logger.debug(f"Hiyerarşi atlandı: {exc}")

        deadline = time.perf_counter() + max(5.0, max_seconds)
        repeats: Dict[str, int] = {}
        empty_responses = 0
        retry_429 = 0
        final_text = ""
        # ⚠ Hedef durumu: olgular/denemeler yapılandırılmış taşınır
        self._hedef = HedefDurumu(hedef=task)
        self._puzzle_bekle = False if self._cog is not None else None
        self._son_hata = ""

        # ⚠ A) Walkthrough cevabı bekleniyorsa döngü **başlamadan** durur
        if self._puzzle_bekle and self._puzzle is not None:
            sorular = self._puzzle.bekleyen_sorular()
            if sorular:
                run.reason = "walkthrough: kullanıcı cevabı bekleniyor"
                run.answer = self._walkthrough_metni(sorular)
                run.ok = False
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="walkthrough", ok=True,
                    text=f"{len(sorular)} parça · "
                         f"{sum(len(s['sorular']) for s in sorular)} soru"))
                return

        # ⚠ 0 (veya negatif) = sınırsız → güvenlik tavanı uygulanır
        etkin_sinir = int(max_steps) if int(max_steps) > 0 else MUTLAK_ADIM
        for step_index in range(max(1, etkin_sinir)):
            if self._cancel.is_set():
                run.ok = False
                run.reason = "iptal edildi"
                return
            if time.perf_counter() > deadline:
                run.reason = f"süre sınırı ({max_seconds:.0f}s)"
                break

            # ⚠ Yetim `tool` / yanıtsız `tool_calls` sağlayıcıda 400 üretir
            self._mesajlari_onar(messages)
            response = self._router.chat(messages, tools=schemas,
                                         max_tokens=budget["max_tokens"])
            # ⚠ TEŞHİS: 400 "role 'tool'" hatasında mesaj listesini diske yaz
            if not response.ok and "role 'tool'" in str(response.error or ""):
                try:
                    import json as _json

                    kayit = []
                    for m in messages[-10:]:
                        kayit.append({
                            "role": getattr(m, "role", "?"),
                            "tool_call_id": getattr(m, "tool_call_id", ""),
                            "tool_calls": [c.get("id") for c in
                                           (getattr(m, "tool_calls", None) or [])],
                            "len": len(getattr(m, "content", "") or ""),
                        })
                    from pathlib import Path as _P3

                    _P3(".neurocore/last_messages.json").write_text(
                        _json.dumps(kayit, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    logger.warning(f"400 teşhis yazıldı · son 10: {kayit}")
                except Exception:
                    pass

            run.prompt_tokens += response.prompt_tokens
            run.completion_tokens += response.completion_tokens

            # ── Hız sınırı (429): kısa süre bekleyip yeniden dene ──
            # Ücretsiz katmanların token penceresi genellikle birkaç saniye
            # içinde yenilenir (`x-ratelimit-reset-tokens: ~20s`). Hemen pes
            # etmek yerine beklemek, kullanıcının aynı isteği tekrar yazmasını
            # önler.
            if not response.ok and "429" in (response.error or "") \
                    and retry_429 < self.MAX_RATE_LIMIT_RETRIES:
                retry_429 += 1
                wait = self.RATE_LIMIT_WAIT_SECONDS * retry_429
                logger.warning(
                    f"Sağlayıcı hız sınırı — {wait:.0f} sn beklenip yeniden "
                    f"denenecek ({retry_429}/{self.MAX_RATE_LIMIT_RETRIES})"
                )
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="plan",
                    text=(f"Sağlayıcı hız sınırı; {wait:.0f} saniye bekleniyor "
                          f"({retry_429}/{self.MAX_RATE_LIMIT_RETRIES})"),
                ))
                # İptal edilebilir bekleme
                waited = 0.0
                while waited < wait and not self._cancel.is_set():
                    time.sleep(0.25)
                    waited += 0.25
                if self._cancel.is_set():
                    run.ok = False
                    run.reason = "iptal edildi"
                    return
                continue

            # ── İstek çok büyük (sağlayıcı sınırı): araç sayısını küçült ve dene ──
            # `HTTP 413` genellikle araç şemalarının toplamından kaynaklanır;
            # görevi iptal etmek yerine daha az araçla devam edilir.
            if not response.ok and "413" in (response.error or "") and schemas \
                    and len(schemas) > 4:
                schemas = schemas[:max(4, len(schemas) // 2)]
                names = [item["function"]["name"] for item in schemas]
                logger.warning(
                    f"İstek çok büyük — araç sayısı {len(schemas)}'e düşürüldü "
                    f"({', '.join(names[:4])}…)"
                )
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="plan",
                    text=(f"Sağlayıcı isteği çok büyük buldu; araç listesi "
                          f"{len(schemas)} araca indirildi ve yeniden denenecek"),
                ))
                continue

            if not response.ok:
                run.ok = False
                run.error = response.error
                run.reason = self._failure_reason(response.error)
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="error",
                    text=f"model hatası: {response.error}", ok=False,
                ))
                # Kullanıcıya **gerçek** gerekçeyi göster: "adım sınırına
                # ulaştım" demek yanıltıcıdır, çünkü asıl sorun model
                # çağrısının başarısız olmasıdır.
                run.answer = self._model_failure_answer(response.error)
                break

            # ── Araç çağrıları (şema modu) ──
            calls = self._extract_tool_calls(response) if use_tools else \
                self._extract_json_calls(response.text)

            if not calls:
                # Araç yok → nihai yanıt olmalı.
                text = (response.text or "").strip()
                if text:
                    # ⚠ BİLİŞSEL KAPI: yanıt araç kanıtıyla destekleniyor mu?
                    # Desteklenmiyorsa **bir kez** düzeltme istenir (bounded).
                    text = self._cevabi_dogrula(task, text, run)
                    final_text = text
                    break

                # Boş yanıt: bazı modeller (özellikle "reasoning" modelleri)
                # `max_tokens` bütçesini düşünme token'larında tüketip hiç
                # metin üretmez. Bunu "nihai yanıt" saymak, kullanıcıya boş
                # bir mesaj göstermek ve adım sınırını boşa harcamak olur —
                # bu yüzden modele açıkça yanıt vermesi söylenir.
                empty_responses += 1
                if empty_responses >= self.MAX_EMPTY_RESPONSES:
                    run.ok = False
                    run.reason = "model art arda boş yanıt verdi"
                    run.error = (
                        f"model {empty_responses} kez boş yanıt verdi "
                        f"(bitiş nedeni: {response.finish_reason or 'bilinmiyor'}"
                        f", token: {response.completion_tokens})"
                    )
                    self._add_step(run, AgentStep(
                        index=len(run.steps), kind="error", ok=False,
                        text=(
                            "Model boş yanıt verdi. Olası nedenler: (1) model "
                            "düşünme token'ları için ayrılan bütçeyi tüketti — "
                            "daha büyük bir model veya daha çok token deneyin; "
                            "(2) model araç çağrısını desteklemiyor; "
                            "(3) istek sağlayıcı tarafından kısmen reddedildi. "
                            f"Bit iş nedeni: {response.finish_reason or 'yok'}"
                        ),
                    ))
                    break

                messages.append(LLMMessage(
                    "user",
                    "Yanıtın boş geldi. Şimdi ya bir araç çağır ya da sorunun "
                    "yanıtını doğrudan yaz. Boş yanıt verme.",
                ))
                continue

            # Mesaj geçmişine asistan turunu ekle.
            # `assistant_message()` sağlayıcının **ham** mesajını kullanır:
            # Gemini'nin `thought_signature`'ı gibi ek alanlar korunmalıdır,
            # aksi hâlde sonraki istek
            # `HTTP 400 Function call is missing a thought_signature` ile
            # reddedilir.
            messages.append(LLMMessage(
                role="assistant",
                content=response.text or "",
                tool_calls=response.tool_calls if use_tools else [],
                raw=response.assistant_message() if use_tools else None,
            ))
            if not use_tools and response.text:
                # JSON istemi modunda metni de sakla (bağlam için)
                messages[-1] = LLMMessage("assistant", response.text)

            step_made_progress = False
            for call in calls:
                if self._cancel.is_set():
                    run.ok = False
                    run.reason = "iptal edildi"
                    return

                tool_name = call.get("name", "")
                arguments = call.get("args", {})
                call_id = call.get("id", "")

                # ── Araç adı doğrulaması ──
                # Model bazen ön eki düşürür (`code.create_file` yerine
                # `create_file`) veya hiç var olmayan bir araç uydurur.
                # Doğrulamadan çağırmak, her turda başarısız bir deneme yapıp
                # sağlayıcı kotasını tüketir. Bunun yerine modele **doğru ad**
                # söylenir; çağrı çalıştırılmaz.
                resolved = self._resolve_tool_name(tool_name)
                if resolved is None:
                    suggestion = self._suggest_tool_name(tool_name)
                    hint = (f" Belki `{suggestion}` demek istedin?"
                            if suggestion else
                            f" Kullanılabilir araçlar: {', '.join(selected_tools[:8])}…")
                    logger.warning(f"Geçersiz araç adı: {tool_name!r} —{hint}")
                    self._add_step(run, AgentStep(
                        index=len(run.steps), kind="error", ok=False,
                        text=f"geçersiz araç adı: {tool_name}",
                    ))
                    messages.append(LLMMessage(
                        "user",
                        f"`{tool_name}` adında bir araç YOK, bu yüzden "
                        f"çalıştırılmadı.{hint} Lütfen tam adı kullan.",
                    ))
                    continue
                tool_name = resolved

                # ── Döngü tespiti ──
                signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
                repeats[signature] = repeats.get(signature, 0) + 1
                if repeats[signature] > self.REPEAT_LIMIT:
                    notice = (
                        f"AYNI ÇAĞRI {repeats[signature]} KEZ TEKRARLANDI: {tool_name}. "
                        "Bu yaklaşım işe yaramıyor — farklı bir yol dene veya "
                        "kullanıcıya durumu bildir."
                    )
                    messages.append(LLMMessage("user", notice))
                    self._add_step(run, AgentStep(
                        index=len(run.steps), kind="error", text=notice, ok=False,
                    ))
                    continue

                result = self._execute(tool_name, arguments, run)
                if result is None:
                    run.ok = False
                    run.reason = "onay reddedildi"
                    run.answer = (
                        f"`{tool_name}` çağrısı için onay verilmedi, bu yüzden "
                        "işlemi durdurdum. Onay verirseniz devam edebilirim."
                    )
                    return

                step_made_progress = True
                result_text = result.render(max_chars=6000)

                if use_tools and call_id:
                    messages.append(LLMMessage(
                        role="tool", content=result_text,
                        tool_call_id=call_id, name=tool_name,
                    ))
                else:
                    messages.append(LLMMessage(
                        "user", f"Araç sonucu ({tool_name}):\n{result_text}",
                    ))

                # ⚠ Hedef durumu: bilinen olgu / deneme
                if self._hedef is not None:
                    if result.ok:
                        self._hedef.olgu_ekle(_kisa_olgu(result))
                    else:
                        self._hedef.deneme_ekle(f"{tool_name} → HATA")

                # ⚠ HATA TAKSONOMİSİ: türüne göre **farklı** kurtarma ver
                # (aynı çağrıyı körlemesine tekrarlamak yerine)
                if not result.ok and self._cog is not None:
                    hata_metni = str(result.error or result_text or "")
                    self._son_hata = hata_metni[:300]
                    tur = self._cog.siniflandir(hata_metni)
                    if tur != HataTuru.YOK:
                        talimat = kurtarma_talimati(tur)
                        self._counter["cog_recovery"] += 1
                        self._add_step(run, AgentStep(
                            index=len(run.steps), kind="recovery", ok=False,
                            text=f"{tur.value}: {talimat[:160]}",
                        ))
                        messages.append(LLMMessage(
                            "user", f"KURTARMA ({tur.value}): {talimat}"))

            # ⚠ BİLİŞSEL İZLEME: yalnız araç çağrılan turlarda çalışır.
            if calls and self._hedef is not None:
                self._hedef.deneme_ekle(
                    f"{len(calls)} araç → {'ilerleme' if step_made_progress else 'ilerleme yok'}")

            # ⚠ Periyodik ÖZ-YANSIMA (bounded: en fazla max_reflect kez)
            self._yansima_kapisi(task, run, messages)

            # ⚠ KONTROL NOKTASI: her araç adımında güncelle → kesinti olursa
            # "nerede kaldım" bilgisi kaybolmasın (12 adımı beklemez).
            try:
                self._kontrol_guncelle(run, "devam ediyor")
            except Exception as exc:
                logger.debug(f"Kontrol güncellenemedi: {exc}")

            # ⚠ FAZ RAPORU: her FAZ_ADIM adımda durum ver, sonra devam et
            try:
                self._faz_raporu(task, run, messages)
            except Exception as exc:
                logger.debug(f"Faz raporu atlandı: {exc}")

            # ⚠ BAĞLAM SIKIŞTIRMA: uzun çalışmada eski mesajları özete indir
            try:
                self._baglami_sikistir(messages)
            except Exception as exc:
                logger.debug(f"Sıkıştırma atlandı: {exc}")

            if not step_made_progress:
                messages.append(LLMMessage(
                    "user", "Hiçbir araç çalışmadı. Farklı bir yaklaşım dene veya "
                            "elindeki bilgiyle yanıt ver.",
                ))

        else:
            run.reason = f"güvenlik tavanı ({etkin_sinir} adım)"

        if not final_text:
            # Çalışma **hatayla** bittiyse "adım sınırına ulaştım" demek
            # yanıltıcıdır: kullanıcı adım sayısını artırmayı dener ama asıl
            # sorun sağlayıcı hatasıdır. Bu yüzden gerçek gerekçe gösterilir
            # ve o ana kadar yapılan işler özet olarak eklenir.
            if run.cancelled or "iptal" in (run.reason or ""):
                final_text = (
                    "**Çalışma durduruldu.** Çalışan komutlar varsa "
                    "sonlandırıldı.\n\n"
                    "Aşağıda durdurulmadan önce tamamlanan adımlar listelenmiştir. "
                    "Devam etmek isterseniz görevi yeniden gönderin; kaldığı "
                    "yerden değil baştan başlar ama dosya değişiklikleri diskte "
                    "korunur."
                )
                progress = self._progress_lines(run)
                if progress:
                    final_text += "\n\n**Tamamlanan adımlar:**\n" + progress
            elif not run.ok and run.error:
                final_text = self._model_failure_answer(run.error)
                progress = self._progress_lines(run)
                if progress:
                    final_text += "\n\n**Bu çalışmada tamamlanan adımlar:**\n" + progress
            else:
                final_text = self._closing_summary(run)

        run.answer = final_text.strip() or self._closing_summary(run)

        # ⚠ Kontrol noktası: tamamlandı olarak kapat + arşivle
        try:
            # ⚠ İptal/hata "tamamlandı" DEĞİLDİR: devam edilebilir kalmalı
            _iptal = bool(getattr(run, "cancelled", False))
            self._kontrol_bitir(
                run,
                tamamlandi=bool(run.ok and not _iptal),
                sebep=("kullanıcı durdurdu" if _iptal
                       else (run.reason or "tamamlandı")))
        except Exception as exc:
            logger.debug(f"Kontrol noktası kapatılamadı: {exc}")

        # ⚠ C) Planı kasaya yaz (benzer görevlerde geri çağrılır)
        try:
            self._puzzle_kasaya(run)
        except Exception as exc:
            logger.debug(f"Puzzle kasa yazımı atlandı: {exc}")

        # ⚠ Başarılı akışı **prosedür** olarak sakla (sonraki görevlerde ipucu)
        try:
            self._playbook_kaydet(task, run)
        except Exception as exc:
            logger.debug(f"Playbook kaydı atlandı: {exc}")
        if not run.reason:
            run.reason = "tamamlandı"
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="answer", text=run.answer[:4000],
        ))

    def _progress_lines(self, run: AgentRun) -> str:
        """Tamamlanan araç çağrılarını madde listesi olarak döndürür."""
        lines: List[str] = []
        for step in run.steps:
            if step.kind != "tool":
                continue
            mark = "✓" if step.ok else "✗"
            lines.append(f"- {mark} `{step.tool}` — {_brief_args(step.args, 60)}")
        if run.files_changed:
            lines.append("")
            lines.append("Değişen dosyalar: "
                         + ", ".join(f"`{name}`" for name in run.files_changed[:10]))
        return "\n".join(lines)

    def _failure_reason(self, error: str) -> str:
        """Sağlayıcı hatasını kısa, anlaşılır bir nedene çevirir."""
        text = error or ""
        if "413" in text or "too large" in text.lower():
            return "istek sağlayıcı sınırını aştı (413)"
        if "429" in text or "rate limit" in text.lower():
            return "sağlayıcı hız sınırı (429) — biraz bekleyin"
        if "401" in text or "403" in text:
            return "kimlik doğrulama hatası — API anahtarını kontrol edin"
        if "404" in text or "does not exist" in text.lower():
            return "model bulunamadı — model adını kontrol edin"
        if "zaman aşımı" in text or "timeout" in text.lower():
            return "model çağrısı zaman aşımına uğradı"
        if "bağlantı" in text or "ağ" in text:
            return "ağ/sağlayıcı bağlantı hatası"
        return "model çağrısı başarısız"

    def _model_failure_answer(self, error: str) -> str:
        """
        Model çağrısı başarısız olduğunda kullanıcıya gösterilecek mesaj.

        "Adım sınırına ulaştım" demek yanıltıcıdır: kullanıcı adım sayısını
        artırmayı dener ama asıl sorun sağlayıcı hatasıdır. Bu yüzden gerçek
        gerekçe ve **somut** çözüm önerisi yazılır.
        """
        text = error or ""
        if "413" in text or "too large" in text.lower():
            return (
                "**Model çağrısı başarısız: istek sağlayıcı sınırını aştı (HTTP 413).**\n\n"
                "Araç tanımlarının toplamı sağlayıcının istek boyutu sınırını "
                "aşıyor. Çözümler:\n\n"
                "- Daha büyük bağlam penceresi olan bir model seçin\n"
                "- Daha yüksek kotalı bir sağlayıcıya geçin (DeepSeek, Gemini, "
                "OpenRouter)\n"
                "- İsteği daha dar kapsamlı yazın (ör. yalnızca kod araçları "
                "gerektiren bir görev)"
            )
        if "429" in text or "rate limit" in text.lower() \
                or "quota" in text.lower() or "exceeded" in text.lower():
            return (
                "**Model çağrısı başarısız: sağlayıcı kotası doldu (HTTP 429).**\n\n"
                "Ücretsiz katmanların **günlük** token kotası tükendi. Bu, kodda "
                "bir hata değil; sağlayıcının sınırıdır ve genellikle 24 saat "
                "içinde sıfırlanır. Çözümler:\n\n"
                "- **Ayarlar → Dil Modeli** bölümünden başka bir sağlayıcıya geçin "
                "(birden fazla anahtarınız varsa en hızlı çözüm budur)\n"
                "- Kotası sıfırlanana kadar bekleyin\n"
                "- Ücretli bir plana geçin\n\n"
                "Aşağıda bu çalışmada tamamlanan adımlar listelenmiştir."
            )
        if "401" in text or "403" in text:
            return (
                "**Model çağrısı başarısız: kimlik doğrulama hatası.**\n\n"
                "Ayarlar → Dil Modeli bölümünden API anahtarını yeniden girin "
                "ve **Bağlantıyı test et** düğmesine basın."
            )
        if "404" in text or "does not exist" in text.lower():
            return (
                "**Model çağrısı başarısız: model bulunamadı (HTTP 404).**\n\n"
                "Model adı geçersiz veya erişiminiz yok. Ayarlar → Dil Modeli "
                "bölümünde önerilen modellerden birini seçin."
            )
        return (
            f"**Model çağrısı başarısız.**\n\n```\n{text[:600]}\n```\n\n"
            "Ayarlar → Dil Modeli bölümünden bağlantıyı test edin."
        )

    def _ask_plan(self, task: str, messages: Sequence[LLMMessage],
                  project_context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Modelden plan ister.

        ⚠ Düşünme döngüsü bağlıysa davranış **değişir**:
            1. Bulut modeli **N farklı plan** önerir (tek çağrı)
            2. Yerel model bunları kendi olasılığıyla **sıralar**
            3. Belirsizlik eşiğin altındaysa bulutun ilk planı kullanılır

        Gerekçe (ölçüm): yerel model akıcı metin **üretemiyor** ama
        **seçebiliyor** (Blender kodunda %70, yaklaşımda %50; rastgele %25).
        Bu yüzden plan yazdırmak yerine adaylar arasından seçtiriyoruz.

        Returns:
            `(plan_metni, meta)` — düşünme yoksa meta boş sözlüktür.
        """
        assert self._router is not None
        self._last_plan_meta = {}
        prompt = PLAN_PROMPT.format(
            task=task,
            workspace=(project_context or self._workspace or "bilinmiyor")[:1500],
        )

        # ── Düşünme yok: eski davranış (tek çağrı) ──
        if self._thinker is None:
            response = self._router.chat(
                list(messages) + [LLMMessage("user", prompt)], max_tokens=800,
            )
            return (response.text.strip() if response.ok else ""), {}

        # ── 1) BULUT: N farklı plan (tek çağrı — maliyet 1 çağrı) ──
        adet = max(2, int(self.PLAN_CANDIDATES))
        istek = (
            f"{prompt}\n\nTam olarak {adet} FARKLI plan öner. "
            "Her planı `### PLAN 1`, `### PLAN 2` biçiminde başlıkla ayır. "
            "Planlar gerçekten farklı olsun (ör. en kısa, en güvenli, "
            "en ölçeklenebilir)."
        )
        response = self._router.chat(
            list(messages) + [LLMMessage("user", istek)], max_tokens=1400,
        )
        metin = response.text.strip() if response.ok else ""
        if not metin:
            return "", {}
        adaylar = self._parse_plans(metin)
        if len(adaylar) < 2:
            # Ayrıştırılamadı — olduğu gibi kullan (sessiz kayıp yok)
            logger.info("Plan adayları ayrıştırılamadı — tek plan kullanılıyor")
            return metin, {"candidates": 1, "source": "cloud",
                           "note": "adaylar ayrıştırılamadı"}

        # ── 2) YEREL: seç ──
        try:
            sonuc = self._thinker.think(
                task, lambda k: adaylar[:k],
                context=(project_context or "")[:800],
            )
        except Exception as exc:
            logger.warning(
                f"Düşünme döngüsü çalışmadı: {type(exc).__name__}: {exc}")
            return adaylar[0], {"candidates": len(adaylar), "source": "cloud",
                                "note": f"düşünme hatası: {type(exc).__name__}"}

        meta: Dict[str, Any] = {
            "candidates": len(adaylar),
            "source": sonuc.source,
            "confidence": sonuc.confidence,
            "margin": sonuc.margin,
        }
        if sonuc.ok and sonuc.chosen:
            return sonuc.chosen, meta
        return adaylar[0], {**meta, "note": sonuc.error or "seçim yapılamadı"}

    #: ⚠ Plan adayı başlığı: `### PLAN 1`, `PLAN 2:`, `**PLAN 3**` …
    _PLAN_SPLIT = re.compile(
        r"(?im)^\s*(?:#{1,4}\s*)?\**\s*PLAN\s*[0-9]+\s*\**\s*[:.)\-]?"
    )

    def _parse_plans(self, metin: str) -> List[str]:
        """Plan metnini adaylara böler (en az 2 parça yoksa tek aday)."""
        parcalar = [p.strip() for p in self._PLAN_SPLIT.split(metin)
                    if p and p.strip()]
        if len(parcalar) >= 2:
            return parcalar
        return [metin.strip()]

    def _extract_tool_calls(self, response: LLMResponse) -> List[Dict[str, Any]]:
        """Şema tabanlı araç çağrılarını ayrıştırır."""
        calls: List[Dict[str, Any]] = []
        for raw in response.tool_calls:
            function = raw.get("function") or {}
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append({
                "id": str(raw.get("id", "") or f"call_{len(calls)}"),
                "name": name, "args": arguments,
            })
        return calls

    def _extract_json_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        JSON istemi modunda metinden araç çağrılarını çıkarır.

        Hem ```json bloğunu hem de çıplak JSON nesnesini tanır. Model
        `{"tool": "ad", "args": {...}}` veya `{"name": ..., "arguments": ...}`
        yazabilir.
        """
        if not text:
            return []
        candidates: List[str] = []

        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            candidates.append(match.group(1))

        # Çıplak JSON: dengeli parantez taraması
        depth = 0
        start = -1
        for index, character in enumerate(text):
            if character == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start:index + 1])
                    start = -1

        calls: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            # {"tool": ..., "args": {...}} veya içinde "tool"/"tools" olan sarmalayıcı
            entries: List[Dict[str, Any]] = []
            if "tool" in data or "tools" in data or "calls" in data:
                if isinstance(data.get("tools"), list):
                    entries = [item for item in data["tools"] if isinstance(item, dict)]
                elif isinstance(data.get("calls"), list):
                    entries = [item for item in data["calls"] if isinstance(item, dict)]
                else:
                    entries = [data]

            for entry in entries:
                name = str(entry.get("tool") or entry.get("name") or "").strip()
                if not name:
                    continue
                arguments = entry.get("args") or entry.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append({"id": "", "name": name, "args": arguments})

        return calls

    def _closing_summary(self, run: AgentRun) -> str:
        """Çalışma kesildiğinde (süre/güvenlik tavanı/hata) eldeki bilgiyle özet."""
        if not run.steps:
            return ("Görevi tamamlayamadım: hiçbir adım atılamadı. "
                    "İsteği daha somut biçimde yazabilir misiniz?")
        # ⚠ YANILTICI OLMAYAN DİL: adım sınırı varsayılan olarak yok; buraya
        # düşüldüyse sebep süre, güvenlik tavanı veya hatadır.
        neden = (getattr(run, "reason", "") or "").strip()
        baslik = ("Çalışmayı burada noktaladım"
                  + (f" ({neden})." if neden else "."))
        lines = [baslik, "", "Şu ana kadar yaptıklarım:", ""]
        for step in run.steps:
            if step.kind == "tool":
                mark = "✓" if step.ok else "✗"
                lines.append(f"- {mark} `{step.tool}` — {_brief_args(step.args)}")
        if run.files_changed:
            lines.extend(["", f"Değişen dosyalar: "
                              f"{', '.join('`' + f + '`' for f in run.files_changed)}"])
        lines.extend(["", "Devam etmemi isterseniz 'devam et' yazın."])
        return "\n".join(lines)

    # ─── Araç Yürütme (onay akışıyla) ───────────────────────────────────

    def _resolve_tool_name(self, name: str) -> Optional[str]:
        """
        Araç adını kayıtlı bir araçla eşleştirir.

        İki farklı sapma düzeltilir:
            1. **Normalizasyon** — şemalar API-güvenli adla gönderilir
               (`code.create_file` → `code_create_file`); model bu adı
               döndürür, kayıtlı adla eşleştirilmelidir.
            2. **Ön ek düşmesi** — model `code.create_file` yerine
               `create_file` diyebilir.

        Belirsizlik varsa (birden fazla aday) None döner ve model uyarılır:
        yanlış aracı çalıştırmaktansa sormak yeğdir.

        Returns:
            Kayıtlı tam araç adı veya None
        """
        if not name:
            return None
        if self._registry.has(name):
            return name
        # Kayıt defterinin kendi çözücüsü (normalizasyon + kuyruk eşleşmesi)
        resolved = self._registry.resolve(name)
        if resolved:
            logger.info(f"Araç adı düzeltildi: {name!r} → {resolved!r}")
            return resolved
        return None

    def _suggest_tool_name(self, name: str) -> str:
        """
        Geçersiz araç adı için **en yakın** kayıtlı aracı önerir.

        `difflib` benzerliği kullanılır; eşik altındaki tahminler
        döndürülmez — yanlış öneri, öneri vermemekten kötüdür (modeli
        yanlış araca yönlendirir).
        """
        if not name:
            return ""
        target = name.lower().replace("-", "_")
        names = list(self._registry.names())
        # 1) Tam kuyruk eşleşmesi en güçlü sinyaldir (`create_file` →
        #    `code.create_file`)
        exact = [item for item in names
                 if item.lower().split(".")[-1] == target]
        if len(exact) == 1:
            return exact[0]
        # 2) Bulanık eşleşme
        close = difflib.get_close_matches(target, names, n=1, cutoff=0.72)
        return close[0] if close else ""

    def _uzaktan_dene(self, tool_name: str,
                      arguments: Dict[str, Any]) -> Optional[Any]:
        """
        Aracı **komşu düğümde** dener (yalnız okuma araçları).

        ⚠ Neden yalnız okuma? Uzak düğüm bizim adımıza dosya yazamaz /
        kabuk çalıştıramaz; izin listesi `IZINLI_ARACLAR` ile ortaktır.

        Returns:
            `(ToolResult, kaynak_adresi)` veya `None` (komşu yok/yanıtsız)
        """
        if self._peer is None:
            return None
        try:
            from neurocore.net.peer_pool import IZINLI_ARACLAR
        except Exception:                          # pragma: no cover
            return None
        if tool_name not in IZINLI_ARACLAR:
            return None
        self._counter["peer_calls"] += 1
        try:
            yanit = self._peer.komsu_dene(
                {"tool": tool_name, "args": arguments}, deneme=1)
        except Exception as exc:
            logger.debug(f"Komşu denemesi çöktü: {type(exc).__name__}: {exc}")
            self._counter["peer_fallback"] += 1
            return None
        if not yanit or not yanit.get("ok"):
            # ⚠ Yanıt yok → çağıran **yerelde** çalıştıracak (iş kaybolmaz)
            self._counter["peer_fallback"] += 1
            return None
        veri = yanit.get("sonuc")
        if not isinstance(veri, dict):
            self._counter["peer_fallback"] += 1
            return None
        self._counter["peer_ok"] += 1
        return (
            ToolResult(
                ok=bool(veri.get("ok", True)),
                tool=str(veri.get("tool") or tool_name),
                data=veri.get("data"),
                error=str(veri.get("error") or ""),
                duration_ms=float(veri.get("duration_ms") or 0.0),
                safety=str(veri.get("safety") or "read"),
                truncated=bool(veri.get("truncated")),
            ),
            str(yanit.get("kaynak") or "komsu"),
        )

    # ─── Bilişsel Katman Kancaları ──────────────────────────────────────

    # ─── Playbook / Hiyerarşi / Sıkıştırma ──────────────────────────────

    def _playbook_ipucu(self, task: str, run: AgentRun,
                        messages: List[LLMMessage]) -> None:
        """Benzer başarılı akış varsa plana ipucu ekler."""
        store = self._playbooks
        if store is None:
            return
        bulunan = store.bul(task, adet=1)
        if not bulunan:
            return
        pb, skor = bulunan[0]
        self._counter["playbook_hit"] += 1
        self._kullanilan_playbook = pb.id
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="playbook", ok=True,
            text=f"benzerlik {skor:.2f} · {pb.goal[:60]}",
        ))
        messages.append(LLMMessage("user", "İPUCU (önceki çözüm):\n" + pb.ipucu()))

    def _hiyerarsi_ekle(self, task: str, run: AgentRun,
                        messages: List[LLMMessage]) -> None:
        """
        ⚠ Uzun görevlerde hedef → alt hedef → adım planı üretir (tek çağrı).

        Kısa görevlerde **çalışmaz**: gereksiz maliyet olur.
        """
        if not self._hier_aktif or self._router is None:
            return
        if len(str(task or "").strip()) < HIER_MIN_UZUNLUK:
            return
        from neurocore.agent.hierarchy import planla as _hiyerarsi_planla

        araclar: List[str] = []
        if self._registry is not None:
            try:
                araclar = list(self._registry.names())
            except Exception:
                araclar = []
        sonuc = _hiyerarsi_planla(self._router, task, araclar=araclar)
        if not sonuc.ok:
            self._counter["hier_fail"] += 1
            return
        self._counter["hier_plan"] += 1
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="plan", ok=True,
            text=f"hiyerarşi: {len(sonuc.alt_hedefler)} alt hedef · "
                 f"{len(sonuc.araclar())} araç",
        ))
        messages.append(LLMMessage(
            "user",
            "HİYERARŞİK PLAN (bu sırayı izle; gerekirse gerekçesiyle sap):\n"
            + sonuc.ozet()))

    # ─── Proje kapsamlı durum (.neurocore) ────────────────────────────

    def set_workspace(self, path: Any) -> None:
        """
        Etkin çalışma alanını değiştirir ve durum yollarını **proje içine**
        taşır: `<proje>/.neurocore/`.

        ⚠ ÖLÇÜLEN EKSİK: `apply_workspace` yalnız kod/kabuk katmanlarını
        güncelliyordu; döngü başlangıç dizininde kalıyordu → plan/kontrol
        dosyaları proje klasöründe **oluşmuyordu**.
        """
        try:
            kok = Path(str(path)).expanduser().resolve()
        except (OSError, TypeError):
            return
        self._workspace = str(kok)
        durum = kok / ".neurocore"
        self._kontrol_yol = durum / "kontrol.json"
        self._kontrol_arsiv = durum / "kontrol_gecmis.jsonl"
        self._puzzle_yol = durum / "puzzle.json"
        self._proje_yol = durum / "proje.json"

    def proje_karti(self, run: Optional[AgentRun] = None) -> Dict[str, Any]:
        """
        Projenin **künyesi**: ne olduğu, nasıl çalıştırıldığı, nerede kalındığı.

        Yazıldığı yer: `<proje>/.neurocore/proje.json`
        """
        import time as _t

        kok = Path(self._workspace or ".").resolve()
        komutlar: List[str] = []
        try:
            paket = kok / "package.json"
            if paket.exists():
                import json as _j

                veri = _j.loads(paket.read_text(encoding="utf-8"))
                betikler = (veri.get("scripts") or {})
                for ad in ("start", "dev", "test", "build"):
                    if ad in betikler:
                        komutlar.append(f"npm run {ad}" if ad != "start"
                                        else "npm start")
            for ad, komut in (("pyproject.toml", "python -m pytest"),
                              ("requirements.txt", "python main.py")):
                if (kok / ad).exists():
                    komutlar.append(komut)
        except Exception:
            pass
        kart = {
            "proje": kok.name,
            "kok": str(kok),
            "gorev": (run.task[:400] if run is not None
                      else getattr(self._kontrol, "gorev", "")[:400]),
            "guncellendi": _t.strftime("%Y-%m-%dT%H:%M:%S"),
            "komutlar": komutlar,
            "durum": "devam ediyor",
        }
        if self._kontrol is not None:
            kart["adim"] = self._kontrol.arac_sayisi()
            kart["son_adim"] = self._kontrol.son_adim()
            kart["dosyalar"] = self._kontrol.dosyalar[:20]
            kart["durum"] = ("tamamlandı" if self._kontrol.tamamlandi
                             else "devam ediyor")
        if self._puzzle is not None:
            kart["puzzle"] = {
                "parca": self._puzzle.sayi(),
                "tamamlanan": self._puzzle.tamamlanan(),
                "aktif": self._puzzle.aktif_id,
                "genisletme": self._puzzle.genisletme,
            }
        return kart

    def _proje_kaydet(self, run: Optional[AgentRun] = None) -> None:
        """Proje kartını + puzzle ağacını proje klasörüne yazar."""
        if not getattr(self, "_proje_yol", None):
            return
        try:
            import json as _j

            self._proje_yol.parent.mkdir(parents=True, exist_ok=True)
            self._proje_yol.write_text(
                _j.dumps(self.proje_karti(run), ensure_ascii=False, indent=1),
                encoding="utf-8")
            if self._puzzle is not None and getattr(self, "_puzzle_yol", None):
                self._puzzle_yol.write_text(
                    _j.dumps(self._puzzle.to_dict(), ensure_ascii=False,
                             indent=1), encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Proje kartı yazılamadı: {exc}")

    def _proje_mesaji(self, messages: List[LLMMessage], run: AgentRun) -> None:
        """
        Proje kartı varsa modele "bu proje nedir, nerede kaldın" bilgisini ver.

        ⚠ Yeni bir sohbet açılsa bile ajan projeyi tanır (dosyadan okur).
        """
        yol = getattr(self, "_proje_yol", None)
        if yol is None or not Path(yol).exists():
            return
        try:
            import json as _j

            kart = _j.loads(Path(yol).read_text(encoding="utf-8"))
        except Exception:
            return
        satirlar = ["PROJE KARTI (bu klasörde daha önce çalışıldı):",
                    f"  proje: {kart.get('proje')} · durum: {kart.get('durum')}"]
        if kart.get("gorev"):
            satirlar.append(f"  son görev: {str(kart['gorev'])[:160]}")
        if kart.get("adim"):
            satirlar.append(f"  ilerleme: {kart['adim']} adım · "
                            f"son: {str(kart.get('son_adim') or '')[:90]}")
        if kart.get("komutlar"):
            satirlar.append("  çalıştırma: " + " · ".join(kart["komutlar"]))
        if kart.get("dosyalar"):
            satirlar.append("  değişen dosyalar: "
                            + ", ".join(kart["dosyalar"][:8]))
        if kart.get("puzzle"):
            pz = kart["puzzle"]
            satirlar.append(f"  plan: {pz.get('parca')} parça · "
                            f"{pz.get('tamamlanan')} tamam · aktif {pz.get('aktif')}")
        messages.append(LLMMessage("user", "\n".join(satirlar)))

    # ─── Kontrol noktası (kaldığı yerden devam) ────────────────────────

    def kontrol_durumu(self) -> Dict[str, Any]:
        """Arayüz/API için: son çalışmanın nerede kaldığı."""
        if self._kontrol is None:
            from neurocore.agent.checkpoint import Kontrol

            if self._kontrol_yol is not None:
                self._kontrol = Kontrol.yukle(self._kontrol_yol)
        if self._kontrol is None:
            return {}
        return {
            **self._kontrol.to_dict(),
            "ozet": self._kontrol.ozet(),
            "son_adim": self._kontrol.son_adim(),
        }

    def _kontrol_hazirla(self, task: str, run: AgentRun) -> None:
        """
        ⚠ Yeni çalışma mı, devam mı?

        Benzer ve **tamamlanmamış** bir kontrol noktası varsa onu yükler;
        aksi hâlde temiz bir kayıt açar.
        """
        from neurocore.agent.checkpoint import Kontrol, devam_edilebilir

        self._kontrol = None
        if self._kontrol_yol is None:
            return
        onceki = Kontrol.yukle(self._kontrol_yol)
        if devam_edilebilir(onceki, task):
            self._kontrol = onceki
            self._kontrol.gorev = self._kontrol.gorev or task
            self._counter["kontrol_devam"] = self._counter.get(
                "kontrol_devam", 0) + 1
            logger.info(f"Kontrol noktasından devam: "
                        f"{onceki.arac_sayisi()} adım geri yüklendi")
            # Olguları hedef durumuna taşı (tekrar iş yapılmasın)
            for olgu in onceki.olgular:
                if self._hedef is not None:
                    self._hedef.olgu_ekle(olgu)
            return
        self._kontrol = Kontrol(
            gorev=task, workspace=str(self._workspace or ""),
            basladi=time.time(), guncellendi=time.time(),
        )

    def _kontrol_guncelle(self, run: AgentRun, sebep: str = "") -> None:
        """Çalışma durumunu diske yazar (her faz raporunda + bitişte)."""
        if self._kontrol is None or self._kontrol_yol is None:
            return
        try:
            from neurocore.agent.checkpoint import Kontrol  # noqa: F401

            adimlar = []
            for step in run.steps:
                if step.kind != "tool":
                    continue
                adimlar.append({
                    "tool": step.tool or "?",
                    "ozet": _kisa_args(step.args) if step.args else "",
                    "ok": bool(step.ok),
                })
            self._kontrol.adimlar = adimlar[-120:]
            self._kontrol.dosyalar = list(run.files_changed)[:40]
            if sebep:
                self._kontrol.sebep = sebep
            self._kontrol.guncellendi = time.time()
            self._kontrol.olgular = (list(self._hedef.facts)[:20]
                                     if self._hedef is not None else [])
            self._kontrol.acik_sorular = (
                [s for p in (self._puzzle.bekleyen_sorular()
                             if self._puzzle is not None else [])
                 for s in p.get("sorular", [])][:10])
            if self._puzzle is not None:
                self._kontrol.puzzle = self._puzzle.to_dict()
            self._kontrol.kaydet(self._kontrol_yol)
            self._proje_kaydet(run)
        except Exception as exc:
            logger.debug(f"Kontrol noktası güncellenemedi: {exc}")

    def _kontrol_bitir(self, run: AgentRun, *, tamamlandi: bool,
                       sebep: str) -> None:
        """Çalışma bittiğinde son durumu yazar ve arşive ekler."""
        if self._kontrol is None:
            return
        self._kontrol.tamamlandi = bool(tamamlandi)
        self._kontrol.sebep = sebep
        self._kontrol_guncelle(run, sebep)
        if self._kontrol_arsiv is not None:
            try:
                from neurocore.agent.checkpoint import Kontrol as _K

                _K.arsivle(self._kontrol, self._kontrol_arsiv)
            except Exception:
                pass
        # ⚠ Proje künyesini son durumla güncelle (proje.json)
        try:
            self._proje_kaydet(run)
        except Exception:
            pass

    def _kontrol_mesaji(self, messages: List[LLMMessage],
                        run: AgentRun) -> None:
        """Devam ederken modele "kaldığın yer" özetini verir."""
        if self._kontrol is None or not self._kontrol.adimlar:
            return
        if self._kontrol.tamamlandi:
            return
        metin = self._kontrol.ozet()
        if not metin:
            return
        messages.append(LLMMessage("user", metin))
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="resume", ok=True,
            text=f"devam: {self._kontrol.arac_sayisi()} adım geri yüklendi · "
                 f"son: {self._kontrol.son_adim()[:80]}"))
        self._counter["kontrol_devam_mesaji"] = self._counter.get(
            "kontrol_devam_mesaji", 0) + 1

    # ─── Puzzle (iç içe fazlar / walkthrough) ──────────────────────────

    def _puzzle_kur(self, task: str, run: AgentRun,
                    messages: List[LLMMessage]) -> None:
        """
        Görevi iç içe fazlara böler ve ilk **walkthrough**'u sunar.

        ⚠ Kullanıcı isteği: plan fazlara ayrılsın, her faz başında durum ve
        netleştirme soruları olsun, kullanıcı bir kısmı değiştirmek isterse
        not düşülsün. Yapı keyfî derinlikte olabilir.
        """
        if not getattr(self, "_puzzle_aktif", True) or self._router is None:
            return
        if len(str(task or "").strip()) < 60:
            return                      # kısa görevde ağaç kurmak gereksiz
        # ⚠ SÜRDÜRME: bitmemiş ve benzer bir puzzle varsa yeniden kurma.
        # Walkthrough onayından sonra "devam" dendiğinde aynı ağaç sürer.
        if self._puzzle is not None and not self._puzzle.tamamlandi():
            try:
                from neurocore.agent.playbooks import benzerlik

                skor = benzerlik(task, self._puzzle.gorev)
            except Exception:
                skor = 0.0
            if skor >= 0.45:
                self._puzzle.bekliyor = False
                self._counter["puzzle_surduruldu"] = self._counter.get(
                    "puzzle_surduruldu", 0) + 1
                self._puzzle_mesaji(messages, run, ilk=True)
                return

        try:
            from neurocore.agent.puzzle import PuzzlePlanner

            planner = PuzzlePlanner(self._router)
            self._puzzle = planner.planla(task)
        except Exception as exc:
            logger.debug(f"Puzzle kurulamadı: {exc}")
            self._puzzle = None
            return
        if self._puzzle is None or self._puzzle.sayi() <= 1:
            return
        self._counter["puzzle_kuruldu"] = self._counter.get(
            "puzzle_kuruldu", 0) + 1
        self._puzzle_mesaji(messages, run, ilk=True)

    def _puzzle_mesaji(self, messages: List[LLMMessage], run: AgentRun,
                       *, ilk: bool = False) -> None:
        """Aktif parçayı + walkthrough sorularını mesaja ekler."""
        if self._puzzle is None:
            return
        yaprak = self._puzzle.aktif_yaprak()
        baslik = "PUZZLE PLANI" if ilk else "SIRADAKİ PARÇA"
        satirlar = [f"{baslik}:", self._puzzle.ozet(sinir=900)]
        if yaprak is not None:
            satirlar.append(f"\nŞU AN: [{yaprak.id}] {yaprak.baslik}")
            if yaprak.olcut:
                satirlar.append(f"Tamamlanma ölçütü: {yaprak.olcut}")
            if yaprak.sorular and not yaprak.cevap and getattr(
                    self, "_puzzle_dur", True):
                # ⚠ A) Kullanıcı cevabı yoksa **dur**: çalışma burada kesilir,
                # sorular arayüze döner; cevap gelince yeni tur devam eder.
                self._puzzle_bekle = True
            if yaprak.sorular and not yaprak.cevap:
                satirlar.append("Netleştirme soruları (bağlamdan cevapla, "
                                "bilinmeyeni varsayım olarak işaretle):")
                satirlar.extend(f"  - {soru}" for soru in yaprak.sorular)
                self._add_step(run, AgentStep(
                    index=len(run.steps), kind="walkthrough", ok=True,
                    text=f"[{yaprak.id}] {len(yaprak.sorular)} soru: "
                         + " | ".join(yaprak.sorular)[:200]))
            if yaprak.not_:
                satirlar.append(f"Kullanıcı revizyon notu: {yaprak.not_}")
        messages.append(LLMMessage("user", "\n".join(satirlar)))

    def _puzzle_ilerle(self, run: AgentRun, messages: List[LLMMessage]) -> None:
        """Parçayı tamamlar ve sıradakini sunar (faz raporuyla birlikte)."""
        if self._puzzle is None or self._puzzle.tamamlandi():
            return
        yeni = self._puzzle.ilerle()
        if yeni is None:
            self._add_step(run, AgentStep(
                index=len(run.steps), kind="plan", ok=True,
                text="puzzle tamamlandı"))
            return
        self._counter["puzzle_ilerleme"] = self._counter.get(
            "puzzle_ilerleme", 0) + 1
        self._puzzle_mesaji(messages, run)

    @staticmethod
    def _walkthrough_metni(sorular: List[Dict[str, Any]]) -> str:
        """Kullanıcıya gösterilecek walkthrough metni."""
        satirlar = ["Plan hazır — devam etmeden önce birkaç şeyi netleştirelim:", ""]
        for parca in sorular:
            satirlar.append(f"**{parca['baslik']}** ([{parca['id']}])")
            satirlar.extend(f"- {soru}" for soru in parca["sorular"])
            satirlar.append("")
        satirlar.append("Cevaplarını yazınca planı güncelleyip devam edeceğim. "
                        "İstersen `Puzzle` sekmesinden bir parçayı "
                        "değiştirmek üzere işaretleyebilirsin.")
        return "\n".join(satirlar)

    def _puzzle_genislet(self, run: AgentRun,
                         messages: List[LLMMessage]) -> None:
        """
        ⚠ B) Aktif yaprak çok kaba ise çalışma sırasında alt parçalara bölünür.

        Sınır: en fazla 2 genişletme, derinlik ≤ 3 (Puzzle.MAX_DERINLIK ile
        birlikte). Hata sessizdir; akış bozulmaz.
        """
        if self._puzzle is None or self._router is None:
            return
        if self._puzzle.genisletme >= 2:
            return
        yaprak = self._puzzle.aktif_yaprak()
        if yaprak is None or yaprak.cocuklar or yaprak.derinlik >= 3:
            return
        if len(yaprak.baslik) < 12:
            return
        try:
            from neurocore.agent.puzzle import PuzzlePlanner

            if self._puzzle_planlayici is None:
                self._puzzle_planlayici = PuzzlePlanner(self._router)
            yeni = self._puzzle_planlayici.genislet(yaprak, max_alt=3)
        except Exception as exc:
            logger.debug(f"Puzzle genişletme atlandı: {exc}")
            return
        if not yeni:
            return
        self._puzzle.genisletme += 1
        self._counter["puzzle_genisletme"] = self._counter.get(
            "puzzle_genisletme", 0) + 1
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="plan", ok=True,
            text=f"dinamik alt faz: [{yaprak.id}] → {len(yeni)} parça"))
        self._puzzle_mesaji(messages, run)

    def _puzzle_kasaya(self, run: AgentRun) -> None:
        """⚠ C) Planı kasaya yazar (benzer görevlerde geri çağrılabilir)."""
        if self._puzzle is None or self._puzzle.tamamlanan() < 2:
            return
        try:
            kok = Path(self._workspace or ".").resolve()
            kasa = None
            for aday in (kok, *kok.parents):
                if (aday / "knowledge_base").is_dir():
                    kasa = aday / "knowledge_base"
                    break
            if kasa is None:
                kasa = kok / "knowledge_base"
            yol = self._puzzle.kasaya_yaz(kasa)
            if yol is not None:
                self._counter["puzzle_kasa"] = self._counter.get(
                    "puzzle_kasa", 0) + 1
        except Exception as exc:
            logger.debug(f"Puzzle kasaya yazılamadı: {exc}")

    def _faz_raporu(self, task: str, run: AgentRun,
                    messages: List[LLMMessage]) -> None:
        """
        Her `FAZ_ADIM` araç adımında **durum raporu** üretir ve devam eder.

        ⚠ Kullanıcı isteği: ajan işi fazlara bölüyor; her faz sonunda ne
        yaptığını, neyin kaldığını bildirip **kendiliğinden** devam etmeli.
        Rapor, modele "şu ana kadar" özeti verir; kullanıcıya arayüzde
        `plan` adımı olarak görünür.
        """
        if self.FAZ_ADIM <= 0:
            return
        arac_adimi = sum(1 for a in run.steps if a.kind == "tool")
        if arac_adimi == 0 or arac_adimi % self.FAZ_ADIM != 0:
            return
        ozet = self._hedef.ozet(sinir=900) if self._hedef is not None else ""
        satirlar = [f"FAZ DURUMU — {arac_adimi} araç adımı tamamlandı."]
        if ozet:
            satirlar.append(ozet)
        # ⚠ Hatalar ayrı alanda tutulmaz: adım kaydından türetilir
        hatalar = [a.text[:90] for a in run.steps
                   if a.kind in ("error", "recovery") and a.text][-3:]
        if hatalar:
            satirlar.append("Son hatalar: " + " · ".join(hatalar))
        satirlar.append(
            "Devam et. Kalan işi sırayla yap; faz bittiğinde kısa durum ver.")
        messages.append(LLMMessage("user", "\n".join(satirlar)))
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="plan", ok=True,
            text=f"faz raporu · {arac_adimi} adım · "
                 f"{len(self._hedef.facts) if self._hedef else 0} olgu"))
        self._counter["faz_raporu"] = self._counter.get("faz_raporu", 0) + 1
        # ⚠ Faz raporuyla birlikte puzzle'da bir sonraki parçaya geç
        try:
            self._puzzle_ilerle(run, messages)
        except Exception as exc:
            logger.debug(f"Puzzle ilerlemedi: {exc}")
        # ⚠ Kontrol noktasını güncelle (kesinti olursa kaldığı yer kalsın)
        try:
            self._kontrol_guncelle(run, "devam ediyor")
        except Exception:
            pass

        # ⚠ B) Aktif yaprak çok kaba ise dinamik alt faz aç
        try:
            self._puzzle_genislet(run, messages)
        except Exception as exc:
            logger.debug(f"Puzzle genişletme atlandı: {exc}")

    def _baglami_sikistir(self, messages: List[LLMMessage]) -> None:
        """
        ⚠ Uzun çalışmada mesaj geçmişini **hedef özetine** indirger.

        Neden: 20+ araç turundan sonra ham geçmiş token bütçesini yer ve
        modelin dikkatini dağıtır. Korunanlar: sistem istemi + son mesajlar.
        """
        if len(messages) <= SIKISTIRMA_ESIK:
            return
        ozet = self._hedef.ozet() if self._hedef is not None else ""
        if not ozet:
            return
        # ⚠ ÖLÇÜLDÜ: kuyruk bir `tool` mesajının ortasından başlarsa
        # sağlayıcı **HTTP 400** veriyor ("Messages with role 'tool' must be
        # a response to a preceding message with 'tool_calls'"). Bu yüzden
        # kuyruk, araç döngüsünün **başına** hizalanır.
        kuyruk = list(messages[-SIKISTIRMA_TUT:])
        while kuyruk and getattr(kuyruk[0], "role", "") == "tool":
            kuyruk.pop(0)
        while (kuyruk and getattr(kuyruk[0], "role", "") == "assistant"
               and getattr(kuyruk[0], "tool_calls", None)):
            # yanıtları düşmüş asistan turu da tek başına geçersizdir
            kuyruk.pop(0)
        yeni = list(messages[:1])
        yeni.append(LLMMessage(
            "user",
            "ÖZET (eski mesajlar sıkıştırıldı):\n" + ozet))
        yeni.extend(kuyruk)
        onceki = len(messages)
        messages[:] = yeni
        # ⚠ Kuyruğun SONUNDA kalan yetim `tool` mesajları da atılmalı:
        # asistan turu düştüyse sağlayıcı yine 400 verir.
        self._mesajlari_onar(messages)
        self._counter["context_compressed"] += 1
        logger.debug(f"Bağlam sıkıştırıldı: {onceki} → {len(messages)} mesaj")

    def _playbook_kaydet(self, task: str, run: AgentRun) -> None:
        """Başarılı akışı prosedür olarak saklar (ödül eşiği üstündeyse)."""
        store = self._playbooks
        if store is None or not run.ok:
            return
        adimlar: List[Dict[str, Any]] = []
        for step in run.steps:
            if step.kind == "tool" and step.tool:
                adimlar.append({"tool": step.tool,
                                "args": _kisa_args(step.args)})
        if len(adimlar) < 2:
            return
        odul = 0.85 if len(adimlar) >= 3 else 0.7
        pb = store.kaydet(task, adimlar, odul)
        if pb is not None:
            self._counter["playbook_saved"] += 1
        if self._kullanilan_playbook:
            try:
                store.kullanim_artir(self._kullanilan_playbook)
            except Exception:
                pass

    @staticmethod
    def _mesajlari_onar(messages: List[LLMMessage]) -> None:
        """
        ⚠ Yetim `tool` mesajlarını atar (sağlayıcı 400'ünü önler).

        Kural: bir `tool` mesajı, **kendisinden önceki** assistant turundaki
        `tool_calls` kimliklerinden biriyle eşleşmelidir. Sıkıştırma/kesme
        sonrası eşleşmeyen mesajlar sağlayıcı tarafından reddedilir.
        """
        bekleyen: set = set()
        temiz: List[LLMMessage] = []
        atilan = 0
        for m in messages:
            rol = getattr(m, "role", "")
            if rol == "assistant":
                cagrilar = getattr(m, "tool_calls", None) or []
                bekleyen = {str(c.get("id") or "") for c in cagrilar
                            if isinstance(c, dict)}
            elif rol == "tool":
                kimlik = str(getattr(m, "tool_call_id", "") or "")
                if kimlik and kimlik in bekleyen:
                    bekleyen.discard(kimlik)
                    temiz.append(m)
                else:
                    atilan += 1
                continue
            temiz.append(m)
        # ⚠ İkinci geçiş: **yanıtsız** tool_calls bırakmak da 400 üretir
        # ("tool_calls must be followed by tool messages"). Yanıtı olmayan
        # çağrıları assistant mesajından düşür; içerik korunur.
        sonuc: List[LLMMessage] = []
        for i, m in enumerate(temiz):
            if getattr(m, "role", "") == "assistant":
                cagrilar = getattr(m, "tool_calls", None) or []
                if cagrilar:
                    yanitlanan = {str(getattr(x, "tool_call_id", "") or "")
                                  for x in temiz[i + 1:]
                                  if getattr(x, "role", "") == "tool"}
                    eksik = [c for c in cagrilar
                             if str(c.get("id") or "") not in yanitlanan]
                    if eksik:
                        logger.debug(
                            f"Yanıtsız tool_calls temizlendi: {len(eksik)}")
                        try:
                            m.tool_calls = [c for c in cagrilar
                                            if c not in eksik] or None
                        except Exception:
                            pass
            sonuc.append(m)
        temiz = sonuc

        if atilan:
            logger.debug(f"Yetim tool mesajı atıldı: {atilan}")
        messages[:] = temiz

    def _kanit_metni(self, run: AgentRun, sinir: int = 2500) -> str:
        """Araç çıktılarından **kanıt** metni üretir (doğrulama için)."""
        parcalar: List[str] = []
        for step in run.steps:
            if step.kind == "tool" and step.result:
                parcalar.append(f"[{step.tool}] {step.result}")
        metin = "\n".join(parcalar)
        return metin[-sinir:] if len(metin) > sinir else metin

    def _cevabi_dogrula(self, task: str, cevap: str, run: AgentRun) -> str:
        """
        ⚠ Nihai yanıtı kanıta karşı denetler; desteksizse **bir kez**
        düzeltme ister. Sınırlı: en fazla 1 düzeltme çağrısı.
        """
        cog = self._cog
        if cog is None or not cog.verify:
            return cevap
        kanit = self._kanit_metni(run)
        if not kanit.strip():
            return cevap
        try:
            sonuc = cog.dogrula(task, cevap, kanit)
        except Exception as exc:
            logger.debug(f"Doğrulama çökmeden atlandı: {exc}")
            return cevap
        self._counter["cog_verify"] += 1
        if not sonuc.ok:
            return cevap
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="verify", ok=sonuc.supported,
            text=(f"kanıt: {'destekli' if sonuc.supported else 'DESTEKSİZ'} "
                  f"· puan {sonuc.score:.2f}"
                  + (f" · eksik: {len(sonuc.unsupported)}"
                     if sonuc.unsupported else "")),
        ))
        if sonuc.supported:
            return cevap
        if self._counter["cog_correct"] >= 1:
            # ⚠ İkinci düzeltme denenmez: kullanıcıyı boş bırakmamak için
            # eldeki yanıt korunur.
            return cevap
        self._counter["cog_correct"] += 1
        try:
            yeni = cog.duzelt(task, cevap, sonuc.unsupported, kanit)
        except Exception as exc:
            logger.debug(f"Düzeltme atlandı: {exc}")
            return cevap
        return yeni or cevap

    def _yansima_kapisi(self, task: str, run: AgentRun,
                        messages: List[LLMMessage]) -> None:
        """
        ⚠ Periyodik öz-yansıma: her `reflect_every` araç adımında bir kez,
        toplam `max_reflect` ile sınırlı. Karar modele **yönergedir**, akışı
        zorla değiştirmez.
        """
        cog = self._cog
        if cog is None or self._hedef is None:
            return
        if self._counter["cog_reflect"] >= cog.max_reflect:
            return
        arac_adimi = sum(1 for st in run.steps if st.kind == "tool")
        if arac_adimi < cog.reflect_every or arac_adimi % cog.reflect_every != 0:
            return
        try:
            sonuc = cog.yansit(task, self._hedef.ozet(),
                               getattr(self, "_son_hata", ""))
        except Exception as exc:
            logger.debug(f"Yansıma atlandı: {exc}")
            return
        self._counter["cog_reflect"] += 1
        if not sonuc.ok:
            return
        self._add_step(run, AgentStep(
            index=len(run.steps), kind="critique", ok=True,
            text=(f"yansıma: {sonuc.verdict}"
                  + (f" · eksik {len(sonuc.missing)}" if sonuc.missing else "")),
        ))
        if sonuc.verdict == "yeniden_planla":
            self._counter["cog_replan"] += 1
            eksik = ", ".join(sonuc.missing) or "belirsiz"
            messages.append(LLMMessage(
                "user",
                f"YENİDEN PLANLA: mevcut yaklaşım ilerlemiyor. Eksikler: {eksik}. "
                f"{sonuc.next_action or 'Küçük, doğrulanabilir bir adımla yeniden başla.'}"))
        elif sonuc.verdict == "bildir":
            messages.append(LLMMessage(
                "user",
                "KULLANICIYA BİLDİR: ilerlemek için izin/girdi gerekiyor. "
                f"{sonuc.next_action or ''}".strip()))
        elif sonuc.verdict == "cevapla":
            messages.append(LLMMessage(
                "user", "Elindeki kanıtla CEVAP VER; yeni araç çağırma."))

    def _execute(self, tool_name: str, arguments: Dict[str, Any],
                 run: AgentRun) -> Optional[ToolResult]:
        """
        Tek bir aracı çalıştırır; politika engellerse onay ister.

        Returns:
            `ToolResult`, veya kullanıcı reddettiyse None (döngü durmalı)
        """
        assert self._registry is not None

        started = time.perf_counter()
        context = self._build_context()

        # ⚠ Ortak havuz (adım 3): izinli **okuma** araçları önce komşuda
        # denenir. Yanıt gelmezse aşağıdaki yerel çağrı yapılır.
        uzak = self._uzaktan_dene(tool_name, arguments)
        if uzak is not None:
            result, kaynak = uzak
        else:
            result, kaynak = (
                self._registry.call(tool_name, arguments, context=context),
                "yerel",
            )

        # ── Onay gerekiyor mu? ──
        # Politika "onay gerekli" dediğinde çağrı **yapılmadan** döner ve
        # `needs_approval=True` olur. Onay alınırsa tek kullanımlık bir izin
        # verilip çağrı yinelenir; izin o çağrıda tükenir, böylece sonraki
        # adımda yeniden sorulur (bkz. ToolRegistry.approve).
        if result.needs_approval:
            tool = self._registry.get(tool_name)
            safety = tool.safety if tool else Safety.EXECUTE
            approved = self._request_approval(tool_name, arguments, safety,
                                              result.reason, run)
            if not approved:
                return None
            self._registry.approve(tool_name, consume=True)
            result = self._registry.call(tool_name, arguments, context=context)

        duration = (time.perf_counter() - started) * 1000
        self._counter["tool_calls"] += 1

        # ── Adım kaydı ──
        # ⚠ Kaynak görünür olsun: `[uzak:relay:25949]` / `[yerel]`
        step = AgentStep(
            index=len(run.steps), kind="tool", tool=tool_name, args=arguments,
            text=(f"[uzak:{kaynak}]" if kaynak != "yerel" else "[yerel]"),
            result=result.render(max_chars=3000), ok=result.ok,
            duration_ms=duration,
        )
        self._add_step(run, step)

        if tool_name not in run.tools_used:
            run.tools_used.append(tool_name)

        # ── Yan etkileri topla (özet için) ──
        data = result.data if isinstance(result.data, dict) else {}
        if tool_name in ("code.edit_file", "code.create_file") and result.ok:
            path = str(data.get("path", ""))
            if path and path not in run.files_changed:
                run.files_changed.append(path)
        if tool_name in ("code.run_command", "shell.run", "shell.session_run"):
            command = str(data.get("command") or arguments.get("command", ""))
            if command and command not in run.commands_run:
                run.commands_run.append(command[:200])

        return result

    def _build_context(self) -> ToolContext:
        """Araç çağrısı bağlamını üretir."""
        assert self._registry is not None
        return ToolContext(
            workspace=self._workspace,
            event_bus=self._event_bus,
            config=self._config,
            cancel_event=self._cancel,
            autonomy=self._registry.policy,
            dry_run=self._registry.policy.dry_run,
        )

    def _request_approval(self, tool_name: str, arguments: Dict[str, Any],
                          safety: Safety, reason: str, run: AgentRun) -> bool:
        """
        Kullanıcıdan onay ister.

        Geri çağrım yoksa (komut satırı modu) **reddedilir** — sessizce
        çalıştırmak, kullanıcının bilmediği bir değişikliği yapmak olurdu.
        """
        self._counter["approvals"] += 1
        run.approvals += 1
        label = SAFETY_LABELS.get(safety, safety.value)
        summary = _approval_summary(tool_name, arguments)

        step = AgentStep(
            index=len(run.steps), kind="approval",
            text=f"{tool_name} ({label}) — {summary}",
            tool=tool_name, args=arguments, ok=True,
        )
        self._add_step(run, step)
        self._emit("agent.approval_requested", {
            "tool": tool_name, "args": arguments, "safety": safety.value,
            "safety_label": label, "reason": reason, "summary": summary,
        })

        if self._approval is None:
            self._counter["denials"] += 1
            run.denials += 1
            self._emit("agent.approval_denied", {
                "tool": tool_name, "reason": "onay mekanizması yok",
            })
            return False

        try:
            decision = self._approval(tool_name, arguments, safety.value,
                                      reason, summary)
        except Exception as exc:
            logger.warning(f"Onay geri çağrımı hata verdi: {exc}")
            self._counter["denials"] += 1
            run.denials += 1
            return False

        approved, remember = _normalize_decision(decision)
        if approved:
            self._counter["approvals"] += 0
            if remember:
                self._registry.approve(tool_name, remember=True)
            self._emit("agent.approval_granted", {
                "tool": tool_name, "remember": remember,
            })
            logger.info(f"Onay verildi: {tool_name}{' (hatırla)' if remember else ''}")
        else:
            self._counter["denials"] += 1
            run.denials += 1
            self._emit("agent.approval_denied", {"tool": tool_name})
            logger.info(f"Onay reddedildi: {tool_name}")
        return approved

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def _add_step(self, run: AgentRun, step: AgentStep) -> None:
        """Adımı kaydeder ve dinleyiciyi bilgilendirir."""
        run.steps.append(step)
        self._counter["steps"] += 1
        if self._on_step is not None:
            try:
                self._on_step(step)
            except Exception as exc:
                logger.debug(f"Adım dinleyicisi hata verdi: {exc}")
        self._emit("agent.step", step.to_dict(max_result=1500))

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<AgentLoop mode={self.mode} runs={self._counter['runs']} "
                f"tools={len(self._registry) if self._registry else 0}>")


# ─── Modül Düzeyi Yardımcılar ───────────────────────────────────────────────

def _brief_args(args: Dict[str, Any], max_length: int = 70) -> str:
    """Argümanların kısa önizlemesi (özet satırları için)."""
    if not args:
        return ""
    parts: List[str] = []
    for key, value in list(args.items())[:3]:
        text = str(value).replace("\n", "⏎")
        if len(text) > 30:
            text = text[:30] + "…"
        parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    return joined[:max_length]


def _approval_summary(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Onay isteğinde kullanıcıya gösterilecek **anlaşılır** özet.

    Ortak uygulama `agent.tools.describe_tool_call` içindedir; arayüz paneli
    de aynı metni gösterir ki kullanıcı her yerde tutarlı bilgi görsün.
    """
    from neurocore.agent.tools import describe_tool_call

    return describe_tool_call(tool_name, arguments)


def _normalize_decision(decision: Any) -> Tuple[bool, bool]:
    """
    Onay geri çağrımının dönüşünü (onaylandı, hatırla) ikilisine çevirir.

    Kabul edilen biçimler: `True`, `False`, `None`, `"evet"`, `"hayir"`,
    `{"approved": True, "remember": False}`, `(True, True)`.
    """
    if decision is None:
        return False, False
    if isinstance(decision, bool):
        return decision, False
    if isinstance(decision, (tuple, list)) and decision:
        first = bool(decision[0])
        second = bool(decision[1]) if len(decision) > 1 else False
        return first, second
    if isinstance(decision, dict):
        return bool(decision.get("approved") or decision.get("allow")), \
            bool(decision.get("remember") or decision.get("always"))
    if isinstance(decision, str):
        lowered = decision.strip().lower()
        return lowered in ("evet", "e", "yes", "y", "onayla", "allow", "true", "1"), \
            lowered in ("hepsi", "all", "always", "hatırla")
    return bool(decision), False


_PATH_RE = re.compile(
    r"(?:[\w\-./\\]+\.(?:py|js|ts|tsx|jsx|gd|cs|json|yaml|yml|toml|md|txt|cfg|ini|"
    r"html|css|sh|ps1|bat|c|cpp|h|hpp|rs|go|java|kt|lua|gd|tscn|tres|blend|glsl|sql))",
    re.IGNORECASE,
)


def _extract_path(text: str) -> str:
    """Metinden dosya yolu çıkarır (kural tabanlı mod için)."""
    for match in _PATH_RE.finditer(text or ""):
        candidate = match.group(0).strip(".,;:'\"")
        if len(candidate) > 2:
            return candidate
    return ""


_COMMAND_RE = re.compile(
    r"(?:komutu(?:\s+çalıştır)?|çalıştır|execute|run)\s*[:\-]?\s*[`\"']([^`\"']{3,300})[`\"']",
    re.IGNORECASE,
)


def _extract_command(text: str) -> str:
    """Metinden açıkça belirtilmiş komutu çıkarır."""
    match = _COMMAND_RE.search(text or "")
    return match.group(1).strip() if match else ""


_STOPWORDS = {
    "ara", "bul", "nerede", "kullanılıyor", "kullanılır", "search", "find",
    "where", "göster", "yap", "et", "bir", "bu", "şu", "ve", "ile", "için",
}


def _extract_symbol(text: str) -> str:
    """
    Metinden aranacak sembol adını çıkarır.

    Türkçe cümlelerde sembol adı çoğunlukla bir **tür sözcüğünden** hemen önce
    gelir: "`kare` fonksiyonu", "`Hesap` sınıfı", "`topla` metodu". Bu kalıp
    tanınmazsa "kare fonksiyonu nerede" gibi en doğal istek boş döner ve ajan
    hiç arama yapmaz.

    Puanlama: `()` ve alt çizgi en güçlü işaretlerdir; tür sözcüğünden hemen
    önce gelmek de güçlü bir işarettir.
    """
    source = text or ""
    # DİKKAT: `[A-Za-z_]` Türkçe harfleri (ı, ş, ğ, ü, ö, ç, İ) kapsamaz.
    # Bu yüzden "Hesap sınıfını bul" cümlesinde "sınıfını" belirteci hiç
    # yakalanmaz ve tür sözcüğü eşleşmesi çalışmaz — ajan sembolü bulamaz.
    # `\w` Python 3'te Unicode duyarlıdır ve doğru davranır.
    tokens = re.findall(r"\w[\w]{2,}", source, re.UNICODE)
    tokens = [token for token in tokens if not token.isdigit()]
    candidates: List[Tuple[int, str, int]] = []

    for index, token in enumerate(tokens):
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in _TYPE_WORDS_TR:
            continue
        score = 0
        if "_" in token:
            score += 3
        if any(character.isupper() for character in token[1:]):
            score += 2
        if f"{token}()" in source or f"{token} (" in source:
            score += 4
        if 4 <= len(token) <= 40:
            score += 1
        # Bir sonraki belirteç tür sözcüğüyse bu bir sembol adıdır.
        # Türkçe ekler yüzünden kök eşleşmesi de denenir
        # ("sınıfını", "fonksiyonunun").
        if index + 1 < len(tokens):
            following = tokens[index + 1].lower()
            if following in _TYPE_WORDS_TR or any(
                following.startswith(root) for root in _TYPE_WORDS_TR
            ):
                score += 3
        if index == 0:
            score += 1
        candidates.append((score, token.strip("()"), index))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[2]))
    return candidates[0][1] if candidates[0][0] >= 3 else ""


#: Türkçe "tür" sözcükleri — sembol adından hemen sonra gelirler
_TYPE_WORDS_TR = {
    "fonksiyon", "fonksiyonu", "fonksiyonun", "fonksiyonunun", "fonksiyonlar",
    "sınıf", "sınıfı", "sınıfın", "sınıfını", "metot", "metodu", "metodun",
    "metodunu", "değişken", "değişkeni", "değişkenin", "yöntem", "yöntemi",
    "modül", "modülü", "paket", "paketi", "dosya", "dosyası", "sembol",
    "sembolü", "arayüz", "arayüzü", "sabit", "sabiti", "parametre",
    "parametresi", "özellik", "özelliği",
}


__all__ = [
    "AgentLoop",
    "AgentRun",
    "AgentStep",
    "RuleRouter",
    "RulePattern",
    "RULE_PATTERNS",
    "SYSTEM_PROMPT",
    "PLAN_PROMPT",
]
