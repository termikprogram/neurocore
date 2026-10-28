"""
NeuroCore Komut Satırı Arayüzü
===============================
Sistemi terminalden sürmek için tam CLI.

Kullanım::

    python -m neurocore init                       # ilk kurulum (tohum + tokenizer)
    python -m neurocore ask "GPU neden ısınır?"    # sorgu
    python -m neurocore ask --deep "X ile Y karşılaştır"
    python -m neurocore learn notlar.md --tags donanım
    python -m neurocore research "termal yönetim" --depth 2
    python -m neurocore train --steps 200
    python -m neurocore evolve --cycles 2
    python -m neurocore search "termal macun"
    python -m neurocore graph --export
    python -m neurocore health --deep
    python -m neurocore stats --json
    python -m neurocore shell                      # etkileşimli mod

Global seçenekler:
    --base PATH      Proje kökü (varsayılan: bu dosyanın iki üst dizini)
    --vault PATH     Kasa yolu
    --no-gpu         GPU kullanma
    --profile NAME   Donanım profilini zorla (minimal/low/medium/high/ultra)
    --quiet          Yalnızca sonucu yazdır
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - zengin çıktı isteğe bağlı
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    HAS_RICH = True
except ImportError:  # pragma: no cover
    Console = None  # type: ignore
    HAS_RICH = False

from neurocore.core.config import HardwareProfile, SystemConfig
from neurocore.core.engine import EngineState, NeuroCoreEngine

_console = Console() if HAS_RICH else None


# ─── Çıktı Yardımcıları ─────────────────────────────────────────────────────

def _print(text: str = "", style: str = "", quiet: bool = False) -> None:
    """Metni (varsa renkli) yazdırır."""
    if quiet:
        return
    if _console is not None:
        _console.print(text, style=style or None, highlight=False)
    else:
        print(text)


def _print_markdown(text: str, quiet: bool = False) -> None:
    """Markdown metnini biçimli yazdırır (rich yoksa düz)."""
    if quiet and not text:
        return
    if _console is not None and HAS_RICH:
        _console.print(Markdown(text))
    else:
        print(text)


def _print_json(payload: Any) -> None:
    """JSON çıktısı."""
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _rule(title: str = "") -> None:
    """Ayırıcı çizgi."""
    if _console is not None:
        _console.rule(title or None)
    elif title:
        print(f"{'─' * 20} {title} {'─' * 20}")


def _panel(title: str, body: str, style: str = "cyan") -> None:
    """Çerçeveli kutu."""
    if _console is not None and HAS_RICH:
        _console.print(Panel(body, title=title, border_style=style))
    else:
        print(f"[{title}]\n{body}\n")


# ─── Motor Fabrikası ────────────────────────────────────────────────────────

def build_engine(args: argparse.Namespace, *, quiet: bool = False,
                 start: bool = False) -> NeuroCoreEngine:
    """
    CLI argümanlarından motor oluşturur ve başlatır.

    Args:
        args: Ayrıştırılmış argümanlar
        quiet: Sessiz mod
        start: Arka plan döngülerini de başlat

    Returns:
        Hazır `NeuroCoreEngine`
    """
    config = SystemConfig()

    if getattr(args, "base", None):
        config.base_path = str(Path(args.base).expanduser().resolve())
        config.knowledge.vault_path = str(Path(config.base_path) / "knowledge_base")
        config.data_path = str(Path(config.base_path) / "data")
        config.models_path = str(Path(config.base_path) / "models")

    if getattr(args, "vault", None):
        config.knowledge.vault_path = str(Path(args.vault).expanduser().resolve())

    if getattr(args, "no_gpu", False):
        config.device = "cpu"
        config.hardware.enable_gpu = False

    if getattr(args, "log_level", None):
        config.log_level = args.log_level

    engine = NeuroCoreEngine(config)
    if getattr(args, "profile", None):
        # Profili zorla: donanım algılamayı atla ama katmanları yine kur
        try:
            profile = HardwareProfile(str(args.profile).lower())
            config.apply_hardware_profile(profile)
            engine.initialize(auto_profile=False)
        except ValueError:
            _print(f"! Bilinmeyen profil: {args.profile} — otomatik algılama kullanılıyor",
                   style="yellow", quiet=quiet)
            engine.initialize()
    else:
        engine.initialize()

    if start:
        engine.start()
    return engine


# ─── Komutlar ───────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> int:
    """İlk kurulum: kasa tohumu + tokenizer eğitimi."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        _rule("İlk kurulum")
        _print(f"Kasa: {engine.knowledge.vault_path}", quiet=quiet)
        _print(f"Profil: {engine.config.hardware_profile} · cihaz: {engine.config.device}",
               quiet=quiet)

        outcome = engine.bootstrap(
            seed_knowledge=not args.no_seed,
            train_tokenizer=not args.no_tokenizer,
            target_vocab=args.vocab,
            force=args.force,
        )
        if args.json:
            _print_json(outcome)
            return 0

        _print(f"\nTohum notları: {outcome.get('seeded_notes', 0)}", quiet=quiet)
        tokenizer = outcome.get("tokenizer") or {}
        if tokenizer:
            _print(
                f"Tokenizer: {tokenizer.get('vocab_size')} token, "
                f"{tokenizer.get('merges')} birleştirme",
                quiet=quiet,
            )
        _print(f"Sözlük boyutu: {outcome.get('vocab_size', 0)}", quiet=quiet)
        if outcome.get("seed_error"):
            _print(f"! Tohumlama hatası: {outcome['seed_error']}", style="yellow", quiet=quiet)
        if outcome.get("tokenizer_error"):
            _print(f"! Tokenizer hatası: {outcome['tokenizer_error']}",
                   style="yellow", quiet=quiet)

        _print("\nSıradaki adımlar:", quiet=quiet)
        _print("  1. python -m neurocore learn <dosya>   # bilgi ekle", quiet=quiet)
        _print("  2. python -m neurocore train --steps 300", quiet=quiet)
        _print("  3. python -m neurocore ask \"...\"", quiet=quiet)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Soru sorar."""
    quiet = args.quiet
    engine = build_engine(args, quiet=quiet)
    try:
        result = engine.query(
            args.question,
            deep=args.deep,
            limit=args.limit,
            learn=not args.no_learn,
            use_memory=not args.no_memory,
        )
        if args.json:
            _print_json(result.to_dict(include_context=args.show_context))
            return 0 if result.ok else 1

        if args.show_context and result.context:
            _rule("Bağlam")
            _print(result.context[:4000], quiet=quiet)
            _rule("Yanıt")

        _print_markdown(result.render(), quiet=quiet)
        if result.gaps and not quiet:
            _print("\n! Bilgi boşlukları: " + "; ".join(result.gaps[:4]), style="yellow")
        if result.learned_note_id and not quiet:
            _print(f"\n→ Yanıt not olarak kaydedildi: {result.learned_note_id}",
                   style="dim")
        if args.feedback:
            signal = engine.feedback(positive=(args.feedback == "up"))
            _print(f"Geri bildirim: {signal}", style="dim", quiet=quiet)
        return 0 if result.ok else 1
    finally:
        engine.shutdown()


def cmd_learn(args: argparse.Namespace) -> int:
    """Dosya veya metinden bilgi öğretir."""
    quiet = args.quiet
    text = args.text or ""

    if args.file:
        path = Path(args.file).expanduser()
        if not path.exists():
            _print(f"! Dosya bulunamadı: {path}", style="red", quiet=quiet)
            return 1
        try:
            text = path.read_text(encoding=args.encoding, errors="replace")
        except OSError as exc:
            _print(f"! Dosya okunamadı: {exc}", style="red", quiet=quiet)
            return 1
        if not args.title:
            args.title = path.stem.replace("_", " ").replace("-", " ").title()

    if not text.strip():
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        if not text.strip():
            _print("! Öğretilecek içerik yok (dosya veya metin verin)",
                   style="red", quiet=quiet)
            return 1

    engine = build_engine(args, quiet=quiet)
    try:
        outcome = engine.learn(
            text,
            title=args.title,
            tags=args.tags,
            kind=args.kind,
            confidence=args.confidence,
            source=args.source or (f"file://{args.file}" if args.file else ""),
            split=not args.no_split,
        )
        if args.json:
            _print_json(outcome)
            return 0 if outcome.get("ok", True) else 1

        if outcome.get("ok", True):
            _print(f"✓ Öğrenildi: {outcome.get('title')} ({outcome.get('id')})",
                   style="green", quiet=quiet)
            _print(
                f"  {outcome.get('words', 0)} kelime · {outcome.get('chunks', 1)} parça · "
                f"{outcome.get('linked', 0)} bağlantı · güven {outcome.get('confidence', 0):.2f}",
                quiet=quiet,
            )
            if outcome.get("tags"):
                _print(f"  etiketler: {', '.join(outcome['tags'][:10])}", quiet=quiet)
            if outcome.get("errors"):
                _print(f"  ! {len(outcome['errors'])} uyarı", style="yellow", quiet=quiet)
            return 0
        _print(f"! Öğrenilemedi: {outcome.get('error')}", style="red", quiet=quiet)
        return 1
    finally:
        engine.shutdown()


def cmd_research(args: argparse.Namespace) -> int:
    """Konu araştırır ve öğrenir."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        _rule(f"Araştırma: {args.topic}")
        report = engine.research(args.topic, depth=args.depth, learn=not args.no_learn)

        if args.json:
            _print_json(report)
            return 0 if report.get("ok") else 1

        _print(
            f"{report['rounds']} tur · {len(report['findings'])} bulgu · "
            f"{len(report['notes_written'])} not · {report['duration_ms']:.0f}ms",
            quiet=quiet,
        )
        for finding in report["findings"][:10]:
            _print(
                f"  • [{finding.get('origin', '?')}] {finding.get('title', '')} "
                f"(skor {finding.get('score', 0):.2f})",
                quiet=quiet,
            )
        if report["notes_written"]:
            _print("\nYazılan notlar:", quiet=quiet)
            for note in report["notes_written"]:
                _print(f"  → {note['title']} ({note['id']})", quiet=quiet)
        if report.get("gaps"):
            _print("\nAraştırma önerileri:", quiet=quiet)
            for gap in report["gaps"][:5]:
                _print(f"  ? {gap}", quiet=quiet)
        return 0 if report.get("ok") else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Modeli eğitir."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        _rule("Eğitim")
        _print(f"Model: {engine.model.describe()}", quiet=quiet)
        outcome = engine.train(
            steps=args.steps,
            from_vault=not args.no_vault,
            from_experiences=not args.no_experience,
            max_minutes=args.minutes,
        )
        if args.json:
            _print_json(outcome)
            return 0 if outcome.get("ok") else 1

        if not outcome.get("ok"):
            _print(f"! Eğitim yapılamadı: {outcome.get('error')}", style="red", quiet=quiet)
            if outcome.get("hint"):
                _print(f"  ipucu: {outcome['hint']}", quiet=quiet)
            return 1

        _print(f"✓ {outcome['steps_completed']}/{outcome['steps_requested']} adım",
               style="green", quiet=quiet)
        _print(
            f"  kayıp: {outcome['initial_loss']:.4f} → {outcome['final_loss']:.4f} "
            f"(%{outcome.get('loss_reduction_percent', 0):.1f} azalma)",
            quiet=quiet,
        )
        _print(
            f"  {outcome['tokens_seen']:,} token · "
            f"{outcome['throughput_tokens_per_sec']:.0f} token/s · "
            f"{outcome['duration_ms'] / 1000:.1f}s · durma: {outcome['stop_reason']}",
            quiet=quiet,
        )
        if outcome.get("validation"):
            validation = outcome["validation"]
            _print(
                f"  doğrulama: kayıp {validation.get('loss')} · "
                f"perplexity {validation.get('perplexity')}",
                quiet=quiet,
            )
        return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """Evrim turu çalıştırır."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        _rule("Evrim")
        outcome = engine.evolve(
            cycles=args.cycles,
            dry_run=args.dry_run,
            train=not args.no_train,
        )
        if args.json:
            _print_json(outcome)
            return 0

        for report in outcome.get("reports", []):
            _print(
                f"Döngü {report.get('cycle')}: {report.get('evaluated', 0)} değerlendirme, "
                f"{report.get('synthesized', 0)} sentez, {report.get('curated', 0)} müfredat, "
                f"{report.get('grow_decisions', 0)} büyüme kararı",
                quiet=quiet,
            )
            _print(
                f"  kalite: {report.get('quality_before', 0):.2f} → "
                f"{report.get('quality_after', 0):.2f} · "
                f"bütünlük: {report.get('integrity_score', 0):.2f}",
                quiet=quiet,
            )
            for lesson in (report.get("lessons") or [])[:5]:
                _print(f"  • {lesson}", quiet=quiet)

        growth = outcome.get("growth")
        if growth:
            decision = growth.get("decision", {})
            _print(
                f"\nBüyüme kararı: {decision.get('action')} — {decision.get('reason')}",
                quiet=quiet,
            )
            _print(f"  uygulandı: {growth.get('applied', {}).get('applied')}", quiet=quiet)

        training = outcome.get("training")
        if training and training.get("steps_completed"):
            _print(
                f"\nEğitim: {training['steps_completed']} adım, "
                f"kayıp {training['initial_loss']:.4f} → {training['final_loss']:.4f}",
                quiet=quiet,
            )
        elif training and training.get("error"):
            _print(f"\n! Eğitim hatası: {training['error']}", style="yellow", quiet=quiet)

        _print(f"\nSüre: {outcome['duration_ms']:.0f}ms", quiet=quiet)
        return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Kasadaki bilgileri arar (yanıt üretmeden)."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        hits = engine.search(args.query, limit=args.limit)
        if args.json:
            _print_json(hits)
            return 0 if hits else 1
        if not hits:
            _print("Sonuç bulunamadı.", style="yellow", quiet=quiet)
            return 1
        table = Table(title=f"Arama: {args.query}") if HAS_RICH and _console else None
        if table is not None:
            table.add_column("Skor", justify="right")
            table.add_column("Başlık")
            table.add_column("Tür")
            table.add_column("Güven", justify="right")
            for hit in hits:
                table.add_row(
                    f"{hit.get('score', 0):.3f}",
                    str(hit.get("title", ""))[:60],
                    str(hit.get("kind", "")),
                    f"{hit.get('confidence', 0):.2f}",
                )
            _console.print(table)  # type: ignore[union-attr]
        else:
            for hit in hits:
                print(f"{hit.get('score', 0):.3f}  {hit.get('title', '')}")
        return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Bilgi grafı raporu."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        if args.json:
            _print_json(engine.knowledge.metrics())
            return 0
        if args.write:
            outcome = engine.graph_note(write=True)
            _print(f"✓ Graf notu yazıldı: {outcome.get('title')} ({outcome.get('id')})",
                   style="green", quiet=quiet)
            return 0
        if args.export:
            path = args.export if isinstance(args.export, str) else None
            target = engine.knowledge.export_graph(path)
            _print(f"✓ Graf dışa aktarıldı: {target}", style="green", quiet=quiet)
            return 0
        markdown = engine.graph_note().get("markdown", "")
        _print_markdown(markdown, quiet=quiet)
        return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Sağlık raporu."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        report = engine.health_report(deep=args.deep)
        if args.json:
            _print_json(report.to_dict())
            return 0
        _panel("Sağlık", report.render(),
               style="green" if report.ok else "yellow")
        if args.deep:
            for name, info in report.layers.items():
                if isinstance(info, dict) and info.get("issues"):
                    _print(f"\n{name}:", style="bold", quiet=quiet)
                    for issue in info["issues"][:10]:
                        _print(f"  • {issue}", quiet=quiet)
        return 0 if report.ok else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """İstatistikler."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        stats = engine.get_stats()
        if args.json or not HAS_RICH or not _console:
            _print_json(stats)
            return 0

        _rule("NeuroCore İstatistikleri")
        table = Table(show_header=True)
        table.add_column("Katman")
        table.add_column("Özet")
        summary_rows = [
            ("motor", f"{stats.get('state')} · {stats.get('profile')}/{stats.get('device')}"),
            ("sorgular", str(stats.get("counters", {}).get("queries", 0))),
            ("öğrenmeler", str(stats.get("counters", {}).get("learnings", 0))),
            ("olaylar", str(stats.get("event_bus", {}).get("total_events", 0))),
        ]
        model = stats.get("model", {})
        if model:
            summary_rows.append((
                "model",
                f"{model.get('params', 0):,} parametre · "
                f"{'eğitilmiş' if model.get('pretrained') else 'EĞİTİLMEMİŞ'} · "
                f"nesil {model.get('growth', {}).get('generation', 0)}",
            ))
        knowledge = stats.get("knowledge", {})
        if knowledge:
            summary_rows.append((
                "bilgi",
                f"{knowledge.get('vault', {}).get('indexed_notes', 0)} not · "
                f"{knowledge.get('graph', {}).get('nodes', 0)} düğüm",
            ))
        memory = stats.get("memory", {})
        if memory:
            summary_rows.append((
                "bellek", f"{memory.get('size', 0)}/{memory.get('capacity', 0)} kayıt"
            ))
        evolution = stats.get("evolution", {})
        if evolution:
            summary_rows.append((
                "evrim",
                f"{evolution.get('buffer_size', evolution.get('experiences', {}).get('count', 0))} "
                f"deneyim",
            ))
        for name, summary in summary_rows:
            table.add_row(name, summary)
        _console.print(table)  # type: ignore[union-attr]
        return 0


def cmd_resource(args: argparse.Namespace) -> int:
    """
    Kaynak nezaketi raporu ve politikası.

    Sistemin ne kadarını kullandığımızı ve kullanıcı bilgisayarı kullanırken
    ajanın nasıl küçüldüğünü gösterir. Politika buradan değiştirilebilir.
    """
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        policy = engine.config.hardware.headroom
        changed: Dict[str, Any] = {}
        if args.factor is not None:
            factor = max(0.05, min(1.0, float(args.factor)))
            policy.cpu = policy.ram = policy.gpu = factor
            changed["factor"] = factor
        if args.max_ram is not None:
            policy.max_ram_percent = max(0.10, min(0.95, float(args.max_ram)))
            changed["max_ram_percent"] = policy.max_ram_percent
        if args.max_gpu is not None:
            policy.max_gpu_percent = max(0.10, min(0.95, float(args.max_gpu)))
            changed["max_gpu_percent"] = policy.max_gpu_percent
        if args.no_limit:
            engine.config.hardware.enforce_headroom = False
            changed["enforce_headroom"] = False
        if args.enforce:
            engine.config.hardware.enforce_headroom = True
            changed["enforce_headroom"] = True

        if changed:
            if engine.allocator is not None:
                engine.allocator.set_policy(policy)
            engine._refresh_budget(force=True)
            _print(f"Politika güncellendi: {changed}", style="green", quiet=quiet)

        report = engine.resource_report()
        if args.json:
            _print_json(report)
            return 0

        _rule("Kaynak Nezaketi")
        _print(report["summary"], quiet=quiet)
        _print("", quiet=quiet)
        _print(f"Politika: {report['policy'].get('summary', '')}", quiet=quiet)
        _print(
            f"Zorunlu: {engine.config.hardware.enforce_headroom} · "
            f"baskı: {report['headroom'].get('pressure')} · "
            f"duraklat: {report['headroom'].get('should_stall')}",
            quiet=quiet,
        )
        _print("", quiet=quiet)

        budget = report["budget"]
        if HAS_RICH and _console is not None:
            table = Table(title="Ajanın bütçesi")
            table.add_column("Parametre")
            table.add_column("Değer", justify="right")
            for key, value in budget.items():
                table.add_row(str(key), str(value))
            _console.print(table)
        else:
            for key, value in budget.items():
                print(f"{key}: {value}")

        if report["headroom"].get("should_stall"):
            _print(
                f"\n! Sistem dolu — ağır iş başlatılmayacak: "
                f"{report['headroom'].get('stall_reason')}",
                style="yellow", quiet=quiet,
            )
        return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Tek satır durum."""
    with build_engine(args, quiet=True) as engine:
        if args.json:
            _print_json(engine.neural_status())
        else:
            print(engine.status_line())
            print(engine.model.describe() if engine.model is not None else "")
            try:
                print(engine.resource_report()["summary"])
            except Exception:
                pass
    return 0


def cmd_os(args: argparse.Namespace) -> int:
    """İşletim sistemi işlemleri (sandbox denetimli)."""
    quiet = args.quiet
    with build_engine(args, quiet=quiet) as engine:
        if engine.os_layer is None:
            _print("! OS katmanı kurulmadı", style="red", quiet=quiet)
            return 1
        if args.report:
            report = engine.os_layer.system_report(write_to=args.report)
            _print(f"✓ Rapor yazıldı: {args.report}", style="green", quiet=quiet)
            if args.json:
                _print_json(report)
            return 0
        if args.command:
            result = engine.os_layer.run(args.command)
            if args.json:
                _print_json(result)
            else:
                _print(result.get("stdout") or result.get("stderr") or "",
                       quiet=quiet)
            return 0 if result.get("ok") else 1
        if args.read:
            _print(engine.os_layer.read(args.read), quiet=quiet)
            return 0
        if args.list:
            for item in engine.os_layer.list_dir(args.list)[:100]:
                print(item)
            return 0
        if args.find:
            for match in engine.os_layer.find(args.find[0], args.find[1])[:100]:
                print(match)
            return 0
        if args.json:
            _print_json(engine.os_layer.system_report())
        else:
            _print(engine.os_layer.health().get("summary", ""), quiet=quiet)
        return 0


def cmd_shell(args: argparse.Namespace) -> int:
    """Etkileşimli kabuk."""
    quiet = False
    with build_engine(args, quiet=quiet, start=True) as engine:
        _panel(
            "NeuroCore Etkileşimli Mod",
            "Komutlar:\n"
            "  /deep <soru>     çok adımlı akıl yürütme\n"
            "  /search <sorgu>  yalnızca kasa araması\n"
            "  /learn <metin>   bilgi öğret\n"
            "  /research <konu> araştır\n"
            "  /train [adım]    modeli eğit\n"
            "  /evolve [döngü]  evrim turu\n"
            "  /graph           graf raporu\n"
            "  /health          sağlık raporu\n"
            "  /status          durum özeti\n"
            "  /up  /down       son yanıta geri bildirim\n"
            "  /quit            çık\n"
            "Diğer her şey soru olarak yorumlanır.",
        )
        while True:
            try:
                line = input("\nneurocore> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                break

            try:
                if line.startswith("/deep "):
                    result = engine.query(line[6:], deep=True)
                    _print_markdown(result.render())
                elif line.startswith("/search "):
                    hits = engine.search(line[8:], limit=8)
                    for hit in hits:
                        print(f"{hit.get('score', 0):.3f}  {hit.get('title', '')}")
                elif line.startswith("/learn "):
                    outcome = engine.learn(line[7:])
                    print(f"✓ {outcome.get('title')} ({outcome.get('id')})"
                          if outcome.get("ok", True) else f"! {outcome.get('error')}")
                elif line.startswith("/research "):
                    report = engine.research(line[10:], depth=2)
                    print(f"{len(report['findings'])} bulgu, "
                          f"{len(report['notes_written'])} not")
                elif line.startswith("/train"):
                    parts = line.split()
                    outcome = engine.train(steps=int(parts[1]) if len(parts) > 1 else None)
                    print(f"✓ {outcome.get('steps_completed')} adım, "
                          f"kayıp {outcome.get('final_loss')}"
                          if outcome.get("ok") else f"! {outcome.get('error')}")
                elif line.startswith("/evolve"):
                    parts = line.split()
                    cycles = int(parts[1]) if len(parts) > 1 else 1
                    outcome = engine.evolve(cycles=cycles, train=True)
                    print(f"✓ {cycles} döngü, {outcome['duration_ms']:.0f}ms")
                elif line == "/graph":
                    _print_markdown(engine.graph_note().get("markdown", ""))
                elif line == "/health":
                    _panel("Sağlık", engine.health_report().render())
                elif line == "/status":
                    print(engine.status_line())
                    print(engine.model.describe())
                elif line in ("/up", "/down"):
                    print(engine.feedback(positive=(line == "/up")))
                elif line.startswith("/"):
                    print("! Bilinmeyen komut. /quit ile çıkabilirsiniz.")
                else:
                    result = engine.query(line)
                    _print_markdown(result.render())
            except KeyboardInterrupt:
                print("\n(kesildi)")
            except Exception as exc:
                print(f"! Hata: {exc}")
    return 0


# ─── Argüman Ayrıştırma ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """CLI argüman ayrıştırıcısını kurar."""
    parser = argparse.ArgumentParser(
        prog="neurocore",
        description="NeuroCore AI — donanım-farkında, çok-modlu, kendi kendini büyüten yapay zeka",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base", help="Proje kökü (varsayılan: otomatik)")
    parser.add_argument("--vault", help="Bilgi kasası yolu")
    parser.add_argument("--profile", choices=[p.value for p in HardwareProfile],
                        help="Donanım profilini zorla")
    parser.add_argument("--no-gpu", action="store_true", help="GPU kullanma")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Günlük seviyesi")
    parser.add_argument("--quiet", "-q", action="store_true", help="Sessiz mod")

    subparsers = parser.add_subparsers(dest="command", metavar="KOMUT")

    # init
    init_parser = subparsers.add_parser("init", help="İlk kurulum (tohum + tokenizer)")
    init_parser.add_argument("--no-seed", action="store_true", help="Tohum not ekleme")
    init_parser.add_argument("--no-tokenizer", action="store_true",
                             help="Tokenizer eğitme")
    init_parser.add_argument("--vocab", type=int, default=4000, help="Hedef sözlük boyutu")
    init_parser.add_argument("--force", action="store_true", help="Zorla yeniden yap")
    init_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    init_parser.set_defaults(func=cmd_init)

    # ask
    ask_parser = subparsers.add_parser("ask", help="Soru sor")
    ask_parser.add_argument("question", help="Soru metni")
    ask_parser.add_argument("--deep", action="store_true", help="Çok adımlı akıl yürütme")
    ask_parser.add_argument("--limit", type=int, default=8, help="Kullanılacak not sayısı")
    ask_parser.add_argument("--no-learn", action="store_true", help="Yanıttan not üretme")
    ask_parser.add_argument("--no-memory", action="store_true", help="Belleği kullanma")
    ask_parser.add_argument("--show-context", action="store_true", help="Bağlamı göster")
    ask_parser.add_argument("--feedback", choices=["up", "down"], help="Geri bildirim ver")
    ask_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    ask_parser.set_defaults(func=cmd_ask)

    # learn
    learn_parser = subparsers.add_parser("learn", help="Bilgi öğret")
    learn_parser.add_argument("text", nargs="?", help="Öğretilecek metin")
    learn_parser.add_argument("--file", "-f", help="Dosyadan oku")
    learn_parser.add_argument("--title", "-t", help="Başlık")
    learn_parser.add_argument("--tags", "-g", nargs="*", help="Etiketler")
    learn_parser.add_argument("--kind", default="concept",
                              choices=["concept", "entity", "source", "project",
                                       "session", "skill", "journal", "meta",
                                       "research", "summary"],
                              help="Not türü")
    learn_parser.add_argument("--confidence", type=float, default=0.6, help="Güven (0..1)")
    learn_parser.add_argument("--source", help="Kaynak etiketi")
    learn_parser.add_argument("--encoding", default="utf-8", help="Dosya kodlaması")
    learn_parser.add_argument("--no-split", action="store_true", help="İçeriği bölme")
    learn_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    learn_parser.set_defaults(func=cmd_learn)

    # research
    research_parser = subparsers.add_parser("research", help="Konu araştır")
    research_parser.add_argument("topic", help="Araştırma konusu")
    research_parser.add_argument("--depth", type=int, default=2, help="Derinlik")
    research_parser.add_argument("--no-learn", action="store_true", help="Not yazma")
    research_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    research_parser.set_defaults(func=cmd_research)

    # train
    train_parser = subparsers.add_parser("train", help="Modeli eğit")
    train_parser.add_argument("--steps", type=int, help="Adım sayısı")
    train_parser.add_argument("--minutes", type=float, help="Zaman sınırı (dakika)")
    train_parser.add_argument("--no-vault", action="store_true", help="Kasa verisini kullanma")
    train_parser.add_argument("--no-experience", action="store_true",
                              help="Deneyim verisini kullanma")
    train_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    train_parser.set_defaults(func=cmd_train)

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Evrim turu")
    evolve_parser.add_argument("--cycles", type=int, default=1, help="Döngü sayısı")
    evolve_parser.add_argument("--dry-run", action="store_true", help="Yalnızca planla")
    evolve_parser.add_argument("--no-train", action="store_true", help="Eğitim yapma")
    evolve_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    evolve_parser.set_defaults(func=cmd_evolve)

    # search
    search_parser = subparsers.add_parser("search", help="Kasadaki bilgileri ara")
    search_parser.add_argument("query", help="Arama sorgusu")
    search_parser.add_argument("--limit", type=int, default=10, help="Sonuç sayısı")
    search_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    search_parser.set_defaults(func=cmd_search)

    # graph
    graph_parser = subparsers.add_parser("graph", help="Bilgi grafı")
    graph_parser.add_argument("--export", nargs="?", const="", help="JSON olarak dışa aktar")
    graph_parser.add_argument("--write", action="store_true", help="Kasaya not olarak yaz")
    graph_parser.add_argument("--json", action="store_true", help="JSON metrikler")
    graph_parser.set_defaults(func=cmd_graph)

    # health
    health_parser = subparsers.add_parser("health", help="Sağlık raporu")
    health_parser.add_argument("--deep", action="store_true", help="Ayrıntılı denetim")
    health_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    health_parser.set_defaults(func=cmd_health)

    # stats
    stats_parser = subparsers.add_parser("stats", help="İstatistikler")
    stats_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    stats_parser.set_defaults(func=cmd_stats)

    # status
    status_parser = subparsers.add_parser("status", help="Tek satır durum")
    status_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    status_parser.set_defaults(func=cmd_status)

    # resource
    resource_parser = subparsers.add_parser(
        "resource", help="Kaynak nezaketi: ne kadarını kullanıyoruz?"
    )
    resource_parser.add_argument(
        "--factor", type=float,
        help="Nezaket çarpanı (varsayılan 0.90 = boştakinin %%90'ı)",
    )
    resource_parser.add_argument("--max-ram", type=float,
                                 help="RAM üst sınırı (0..1, varsayılan 0.85)")
    resource_parser.add_argument("--max-gpu", type=float,
                                 help="VRAM üst sınırı (0..1, varsayılan 0.85)")
    resource_parser.add_argument("--no-limit", action="store_true",
                                 help="Nezaket politikasını devre dışı bırak")
    resource_parser.add_argument("--enforce", action="store_true",
                                 help="Nezaket politikasını etkinleştir")
    resource_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    resource_parser.set_defaults(func=cmd_resource)

    # os
    os_parser = subparsers.add_parser("os", help="İşletim sistemi işlemleri")
    os_parser.add_argument("--report", nargs="?", const="", help="Envanter raporu yaz")
    os_parser.add_argument("--command", "-c", help="Komut çalıştır")
    os_parser.add_argument("--read", help="Dosya oku")
    os_parser.add_argument("--list", help="Dizin listele")
    os_parser.add_argument("--find", nargs=2, metavar=("KÖK", "DESEN"), help="Dosya ara")
    os_parser.add_argument("--json", action="store_true", help="JSON çıktı")
    os_parser.set_defaults(func=cmd_os)

    # shell
    shell_parser = subparsers.add_parser("shell", help="Etkileşimli mod")
    shell_parser.set_defaults(func=cmd_shell)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI giriş noktası.

    Args:
        argv: Argüman listesi (None = `sys.argv`)

    Returns:
        Çıkış kodu (0 = başarılı)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    # Günlük seviyesini uygula (konsol gürültüsünü CLI'dan yönet)
    try:
        from neurocore.utils.logger import NeuroCoreLogger
        level = "ERROR" if args.quiet else args.log_level
        console = not args.quiet
        NeuroCoreLogger.initialize(
            log_dir=str(Path(getattr(args, "base", None) or Path(__file__).parent.parent)
                        / "data" / "logs"),
            level=level, console=console, file=True, force=True,
        )
        NeuroCoreLogger.set_module_level("", level)
    except Exception:
        pass

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _print("\n(kesildi)", style="yellow")
        return 130
    except Exception as exc:
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        _print(f"! Hata: {exc}", style="red")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
