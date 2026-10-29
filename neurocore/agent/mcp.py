"""
MCP istemcisi — Model Context Protocol ile dış uygulamalara bağlanma
=====================================================================
Blender ve Godot gibi uygulamalar MCP sunucusu çalıştırabilir; bu modül
onlara **stdio** üzerinden bağlanır ve araçlarını NeuroCore'a kazandırır.

Protokol (MCP 2024-11-05, stdio taşıma)
---------------------------------------
```
→ {"jsonrpc":"2.0","id":1,"method":"initialize",
   "params":{"protocolVersion":"2024-11-05","capabilities":{},
             "clientInfo":{"name":"neurocore","version":"0.1.0"}}}
← {"result":{"protocolVersion":"2024-11-05","serverInfo":{...}}}
→ {"jsonrpc":"2.0","method":"notifications/initialized"}
→ {"jsonrpc":"2.0","id":2,"method":"tools/list"}
← {"result":{"tools":[{"name","description","inputSchema"}]}}
→ {"jsonrpc":"2.0","id":3,"method":"tools/call",
   "params":{"name":"...","arguments":{...}}}
← {"result":{"content":[{"type":"text","text":"..."}],"isError":false}}
```

Her mesaj **tek satır** JSON'dur (newline-delimited). Sunucu stderr'e
günlük yazar; stdout **yalnız** protokol içindir.

Tasarım kuralları
-----------------
```
· Bağlantı **tembel**: araç listesi istenene kadar süreç başlatılmaz
· Süreç ölürse bir kez yeniden denenir, sonra `hata` işaretlenir
· Tüm çağrılar kilitli (tek süreç, çok iş parçacığı güvenliği)
· Zaman aşımı: 20 sn (araç), 30 sn (başlatma)
```
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.mcp")

PROTOKOL = "2024-11-05"
VARSAYILAN_DOSYA = ".neurocore/mcp.json"
BASLATMA_SURESI = 30.0
ARAC_SURESI = 20.0

#: Hazır bağlantı şablonları (kullanıcı tek tıkla ekleyebilir)
PRESETLER: Dict[str, Dict[str, Any]] = {
    "blender": {
        "ad": "blender",
        "baslik": "Blender MCP",
        "komut": "uvx",
        "args": ["blender-mcp"],
        "aciklama": "Blender'ı MCP üzerinden sürer (nesne, malzeme, render)",
        "kaynak": "https://www.blender.org/lab/mcp-server/",
    },
    "godot": {
        "ad": "godot",
        "baslik": "Godot AI MCP",
        "komut": "uvx",
        "args": ["godot-ai-mcp"],
        "aciklama": "Godot editörünü MCP üzerinden sürer (sahne, düğüm, betik)",
        "kaynak": "https://godotengine.org/asset-library/asset/5050",
    },
}

#: Windows'ta konsol penceresi açılmasın
_FLAGS = 0
if os.name == "nt":
    _FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class MCPArac:
    """Sunucunun bildirdiği tek araç."""

    ad: str
    aciklama: str = ""
    sema: Dict[str, Any] = field(default_factory=dict)

    def arac_tanimi(self, onek: str) -> Dict[str, Any]:
        """OpenAI araç şemasına çevirir."""
        sema = self.sema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": f"{onek}.{self.ad}"[:64],
                "description": (self.aciklama or self.ad)[:280],
                "parameters": sema,
            },
        }


class MCPIstemci:
    """Tek MCP sunucusuyla stdio konuşması."""

    def __init__(self, ad: str, komut: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, cwd: str = "") -> None:
        self.ad = ad
        self.komut = komut
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd or ""
        self._surec: Optional[subprocess.Popen] = None
        self._kilit = threading.RLock()
        self._sayac = 0
        self._araclar: List[MCPArac] = []
        self.hata = ""
        self.baslatildi = False

    # ── süreç ──
    def baslat(self) -> bool:
        """Süreci başlatır ve el sıkışmayı yapar."""
        with self._kilit:
            if self._surec is not None and self._surec.poll() is None:
                return True
            ortam = dict(os.environ)
            ortam.update(self.env)
            try:
                self._surec = subprocess.Popen(
                    [self.komut] + self.args,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, env=ortam,
                    cwd=self.cwd or None, creationflags=_FLAGS,
                )
            except (OSError, ValueError) as exc:
                self.hata = f"başlatılamadı: {type(exc).__name__}: {exc}"[:200]
                logger.warning(f"MCP '{self.ad}' {self.hata}")
                return False
            try:
                yanit = self._istek("initialize", {
                    "protocolVersion": PROTOKOL,
                    "capabilities": {},
                    "clientInfo": {"name": "neurocore", "version": "0.1.0"},
                }, sure=BASLATMA_SURESI)
                if not isinstance(yanit, dict):
                    raise RuntimeError("initialize yanıtı boş")
                self._bildirim("notifications/initialized")
                self.baslatildi = True
                self.hata = ""
                sunucu = (yanit.get("serverInfo") or {}).get("name", self.ad)
                logger.info(f"MCP bağlandı: {self.ad} ({sunucu})")
                return True
            except Exception as exc:
                self.hata = f"el sıkışma başarısız: {type(exc).__name__}: {exc}"[:200]
                logger.warning(f"MCP '{self.ad}' {self.hata}")
                self.kapat()
                return False

    def kapat(self) -> None:
        """Süreci kapatır."""
        with self._kilit:
            if self._surec is not None:
                try:
                    self._surec.terminate()
                    self._surec.wait(timeout=3)
                except Exception:
                    try:
                        self._surec.kill()
                    except Exception:
                        pass
            self._surec = None
            self.baslatildi = False

    # ── JSON-RPC ──
    def _gonder(self, veri: Dict[str, Any]) -> None:
        if self._surec is None or self._surec.stdin is None:
            raise RuntimeError("MCP süreci yok")
        self._surec.stdin.write(json.dumps(veri, ensure_ascii=False) + "\n")
        self._surec.stdin.flush()

    def _bildirim(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        veri: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            veri["params"] = params
        self._gonder(veri)

    def _istek(self, method: str, params: Optional[Dict[str, Any]] = None,
               *, sure: float = ARAC_SURESI) -> Any:
        with self._kilit:
            if self._surec is None or self._surec.stdout is None:
                raise RuntimeError("MCP süreci yok")
            self._sayac += 1
            kimlik = self._sayac
            veri: Dict[str, Any] = {"jsonrpc": "2.0", "id": kimlik,
                                    "method": method}
            if params is not None:
                veri["params"] = params
            self._gonder(veri)

            bitis = time.time() + sure
            while time.time() < bitis:
                satir = self._surec.stdout.readline()
                if not satir:
                    if self._surec.poll() is not None:
                        raise RuntimeError("MCP süreci kapandı")
                    continue
                satir = satir.strip()
                if not satir:
                    continue
                try:
                    yanit = json.loads(satir)
                except json.JSONDecodeError:
                    continue          # sunucu günlük satırı yazmış olabilir
                if yanit.get("id") != kimlik:
                    continue          # başka isteğin yanıtı
                if "error" in yanit:
                    hata = yanit["error"]
                    raise RuntimeError(
                        f"{hata.get('code')}: {hata.get('message')}"[:200])
                return yanit.get("result")
            raise TimeoutError(f"{method} {sure:.0f} sn içinde yanıtlanmadı")

    # ── araçlar ──
    def araclari_yukle(self) -> List[MCPArac]:
        if not self.baslat():
            return []
        try:
            sonuc = self._istek("tools/list") or {}
        except Exception as exc:
            self.hata = f"tools/list: {type(exc).__name__}: {exc}"[:200]
            return []
        araclar: List[MCPArac] = []
        for ham in (sonuc.get("tools") or []):
            if not isinstance(ham, dict) or not ham.get("name"):
                continue
            araclar.append(MCPArac(
                ad=str(ham["name"])[:60],
                aciklama=str(ham.get("description") or "")[:280],
                sema=ham.get("inputSchema") or {},
            ))
        self._araclar = araclar
        return araclar

    def cagir(self, arac: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Araç çağırır; metin içeriğini birleştirip döndürür."""
        if not self.baslat():
            return {"ok": False, "error": self.hata or "bağlanamadı"}
        try:
            sonuc = self._istek("tools/call",
                                {"name": arac, "arguments": arguments or {}})
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
        parcalar: List[str] = []
        for içerik in (sonuc.get("content") or []) if isinstance(sonuc, dict) else []:
            if isinstance(içerik, dict) and içerik.get("type") == "text":
                parcalar.append(str(içerik.get("text") or ""))
        return {
            "ok": not bool(sonuc.get("isError")) if isinstance(sonuc, dict) else True,
            "metin": "\n".join(p for p in parcalar if p)[:6000],
            "ham": sonuc if isinstance(sonuc, dict) else {},
        }

    def durum(self) -> Dict[str, Any]:
        return {
            "ad": self.ad, "komut": self.komut, "args": self.args,
            "bagli": bool(self._surec is not None
                          and self._surec.poll() is None),
            "arac_sayisi": len(self._araclar),
            "araclar": [a.ad for a in self._araclar][:40],
            "hata": self.hata,
        }


class MCPYonetici:
    """Yapılandırılmış MCP sunucularının tümü."""

    def __init__(self, dosya: Optional[Path] = None) -> None:
        self.dosya = Path(dosya) if dosya else Path(VARSAYILAN_DOSYA)
        self._istemciler: Dict[str, MCPIstemci] = {}
        self._yukle()

    # ── yapılandırma ──
    def _yukle(self) -> None:
        try:
            veri = json.loads(self.dosya.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            veri = {}
        for ad, ham in (veri.get("sunucular") or {}).items():
            if not isinstance(ham, dict) or not ham.get("komut"):
                continue
            self._istemciler[ad] = MCPIstemci(
                ad=ad, komut=str(ham["komut"]),
                args=list(ham.get("args") or []),
                env=dict(ham.get("env") or {}),
                cwd=str(ham.get("cwd") or ""),
            )

    def _yaz(self) -> None:
        veri = {"sunucular": {
            ad: {"komut": i.komut, "args": i.args, "env": i.env, "cwd": i.cwd}
            for ad, i in self._istemciler.items()
        }}
        try:
            self.dosya.parent.mkdir(parents=True, exist_ok=True)
            self.dosya.write_text(json.dumps(veri, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        except OSError as exc:
            logger.warning(f"MCP ayarları yazılamadı: {exc}")

    def ekle(self, ad: str, komut: str, args: Optional[List[str]] = None,
             env: Optional[Dict[str, str]] = None, cwd: str = "") -> Dict[str, Any]:
        ad = str(ad or "").strip().lower()
        if not ad or not str(komut or "").strip():
            return {"ok": False, "error": "ad ve komut gerekli"}
        eski = self._istemciler.pop(ad, None)
        if eski is not None:
            eski.kapat()
        self._istemciler[ad] = MCPIstemci(ad=ad, komut=str(komut).strip(),
                                          args=args or [], env=env or {},
                                          cwd=cwd)
        self._yaz()
        return {"ok": True, "ad": ad}

    def sil(self, ad: str) -> Dict[str, Any]:
        istemci = self._istemciler.pop(str(ad or ""), None)
        if istemci is None:
            return {"ok": False, "error": "sunucu yok"}
        istemci.kapat()
        self._yaz()
        return {"ok": True}

    def dene(self, ad: str) -> Dict[str, Any]:
        """Bağlantıyı sınar ve araç listesini döndürür."""
        istemci = self._istemciler.get(str(ad or ""))
        if istemci is None:
            return {"ok": False, "error": "sunucu yok"}
        araclar = istemci.araclari_yukle()
        return {
            "ok": bool(istemci.baslatildi),
            "ad": ad,
            "arac_sayisi": len(araclar),
            "araclar": [a.ad for a in araclar][:40],
            "hata": istemci.hata,
        }

    # ── araç kaydı ──
    def araclar(self) -> List[Dict[str, Any]]:
        """Tüm sunucuların araçlarını OpenAI şemasıyla döndürür."""
        sonuc: List[Dict[str, Any]] = []
        for ad, istemci in self._istemciler.items():
            if not istemci._araclar:
                istemci.araclari_yukle()
            for arac in istemci._araclar:
                sonuc.append(arac.arac_tanimi(f"mcp.{ad}"))
        return sonuc

    def cagir(self, tam_ad: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """`mcp.<sunucu>.<araç>` biçimindeki adı çözer."""
        parcalar = str(tam_ad or "").split(".")
        if len(parcalar) < 3 or parcalar[0] != "mcp":
            return {"ok": False, "error": "geçersiz MCP araç adı"}
        _, sunucu, *kalan = parcalar
        istemci = self._istemciler.get(sunucu)
        if istemci is None:
            return {"ok": False, "error": f"MCP sunucusu yok: {sunucu}"}
        return istemci.cagir(".".join(kalan), arguments)

    def durum(self) -> Dict[str, Any]:
        return {
            "sunucu_sayisi": len(self._istemciler),
            "sunucular": [i.durum() for i in self._istemciler.values()],
            "presetler": PRESETLER,
            "dosya": str(self.dosya),
        }

    def kapat(self) -> None:
        for istemci in self._istemciler.values():
            istemci.kapat()
