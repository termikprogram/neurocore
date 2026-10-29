"""
Sahte MCP sunucusu — istemciyi uçtan uca sınamak için
=====================================================
Gerçek bir MCP sunucusu (Blender/Godot) kurmak ağır; bu betik protokolü
birebir taklit eder: `initialize` → `notifications/initialized` →
`tools/list` → `tools/call`.

Kullanım:
    python tools/mcp_sahte.py            # stdio üzerinde konuşur

Araçlar:
    echo(metin)      → metni geri döndürür
    topla(a, b)      → sayıları toplar
    patlat()         → bilerek hata döndürür (hata yolu testi)
"""

from __future__ import annotations

import json
import sys

ARACLAR = [
    {
        "name": "echo",
        "description": "Verilen metni aynen geri döndürür.",
        "inputSchema": {
            "type": "object",
            "properties": {"metin": {"type": "string"}},
            "required": ["metin"],
        },
    },
    {
        "name": "topla",
        "description": "İki sayıyı toplar.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "patlat",
        "description": "Her zaman hata döndürür (test).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def yanit(kimlik, sonuc=None, hata=None) -> None:
    veri = {"jsonrpc": "2.0", "id": kimlik}
    if hata is not None:
        veri["error"] = hata
    else:
        veri["result"] = sonuc
    sys.stdout.write(json.dumps(veri, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    for satir in sys.stdin:
        satir = satir.strip()
        if not satir:
            continue
        try:
            istek = json.loads(satir)
        except json.JSONDecodeError:
            continue
        method = istek.get("method", "")
        kimlik = istek.get("id")
        params = istek.get("params") or {}

        if method == "initialize":
            yanit(kimlik, {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sahte-mcp", "version": "0.1"},
            })
        elif method == "tools/list":
            yanit(kimlik, {"tools": ARACLAR})
        elif method == "tools/call":
            ad = params.get("name")
            args = params.get("arguments") or {}
            if ad == "echo":
                metin = str(args.get("metin", ""))
                yanit(kimlik, {"content": [{"type": "text",
                                            "text": f"echo: {metin}"}],
                               "isError": False})
            elif ad == "topla":
                try:
                    toplam = float(args.get("a", 0)) + float(args.get("b", 0))
                except (TypeError, ValueError):
                    yanit(kimlik, {"content": [{"type": "text",
                                                "text": "sayı değil"}],
                                   "isError": True})
                    continue
                yanit(kimlik, {"content": [{"type": "text", "text": str(toplam)}],
                               "isError": False})
            elif ad == "patlat":
                yanit(kimlik, {"content": [{"type": "text",
                                            "text": "bilerek patladı"}],
                               "isError": True})
            else:
                yanit(kimlik, None, {"code": -32601,
                                     "message": f"bilinmeyen araç: {ad}"})
        elif method.startswith("notifications/"):
            continue                     # bildirime yanıt verilmez
        else:
            yanit(kimlik, None, {"code": -32601,
                                 "message": f"bilinmeyen yöntem: {method}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
