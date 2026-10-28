"""
NeuroCore Serileştirme Yardımcıları
====================================
JSON, YAML, MessagePack ve Pickle desteği ile güvenli serileştirme.
Sıkıştırma (gzip, lz4) ve tür-koruyucu dönüşüm desteği.
"""

from __future__ import annotations

import gzip
import io
import json
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml


# ─── Format Tanımları ──────────────────────────────────────────────────────

class SerializationFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    MSGPACK = "msgpack"
    PICKLE = "pickle"


class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


# ─── Özel JSON Encoder ─────────────────────────────────────────────────────

class NeuroCoreEncoder(json.JSONEncoder):
    """NumPy, datetime ve özel tipleri destekleyen JSON encoder."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        elif isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}
        elif isinstance(obj, Path):
            return {"__type__": "path", "value": str(obj)}
        elif isinstance(obj, bytes):
            return {"__type__": "bytes", "value": obj.hex()}
        elif isinstance(obj, Enum):
            return {"__type__": "enum", "class": type(obj).__name__, "value": obj.value}
        return super().default(obj)


def _neurocore_decoder(obj: dict) -> Any:
    """Özel tipleri geri yükleyen JSON decoder hook."""
    if "__type__" not in obj:
        return obj
    
    type_name = obj["__type__"]
    if type_name == "ndarray":
        return np.array(obj["data"], dtype=obj.get("dtype", "float32"))
    elif type_name == "datetime":
        return datetime.fromisoformat(obj["value"])
    elif type_name == "set":
        return set(obj["data"])
    elif type_name == "path":
        return Path(obj["value"])
    elif type_name == "bytes":
        return bytes.fromhex(obj["value"])
    return obj


# ─── Ana Serileştirici ─────────────────────────────────────────────────────

class Serializer:
    """
    Çok-formatlı serileştirme yardımcısı.
    
    Desteklenen formatlar: JSON, YAML, MessagePack, Pickle
    Desteklenen sıkıştırma: None, GZIP, LZ4
    
    Kullanım:
        data = {"model": "neurocore", "weights": np.array([1, 2, 3])}
        
        # JSON olarak kaydet
        Serializer.save(data, "config.json")
        loaded = Serializer.load("config.json")
        
        # Sıkıştırılmış kaydet
        Serializer.save(data, "data.json.gz", compression=CompressionType.GZIP)
    """
    
    @staticmethod
    def serialize(data: Any, fmt: SerializationFormat = SerializationFormat.JSON) -> bytes:
        """Veriyi byte dizisine serileştir."""
        if fmt == SerializationFormat.JSON:
            return json.dumps(data, cls=NeuroCoreEncoder, ensure_ascii=False, indent=2).encode("utf-8")
        elif fmt == SerializationFormat.YAML:
            return yaml.dump(data, allow_unicode=True, default_flow_style=False).encode("utf-8")
        elif fmt == SerializationFormat.MSGPACK:
            import msgpack
            return msgpack.packb(data, default=str)
        elif fmt == SerializationFormat.PICKLE:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Desteklenmeyen format: {fmt}")

    @staticmethod
    def deserialize(data: bytes, fmt: SerializationFormat = SerializationFormat.JSON) -> Any:
        """Byte dizisinden veri oku."""
        if fmt == SerializationFormat.JSON:
            return json.loads(data.decode("utf-8"), object_hook=_neurocore_decoder)
        elif fmt == SerializationFormat.YAML:
            return yaml.safe_load(data.decode("utf-8"))
        elif fmt == SerializationFormat.MSGPACK:
            import msgpack
            return msgpack.unpackb(data, raw=False)
        elif fmt == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            raise ValueError(f"Desteklenmeyen format: {fmt}")

    @staticmethod
    def compress(data: bytes, compression: CompressionType = CompressionType.GZIP) -> bytes:
        """Veriyi sıkıştır."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.compress(data)
        raise ValueError(f"Desteklenmeyen sıkıştırma: {compression}")

    @staticmethod
    def decompress(data: bytes, compression: CompressionType = CompressionType.GZIP) -> bytes:
        """Sıkıştırılmış veriyi aç."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.decompress(data)
        raise ValueError(f"Desteklenmeyen sıkıştırma: {compression}")

    @classmethod
    def save(cls, data: Any, path: Union[str, Path],
             fmt: Optional[SerializationFormat] = None,
             compression: CompressionType = CompressionType.NONE) -> Path:
        """
        Veriyi dosyaya kaydet.
        
        Format uzantıdan otomatik algılanır: .json, .yaml/.yml, .msgpack, .pkl
        Sıkıştırma uzantıdan algılanır: .gz, .lz4
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format otomatik algılama
        if fmt is None:
            fmt = cls._detect_format(path)
        
        # Sıkıştırma otomatik algılama
        if compression == CompressionType.NONE:
            if path.suffix == ".gz":
                compression = CompressionType.GZIP
            elif path.suffix == ".lz4":
                compression = CompressionType.LZ4
        
        serialized = cls.serialize(data, fmt)
        
        if compression != CompressionType.NONE:
            serialized = cls.compress(serialized, compression)
        
        path.write_bytes(serialized)
        return path

    @classmethod
    def load(cls, path: Union[str, Path],
             fmt: Optional[SerializationFormat] = None,
             compression: CompressionType = CompressionType.NONE) -> Any:
        """Dosyadan veri yükle."""
        path = Path(path)
        
        if fmt is None:
            fmt = cls._detect_format(path)
        
        if compression == CompressionType.NONE:
            if path.suffix == ".gz":
                compression = CompressionType.GZIP
            elif path.suffix == ".lz4":
                compression = CompressionType.LZ4
        
        data = path.read_bytes()
        
        if compression != CompressionType.NONE:
            data = cls.decompress(data, compression)
        
        return cls.deserialize(data, fmt)

    @staticmethod
    def _detect_format(path: Path) -> SerializationFormat:
        """Dosya uzantısından formatı algıla."""
        # Sıkıştırma uzantısını atla
        name = path.name
        for comp_ext in (".gz", ".lz4"):
            if name.endswith(comp_ext):
                name = name[:-len(comp_ext)]
                break
        
        ext = Path(name).suffix.lower()
        format_map = {
            ".json": SerializationFormat.JSON,
            ".yaml": SerializationFormat.YAML,
            ".yml": SerializationFormat.YAML,
            ".msgpack": SerializationFormat.MSGPACK,
            ".pkl": SerializationFormat.PICKLE,
            ".pickle": SerializationFormat.PICKLE,
        }
        
        return format_map.get(ext, SerializationFormat.JSON)
