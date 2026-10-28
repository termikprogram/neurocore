"""
NeuroCore Tablo Kodlayıcı (L4)
===============================
CSV / TSV / Markdown / JSON / Excel tablolarını ayrıştıran, profillenen ve
deterministik bir gömleme vektörüne çeviren kodlayıcı.

Ayrıştırma:
    - `csv.Sniffer` ile ayırıcı sezimi (virgül / noktalı virgül / sekme / boru)
    - Markdown tablo (`| a | b |` + `|---|---|`)
    - JSON: `list[dict]`, `list[list]`, `dict-of-lists`
    - xlsx → `openpyxl` (veya `pandas`) varsa; yoksa Türkçe mesajlı `TableDecodeError`

Özellik vektörü (ham/native boyut = 320):
    çıkarılan tip histogramı                     →   8
    16 sütun × 6 sayısal özet                    →  96
    16 sütun × 5 şekil ölçütü                    →  80
    16 sütun × 8 başlık hash'i                   → 128
    global ölçütler                              →   8

Ardından sabit tohumlu projeksiyon (`seed=4242`, `salt="neurocore-table-proj"`)
ile `embedding_dim` boyutuna taşınır ve L2-normalize edilir.

Örnek:
    enc = TableEncoder()
    enc.initialize(config, bus)
    print(enc.profile("a,b\\n1,2\\n3,4")["columns"])
    print(enc.answer_hint("a,b\\n1,2", "a toplamı nedir?"))
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    EncodedChunk,
    Modality,
    TableData,
    TableDecodeError,
    attach_native,
    l2_normalize,
    projection_matrix,
    stable_hash_vector,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.table")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Projeksiyon tohumu ve tuzu (deterministik — değiştirilmemeli).
TABLE_PROJ_SEED = 4242
TABLE_PROJ_SALT = "neurocore-table-proj"

#: Başlık hash'i için tuz.
TABLE_HEADER_SALT = "neurocore-table-hdr"

#: Özellik vektöründe dikkate alınan en büyük sütun sayısı.
MAX_COLS = 16

#: Sütun başına başlık hash boyutu.
HEADER_HASH_DIM = 8

#: Çıkarılan tip kovaları (sabit sıra — özellik vektörü buna bağlıdır).
TYPE_BUCKETS: Tuple[str, ...] = (
    "int", "float", "bool", "date", "str", "category", "id", "empty",
)

#: Ham özellik boyutu: 8 + 16*6 + 16*5 + 16*8 + 8.
NATIVE_FEATURE_DIM = 8 + MAX_COLS * 6 + MAX_COLS * 5 + MAX_COLS * HEADER_HASH_DIM + 8

#: Ayrıştırmada işlenecek en büyük satır sayısı.
MAX_ROWS = 20000

#: Markdown tablo ayırıcı hücresi.
_MD_SEPARATOR_CELL_RE = re.compile(r"^:?-{1,}:?$")

_INT_RE = re.compile(r"^[+-]?\d{1,18}$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][+-]?\d+)?$")
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$"
    r"|^\d{1,2}[./]\d{1,2}[./]\d{2,4}$"
    r"|^\d{4}[./]\d{1,2}[./]\d{1,2}$"
)
_BOOL_VALUES = {"true", "false", "yes", "no", "evet", "hayır", "hayir",
                "doğru", "yanlış", "yanlis", "var", "yok"}
_ID_NAME_HINTS = ("id", "no", "kod", "code", "key", "uuid", "ref")

#: Türkçe karakter katlama tablosu (soru/sütun eşleştirmesi için).
_FOLD_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "İ": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "Ö": "o", "Ş": "s", "Ü": "u", "â": "a", "î": "i", "û": "u",
})


# ─── Ana Kodlayıcı Sınıfı ──────────────────────────────────────────────────

class TableEncoder:
    """
    Hafif, deterministik tablo kodlayıcı (istatistiksel özellikler + projeksiyon).

    Ağır bağımlılık (pandas) gerekmez; yalnızca xlsx için `openpyxl` opsiyoneldir.
    """

    name = "multimodal.table"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._openpyxl_available = False
        self._pandas_available = False
        self._proj_cache: Dict[Tuple[int, int], "np.ndarray"] = {}
        self._stats: Dict[str, Any] = {
            "parsed": 0,
            "encoded": 0,
            "errors": 0,
            "rows_processed": 0,
            "cols_processed": 0,
            "profiles": 0,
            "markdown_exports": 0,
            "answer_hints": 0,
        }
        self._format_counts: Dict[str, int] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Kodlayıcıyı yapılandırır (idempotent).

        Args:
            config: `SystemConfig` (`config.neural.embedding_dim` kullanılır)
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        self._openpyxl_available = self._probe_module("openpyxl")
        self._pandas_available = self._probe_module("pandas")
        self._initialized = True
        logger.info("Tablo kodlayıcı hazır (dim=%d, openpyxl=%s, pandas=%s)",
                    self._embedding_dim, self._openpyxl_available,
                    self._pandas_available)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.table_encoder_ready",
                {"dim": self._embedding_dim, "openpyxl": self._openpyxl_available},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        if not self._initialized:
            return
        self._proj_cache.clear()
        self._initialized = False
        logger.info("Tablo kodlayıcı kapatıldı")

    # ─── Ayrıştırma ─────────────────────────────────────────────────────

    def parse(self, table: Any) -> TableData:
        """
        Herhangi bir tablo girdisini `TableData`'ya çevirir.

        Args:
            table: csv/tsv/markdown/json metni | yol | list[dict] | list[list]
                | dict-of-lists | pandas DataFrame | TableData

        Returns:
            Ayrıştırılmış `TableData`

        Raises:
            TableDecodeError: Girdi ayrıştırılamazsa (Türkçe mesajla)
        """
        if isinstance(table, TableData):
            return table
        if table is None:
            raise TableDecodeError("Tablo verisi boş (None)")

        try:
            parsed = self._parse_inner(table)
        except TableDecodeError:
            self._stats["errors"] += 1
            raise
        except Exception as exc:
            self._stats["errors"] += 1
            raise TableDecodeError(f"Tablo ayrıştırılamadı: {exc}") from exc

        # Çıkarılan tipleri hesapla (profil ve özellik vektörü bunu kullanır)
        parsed.dtypes = {
            header: self._infer_column_type(
                header,
                [row[index] for row in parsed.rows] if parsed.rows else [],
            )
            for index, header in enumerate(parsed.headers)
        }
        self._stats["parsed"] += 1
        self._stats["rows_processed"] += parsed.n_rows
        self._stats["cols_processed"] += parsed.n_cols
        self._format_counts[parsed.source] = self._format_counts.get(parsed.source, 0) + 1
        return parsed

    def _parse_inner(self, table: Any) -> TableData:
        """`parse()` gövdesi (sayaç/istisna sarmalayıcısı olmadan)."""
        # ── Metin: yol mu, içerik mi? ───────────────────────────────────
        if isinstance(table, str):
            stripped = table.strip()
            if not stripped:
                raise TableDecodeError("Tablo metni boş")
            if len(stripped) < 4096 and "\n" not in stripped and Path(stripped).exists():
                return self._parse_file(Path(stripped))
            if self._looks_like_markdown(stripped):
                return self.parse_markdown(stripped)
            if stripped[0] in "[{":
                try:
                    return self.parse_json(stripped)
                except TableDecodeError:
                    logger.debug("JSON ayrıştırılamadı, CSV olarak denenecek")
            return self.parse_csv(stripped)

        # ── Baytlar ─────────────────────────────────────────────────────
        if isinstance(table, (bytes, bytearray, memoryview)):
            data = bytes(table)
            if not data:
                raise TableDecodeError("Tablo bayt dizisi boş")
            if data.startswith(b"PK\x03\x04"):
                return self._parse_xlsx_bytes(data)
            return self.parse_csv(self._decode_text(data))

        # ── list[dict] / list[list] / dict-of-lists ─────────────────────
        if isinstance(table, (list, tuple)):
            return self.parse_json(list(table))
        if isinstance(table, dict):
            return self.parse_json(table)

        # ── pandas / openpyxl benzeri nesneler ──────────────────────────
        if self._is_dataframe_like(table):
            return self._parse_dataframe(table)

        raise TableDecodeError(
            f"Desteklenmeyen tablo tipi: {type(table).__name__}")

    def _parse_file(self, path: Path) -> TableData:
        """Dosya uzantısına göre uygun ayrıştırıcıya yönlendirir."""
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xlsm"):
            return self._parse_xlsx(path)
        if suffix == ".xls":
            raise TableDecodeError(
                "Eski .xls biçimi desteklenmiyor — dosyayı .xlsx veya .csv "
                "olarak kaydedin")
        if suffix in (".json", ".jsonl", ".ndjson"):
            return self.parse_json(self._read_text(path))
        if suffix in (".md", ".markdown"):
            content = self._read_text(path)
            if self._looks_like_markdown(content):
                return self.parse_markdown(content)
            return self.parse_csv(content)
        if suffix == ".tsv":
            return self.parse_csv(self._read_text(path), delimiter="\t")
        content = self._read_text(path)
        if self._looks_like_markdown(content):
            return self.parse_markdown(content)
        return self.parse_csv(content)

    def parse_csv(self, text: str, delimiter: Optional[str] = None) -> TableData:
        """
        CSV/TSV metnini ayrıştırır.

        Args:
            text: CSV/TSV içeriği
            delimiter: Ayırıcı (None → `csv.Sniffer`, başarısızsa en sık ayırıcı)

        Returns:
            `TableData` (source="csv")

        Raises:
            TableDecodeError: Metin ayrıştırılamazsa
        """
        if text is None or not str(text).strip():
            raise TableDecodeError("CSV metni boş")
        content = str(text).replace("\r\n", "\n").replace("\r", "\n")

        chosen = delimiter or self._sniff_delimiter(content)
        try:
            reader = csv.reader(io.StringIO(content), delimiter=chosen)
            raw_rows = []
            truncated = False
            for index, row in enumerate(reader):
                if index >= MAX_ROWS:
                    truncated = True
                    break
                raw_rows.append(row)
        except Exception as exc:
            raise TableDecodeError(f"CSV ayrıştırılamadı: {exc}") from exc

        raw_rows = [row for row in raw_rows if any(str(cell).strip() for cell in row)]
        if not raw_rows:
            raise TableDecodeError("CSV içinde veri satırı bulunamadı")

        cells = [[self._normalize_cell(cell) for cell in row] for row in raw_rows]
        if self._row_is_header(cells[0]):
            headers = [str(cell) if cell is not None else f"col_{i}"
                       for i, cell in enumerate(cells[0])]
            body = cells[1:]
        else:
            headers = [f"col_{i}" for i in range(len(cells[0]))]
            body = cells

        table = TableData.from_rows(headers, body, source="csv")
        table.dtypes = {}  # parse() içinde yeniden çıkarılacak
        if truncated:
            table.source = "csv:truncated"
        return table

    def parse_markdown(self, text: str) -> TableData:
        """
        Markdown tablosunu ayrıştırır.

        Args:
            text: Markdown içeriği (tablo satırları `|` ile ayrılmış)

        Returns:
            `TableData` (source="markdown")

        Raises:
            TableDecodeError: Tablo bulunamazsa
        """
        if not text or "|" not in str(text):
            raise TableDecodeError("Markdown tablosu bulunamadı ('|' yok)")

        lines = [line.strip() for line in str(text).splitlines()]
        separator_index = -1
        for index in range(len(lines) - 1):
            if "|" not in lines[index] or "|" not in lines[index + 1]:
                continue
            if self._is_separator_row(lines[index + 1]):
                separator_index = index + 1
                break
        if separator_index < 1:
            raise TableDecodeError("Markdown tablo ayırıcı satırı (|---|) bulunamadı")

        headers = self._split_markdown_row(lines[separator_index - 1])
        rows: List[List[Any]] = []
        for line in lines[separator_index + 1:]:
            if "|" not in line:
                if rows:
                    break
                continue
            cells = self._split_markdown_row(line)
            if not any(str(cell).strip() for cell in cells):
                continue
            rows.append([self._normalize_cell(cell) for cell in cells])

        table = TableData.from_rows(headers, rows, source="markdown")
        table.dtypes = {}
        return table

    def parse_json(self, text_or_obj: Any) -> TableData:
        """
        JSON (veya eşdeğeri Python yapısı) tablosunu ayrıştırır.

        Desteklenen biçimler:
            - `[{"a": 1}, {"a": 2}]` → liste-sözlük
            - `[[1, 2], [3, 4]]` → liste-liste
            - `{"a": [1, 2], "b": [3, 4]}` → sözlük-liste
            - `{"a": 1, "b": 2}` → tek satır

        Args:
            text_or_obj: JSON metni veya Python nesnesi

        Returns:
            `TableData` (source="json")

        Raises:
            TableDecodeError: Yapı tabloya çevrilemezse
        """
        data = text_or_obj
        if isinstance(data, (str, bytes, bytearray)):
            raw = data.decode("utf-8", errors="replace") \
                if isinstance(data, (bytes, bytearray)) else data
            stripped = raw.strip()
            if not stripped:
                raise TableDecodeError("JSON metni boş")
            # JSONL desteği
            if "\n" in stripped and not stripped.startswith("["):
                lines = [line for line in stripped.splitlines() if line.strip()]
                if all(line.strip().startswith("{") for line in lines):
                    try:
                        data = [json.loads(line) for line in lines]
                    except ValueError as exc:
                        raise TableDecodeError(f"JSONL ayrıştırılamadı: {exc}") from exc
                else:
                    data = self._json_loads(stripped)
            else:
                data = self._json_loads(stripped)

        if isinstance(data, TableData):
            return data

        # ── list[dict] ──────────────────────────────────────────────────
        if isinstance(data, list) and data and all(isinstance(i, dict) for i in data):
            headers: List[str] = []
            for item in data:
                for key in item.keys():
                    if str(key) not in headers:
                        headers.append(str(key))
            rows = [[self._normalize_cell(item.get(key)) for key in headers]
                    for item in data]
            table = TableData.from_rows(headers, rows, source="json")
            table.dtypes = {}
            return table

        # ── list[list] ──────────────────────────────────────────────────
        if isinstance(data, list) and data and all(
                isinstance(i, (list, tuple)) for i in data):
            rows_raw = [[self._normalize_cell(cell) for cell in row] for row in data]
            if self._row_is_header([[str(c) if c is not None else "" for c in rows_raw[0]]][0]):
                headers = [str(c) if c is not None else f"col_{i}"
                           for i, c in enumerate(rows_raw[0])]
                body = rows_raw[1:]
            else:
                headers = [f"col_{i}" for i in range(len(rows_raw[0]))]
                body = rows_raw
            table = TableData.from_rows(headers, body, source="json")
            table.dtypes = {}
            return table

        # ── list[scalar] ────────────────────────────────────────────────
        if isinstance(data, list):
            rows = [[self._normalize_cell(item)] for item in data]
            table = TableData.from_rows(["value"], rows, source="json")
            table.dtypes = {}
            return table

        # ── dict-of-lists / dict-of-dicts / dict-skaler ─────────────────
        if isinstance(data, dict) and data:
            values = list(data.values())
            if all(isinstance(v, dict) for v in values):
                # {"satir1": {...}, "satir2": {...}} → satır sözlükleri
                headers = []
                for item in values:
                    for key in item.keys():
                        if str(key) not in headers:
                            headers.append(str(key))
                rows = [[self._normalize_cell(item.get(key)) for key in headers]
                        for item in values]
                table = TableData.from_rows(headers, rows, source="json")
                table.dtypes = {}
                return table
            if all(isinstance(v, (list, tuple)) for v in values):
                headers = [str(k) for k in data.keys()]
                columns = [list(v) for v in values]
                width = max((len(c) for c in columns), default=0)
                rows = [
                    [self._normalize_cell(columns[c][r]) if r < len(columns[c]) else None
                     for c in range(len(columns))]
                    for r in range(width)
                ]
                table = TableData.from_rows(headers, rows, source="json")
                table.dtypes = {}
                return table
            rows = [[self._normalize_cell(v) for v in values]]
            table = TableData.from_rows([str(k) for k in data.keys()], rows, source="json")
            table.dtypes = {}
            return table

        raise TableDecodeError(
            f"JSON yapısı tabloya çevrilemedi: {type(data).__name__}")

    def _parse_dataframe(self, frame: Any) -> TableData:
        """pandas DataFrame benzeri nesneyi `TableData`'ya çevirir."""
        try:
            headers = [str(column) for column in list(frame.columns)]
            rows = [[self._normalize_cell(value) for value in row]
                    for row in frame.values.tolist()]
        except Exception as exc:
            raise TableDecodeError(f"DataFrame tabloya çevrilemedi: {exc}") from exc
        table = TableData.from_rows(headers, rows, source="pandas")
        table.dtypes = {}
        return table

    def _parse_xlsx(self, path: Path) -> TableData:
        """xlsx dosyasını `openpyxl` (yoksa `pandas`) ile okur."""
        data = None
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            raise TableDecodeError(f"xlsx dosyası okunamadı ({path}): {exc}") from exc
        return self._parse_xlsx_bytes(data)

    def _parse_xlsx_bytes(self, data: bytes) -> TableData:
        """
        xlsx baytlarını `openpyxl` ile ayrıştırır (ilk çalışma sayfası).

        Raises:
            TableDecodeError: openpyxl/pandas yoksa veya dosya bozuksa
        """
        if self._openpyxl_available:
            try:
                import openpyxl  # opsiyonel import — fonksiyon içinde

                workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True,
                                                  data_only=True)
                sheet = workbook.worksheets[0]
                rows: List[List[Any]] = []
                for index, row in enumerate(sheet.iter_rows(values_only=True)):
                    if index >= MAX_ROWS:
                        break
                    rows.append([self._normalize_cell(cell) for cell in row])
                workbook.close()
                rows = [row for row in rows if any(cell is not None for cell in row)]
                if not rows:
                    raise TableDecodeError("xlsx çalışma sayfası boş")
                if self._row_is_header([[str(c) if c is not None else "" for c in rows[0]]][0]):
                    headers = [str(c) if c is not None else f"col_{i}"
                               for i, c in enumerate(rows[0])]
                    body = rows[1:]
                else:
                    headers = [f"col_{i}" for i in range(len(rows[0]))]
                    body = rows
                table = TableData.from_rows(headers, body, source="xlsx")
                table.dtypes = {}
                return table
            except TableDecodeError:
                raise
            except Exception as exc:
                logger.warning("openpyxl ile okunamadı, pandas denenecek: %s", exc)

        if self._pandas_available:
            try:
                import pandas  # opsiyonel import — fonksiyon içinde

                frame = pandas.read_excel(io.BytesIO(data))
                return self._parse_dataframe(frame)
            except Exception as exc:
                raise TableDecodeError(f"xlsx pandas ile okunamadı: {exc}") from exc

        raise TableDecodeError(
            "xlsx okunamıyor: 'openpyxl' veya 'pandas' paketi kurulu değil "
            "(pip install openpyxl)")

    # ─── Profil Çıkarma ─────────────────────────────────────────────────

    def profile(self, table: Any) -> dict:
        """
        Tablonun sütun bazlı istatistiksel profilini çıkarır.

        Args:
            table: Herhangi bir tablo girdisi

        Returns:
            {
              "n_rows", "n_cols", "source", "columns": [
                 {"name","index","inferred_type","null_ratio","unique_ratio",
                  "n_unique","n_null","min","max","mean","std",
                  "top_values":[{"value","count","ratio"}],"samples":[...]}
              ],
              "numeric_columns", "categorical_columns", "id_columns", "date_columns"
            }
        """
        parsed = self.parse(table)
        self._stats["profiles"] += 1
        columns: List[Dict[str, Any]] = []
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        id_columns: List[str] = []
        date_columns: List[str] = []

        for index, header in enumerate(parsed.headers):
            values = [row[index] for row in parsed.rows]
            inferred = self._infer_column_type(header, values)
            total = len(values)
            non_null = [v for v in values if v is not None and str(v).strip() != ""]
            null_ratio = 1.0 - (len(non_null) / total) if total else 0.0
            counts = self._value_counts(non_null)
            n_unique = len(counts)
            unique_ratio = (n_unique / len(non_null)) if non_null else 0.0

            entry: Dict[str, Any] = {
                "name": header,
                "index": index,
                "inferred_type": inferred,
                "null_ratio": round(null_ratio, 6),
                "unique_ratio": round(unique_ratio, 6),
                "n_unique": n_unique,
                "n_null": total - len(non_null),
                "samples": [str(v) for v in non_null[:3]],
            }

            if inferred in ("int", "float", "id") or self._is_numeric_column(non_null):
                numeric_values = [v for v in (self._to_float(item) for item in non_null)
                                  if v is not None]
                if numeric_values:
                    array = np.asarray(numeric_values, dtype=np.float64)
                    entry.update({
                        "min": round(float(array.min()), 6),
                        "max": round(float(array.max()), 6),
                        "mean": round(float(array.mean()), 6),
                        "std": round(float(array.std()), 6),
                    })
                    numeric_columns.append(header)
            if inferred in ("str", "category", "bool") and not entry.get("mean"):
                entry["top_values"] = [
                    {"value": value, "count": count, "ratio": round(count / max(1, len(non_null)), 4)}
                    for value, count in sorted(counts.items(),
                                               key=lambda kv: (-kv[1], kv[0]))[:5]
                ]
                categorical_columns.append(header)
            if inferred == "id":
                id_columns.append(header)
            if inferred == "date":
                date_columns.append(header)
            columns.append(entry)

        return {
            "n_rows": parsed.n_rows,
            "n_cols": parsed.n_cols,
            "source": parsed.source,
            "headers": list(parsed.headers),
            "columns": columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "id_columns": id_columns,
            "date_columns": date_columns,
        }

    def to_markdown(self, table: Any, max_rows: int = 50) -> str:
        """
        Tabloyu markdown biçimine çevirir (kasadaki notlar için).

        Args:
            table: Herhangi bir tablo girdisi
            max_rows: Yazılacak en büyük satır sayısı

        Returns:
            Markdown tablo metni
        """
        parsed = self.parse(table)
        self._stats["markdown_exports"] += 1
        limit = max(0, int(max_rows))
        shown = parsed.rows[:limit]
        headers = [self._escape_markdown_cell(h) for h in parsed.headers]
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join([" --- "] * max(1, len(headers))) + "|"]
        for row in shown:
            cells = [self._escape_markdown_cell("" if cell is None else cell)
                     for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
        if parsed.n_rows > len(shown):
            lines.append("")
            lines.append(f"*({parsed.n_rows - len(shown)} satır daha var — "
                         f"toplam {parsed.n_rows} satır)*")
        return "\n".join(lines)

    def answer_hint(self, table: Any, question: str) -> dict:
        """
        Sorudaki sütun adlarını bulup kaba toplulaştırmalar üretir (naif yardımcı).

        Sayısal sütunlar için count/sum/mean/min/max/std, diğerleri için en sık
        görülen 5 değer hesaplanır.

        Args:
            table: Herhangi bir tablo girdisi
            question: Doğal dil sorusu

        Returns:
            {"question", "matched_columns", "aggregates", "intent", "n_rows",
             "n_cols", "note"}
        """
        parsed = self.parse(table)
        profile = self.profile(parsed)
        self._stats["answer_hints"] += 1

        folded_question = self._fold(str(question or ""))
        matched: List[str] = []
        for column in profile["columns"]:
            name = column["name"]
            folded_name = self._fold(name)
            if not folded_name:
                continue
            if folded_name in folded_question:
                matched.append(name)
                continue
            # Çok kelimeli başlıklarda kelime bazlı kısmi eşleşme
            tokens = [t for t in re.split(r"[\s_\-]+", folded_name) if len(t) >= 4]
            if tokens and all(token in folded_question for token in tokens):
                matched.append(name)

        aggregates: Dict[str, Any] = {}
        for name in matched:
            index = parsed.headers.index(name)
            values = [row[index] for row in parsed.rows]
            non_null = [v for v in values if v is not None and str(v).strip() != ""]
            numeric_values = [v for v in (self._to_float(item) for item in non_null)
                              if v is not None]
            entry: Dict[str, Any] = {
                "count": len(non_null),
                "null_count": len(values) - len(non_null),
                "inferred_type": parsed.dtypes.get(name, "str"),
            }
            if numeric_values and (parsed.dtypes.get(name) in ("int", "float", "id")
                                   or len(numeric_values) == len(non_null)):
                array = np.asarray(numeric_values, dtype=np.float64)
                entry.update({
                    "sum": round(float(array.sum()), 6),
                    "mean": round(float(array.mean()), 6),
                    "min": round(float(array.min()), 6),
                    "max": round(float(array.max()), 6),
                    "std": round(float(array.std()), 6),
                })
            counts = self._value_counts(non_null)
            entry["most_common"] = [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(),
                                           key=lambda kv: (-kv[1], kv[0]))[:5]
            ]
            aggregates[name] = entry

        note = ""
        if not matched:
            note = ("Soruda hiçbir sütun adı bulunamadı — sütun adlarını "
                    "birebir kullanmayı deneyin")
        elif not aggregates:
            note = "Eşleşen sütunlar için toplulaştırma üretilemedi"

        return {
            "question": str(question or ""),
            "matched_columns": matched,
            "aggregates": aggregates,
            "intent": self._question_intent(folded_question),
            "n_rows": parsed.n_rows,
            "n_cols": parsed.n_cols,
            "available_columns": list(parsed.headers),
            "note": note,
        }

    # ─── Kodlama ────────────────────────────────────────────────────────

    def encode(self, table: Any) -> EncodedChunk:
        """
        Tabloyu `EncodedChunk`'a kodlar.

        Args:
            table: Herhangi bir tablo girdisi

        Returns:
            `Modality.TABLE` tipli `EncodedChunk`

        Raises:
            TableDecodeError: Tablo ayrıştırılamazsa
        """
        parsed = self.parse(table)
        profile = self.profile(parsed)
        features = self._feature_vector(parsed, profile)
        vector = self._project(features, self._embedding_dim)

        confidence = 0.5
        if parsed.n_rows >= 3 and parsed.n_cols >= 2:
            confidence += 0.2
        if parsed.n_rows >= 20:
            confidence += 0.1
        if parsed.n_cols < 2:
            confidence -= 0.25
        if parsed.n_rows == 0:
            confidence = 0.1
        confidence = max(0.05, min(0.95, confidence))

        type_histogram = {bucket: 0 for bucket in TYPE_BUCKETS}
        for column in profile["columns"]:
            bucket = column["inferred_type"]
            type_histogram[bucket] = type_histogram.get(bucket, 0) + 1

        null_ratio = 0.0
        if parsed.n_rows and parsed.n_cols:
            total_cells = parsed.n_rows * parsed.n_cols
            null_cells = sum(1 for row in parsed.rows for cell in row
                             if cell is None or str(cell).strip() == "")
            null_ratio = null_cells / total_cells

        meta: Dict[str, Any] = {
            "n_rows": parsed.n_rows,
            "n_cols": parsed.n_cols,
            "headers": list(parsed.headers[:MAX_COLS]),
            "truncated_headers": parsed.n_cols > MAX_COLS,
            "dtypes": dict(parsed.dtypes),
            "type_histogram": type_histogram,
            "source": parsed.source,
            "null_ratio": round(null_ratio, 6),
            "numeric_columns": list(profile["numeric_columns"]),
            "categorical_columns": list(profile["categorical_columns"]),
            "id_columns": list(profile["id_columns"]),
            "date_columns": list(profile["date_columns"]),
            "embedding_dim": self._embedding_dim,
            "proj_seed": TABLE_PROJ_SEED,
            "proj_salt": TABLE_PROJ_SALT,
            "has_header": all(not h.startswith("col_") for h in parsed.headers),
        }
        attach_native(meta, features)

        chunk = EncodedChunk(
            modality=Modality.TABLE,
            vector=vector,
            confidence=round(confidence, 4),
            meta=meta,
        )
        self._stats["encoded"] += 1
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.table_encoded",
                {"dim": len(vector), "n_rows": parsed.n_rows,
                 "n_cols": parsed.n_cols, "confidence": chunk.confidence},
                source=self.name,
            )
        return chunk

    def _feature_vector(self, parsed: TableData, profile: Dict[str, Any]
                        ) -> "np.ndarray":
        """
        Ham 320 boyutlu özellik vektörünü üretir (sabit düzen — sütun sayısından
        bağımsız, `MAX_COLS` sütuna kadar doldurulur).

        Args:
            parsed: Ayrıştırılmış tablo
            profile: `profile()` çıktısı

        Returns:
            Şekil (NATIVE_FEATURE_DIM,) float64 dizi
        """
        # (1) Tip histogramı → 8
        type_counts = np.zeros((len(TYPE_BUCKETS),), dtype=np.float64)
        bucket_index = {bucket: i for i, bucket in enumerate(TYPE_BUCKETS)}
        for column in profile["columns"]:
            type_counts[bucket_index.get(column["inferred_type"], 4)] += 1.0
        if parsed.n_cols:
            type_counts /= float(parsed.n_cols)

        numeric_block = np.zeros((MAX_COLS * 6,), dtype=np.float64)
        shape_block = np.zeros((MAX_COLS * 5,), dtype=np.float64)
        header_block = np.zeros((MAX_COLS * HEADER_HASH_DIM,), dtype=np.float64)

        for index, column in enumerate(profile["columns"][:MAX_COLS]):
            values = [row[index] for row in parsed.rows] if index < parsed.n_cols else []
            non_null = [v for v in values if v is not None and str(v).strip() != ""]
            counts = self._value_counts(non_null)
            spread = 1.0
            if "min" in column and "max" in column:
                spread = max(1e-9, float(column["max"]) - float(column["min"]))
                numeric_block[index * 6 + 0] = column["null_ratio"]
                numeric_block[index * 6 + 1] = column["unique_ratio"]
                numeric_block[index * 6 + 2] = (float(column["mean"]) - float(column["min"])) / spread
                numeric_block[index * 6 + 3] = min(1.0, float(column["std"]) / spread)
                numeric_block[index * 6 + 4] = 0.0
                numeric_block[index * 6 + 5] = 1.0
            else:
                numeric_block[index * 6 + 0] = column["null_ratio"]
                numeric_block[index * 6 + 1] = column["unique_ratio"]

            mean_length = (sum(len(str(v)) for v in non_null) / len(non_null)) \
                if non_null else 0.0
            shape_block[index * 5 + 0] = column["null_ratio"]
            shape_block[index * 5 + 1] = column["unique_ratio"]
            shape_block[index * 5 + 2] = 1.0 if column["inferred_type"] == "id" else 0.0
            shape_block[index * 5 + 3] = min(1.0, mean_length / 64.0)
            shape_block[index * 5 + 4] = min(1.0, len(counts) / 100.0)

            header_vector = stable_hash_vector(str(column["name"]),
                                               HEADER_HASH_DIM, TABLE_HEADER_SALT)
            header_block[index * HEADER_HASH_DIM:(index + 1) * HEADER_HASH_DIM] = \
                np.asarray(header_vector, dtype=np.float64)

        # (5) Global ölçütler → 8
        total_cells = max(1, parsed.n_rows * parsed.n_cols)
        null_cells = sum(1 for row in parsed.rows for cell in row
                         if cell is None or str(cell).strip() == "")
        numeric_ratio = (len(profile["numeric_columns"]) / parsed.n_cols) \
            if parsed.n_cols else 0.0
        categorical_ratio = (len(profile["categorical_columns"]) / parsed.n_cols) \
            if parsed.n_cols else 0.0
        header_uniqueness = (len(set(parsed.headers)) / parsed.n_cols) \
            if parsed.n_cols else 0.0
        globals_block = np.asarray([
            min(1.0, math.log1p(parsed.n_rows) / math.log1p(100000.0)),
            min(1.0, parsed.n_cols / 64.0),
            null_cells / total_cells,
            numeric_ratio,
            categorical_ratio,
            min(1.0, math.log1p(total_cells) / math.log1p(1e6)),
            1.0 if all(not h.startswith("col_") for h in parsed.headers) else 0.0,
            header_uniqueness,
        ], dtype=np.float64)

        features = np.concatenate([
            type_counts, numeric_block, shape_block, header_block, globals_block,
        ]).astype(np.float64)
        if features.shape[0] != NATIVE_FEATURE_DIM:  # pragma: no cover - tutarlılık
            logger.debug("Tablo özellik boyutu beklenenden farklı: %d != %d",
                         features.shape[0], NATIVE_FEATURE_DIM)
        return features

    def _project(self, features: "np.ndarray", out_dim: int) -> List[float]:
        """
        Ham özellik vektörünü sabit tohumlu rastgele projeksiyonla `out_dim`'e taşır.

        Args:
            features: Ham özellik vektörü
            out_dim: Hedef gömme boyutu

        Returns:
            L2-normalize edilmiş gömme vektörü
        """
        native_dim = int(features.shape[0])
        key = (native_dim, int(out_dim))
        matrix = self._proj_cache.get(key)
        if matrix is None:
            matrix = projection_matrix(native_dim, int(out_dim),
                                       TABLE_PROJ_SEED, TABLE_PROJ_SALT)
            self._proj_cache[key] = matrix
        projected = features.astype(np.float64) @ matrix
        return l2_normalize(projected.tolist())

    # ─── Ayrıştırma Yardımcıları ────────────────────────────────────────

    @staticmethod
    def _sniff_delimiter(text: str) -> str:
        """
        Ayırıcıyı sezer: önce `csv.Sniffer`, sonra en sık görülen ayırıcı.

        Args:
            text: CSV içeriği

        Returns:
            Seçilen ayırıcı karakter (varsayılan virgül)
        """
        sample = "\n".join(text.splitlines()[:20])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            pass
        best, best_count = ",", 0
        for candidate in (",", ";", "\t", "|"):
            counts = [line.count(candidate) for line in sample.splitlines()[:10]
                      if line.strip()]
            if not counts or len(set(counts)) != 1:
                continue
            if counts[0] > best_count:
                best, best_count = candidate, counts[0]
        return best

    @staticmethod
    def _json_loads(text: str) -> Any:
        """JSON metnini ayrıştırır (Türkçe mesajlı hata ile)."""
        try:
            return json.loads(text)
        except ValueError as exc:
            raise TableDecodeError(f"JSON ayrıştırılamadı: {exc}") from exc

    @staticmethod
    def _normalize_cell(cell: Any) -> Any:
        """Boş hücreleri None'a çevirir, diğerlerini string'e indirger."""
        if cell is None:
            return None
        if isinstance(cell, str):
            stripped = cell.strip()
            return stripped if stripped else None
        if isinstance(cell, float) and math.isnan(cell):
            return None
        return cell

    @staticmethod
    def _row_is_header(row: Sequence[Any]) -> bool:
        """İlk satırın başlık olup olmadığını sezer (tümü sayısal değilse başlık)."""
        cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
        if not cells:
            return False
        numeric = sum(1 for cell in cells if TableEncoder._is_numberish(cell))
        return numeric < len(cells)

    @staticmethod
    def _is_numberish(value: str) -> bool:
        """Metnin sayı görünümlü olup olmadığını kontrol eder."""
        return bool(_INT_RE.match(value) or _FLOAT_RE.match(value))

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        """Metnin markdown tablo içerip içermediğini sezer."""
        if "|" not in text:
            return False
        lines = text.splitlines()
        for index in range(len(lines) - 1):
            if "|" in lines[index] and TableEncoder._is_separator_row(lines[index + 1]):
                return True
        return False

    @staticmethod
    def _is_separator_row(line: str) -> bool:
        """Markdown ayırıcı satırı mı (|---|---|)?"""
        if "|" not in line or "-" not in line:
            return False
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            return False
        return all(_MD_SEPARATOR_CELL_RE.match(cell) for cell in cells if cell != "")

    @staticmethod
    def _split_markdown_row(line: str) -> List[str]:
        """Markdown satırını hücrelere ayırır (kaçışlı `\\|` desteklenir)."""
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = re.split(r"(?<!\\)\|", stripped)
        return [cell.replace("\\|", "|").strip() for cell in cells]

    @staticmethod
    def _escape_markdown_cell(value: Any) -> str:
        """Markdown hücresi için `|` ve satır sonlarını kaçırır."""
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text

    @staticmethod
    def _read_text(path: Path) -> str:
        """Metin dosyasını (utf-8 → cp1254 → latin-1 sırasıyla) okur."""
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise TableDecodeError(f"Dosya okunamadı ({path}): {exc}") from exc
        raise TableDecodeError(f"Dosya hiçbir kodlamayla okunamadı: {path}")

    @staticmethod
    def _decode_text(data: bytes) -> str:
        """Baytları metne çevirir (utf-8 → cp1254 → latin-1)."""
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    # ─── Tip Çıkarımı ve İstatistik ─────────────────────────────────────

    def _infer_column_type(self, name: str, values: Sequence[Any]) -> str:
        """
        Bir sütunun tipini çıkarır.

        Args:
            name: Sütun adı (id sezgisi için kullanılır)
            values: Hücre değerleri

        Returns:
            "int" | "float" | "bool" | "date" | "str" | "category" | "id" | "empty"
        """
        non_null = [v for v in values if v is not None and str(v).strip() != ""]
        if not non_null:
            return "empty"

        texts = [str(v).strip() for v in non_null]
        unique_ratio = len(set(texts)) / len(texts)
        n_unique = len(set(texts))
        mean_length = sum(len(text) for text in texts) / len(texts)
        lowered = {text.lower() for text in texts}

        if lowered <= _BOOL_VALUES:
            return "bool"
        if all(_INT_RE.match(text) for text in texts):
            if unique_ratio == 1.0 and self._looks_like_id(name):
                return "id"
            return "int"
        if all(_FLOAT_RE.match(text) for text in texts):
            if unique_ratio == 1.0 and self._looks_like_id(name):
                return "id"
            return "float"
        if all(_DATE_RE.match(text) for text in texts):
            return "date"
        if unique_ratio == 1.0 and self._looks_like_id(name):
            return "id"
        # Kategorik sezgisi: düşük kardinalite + kısa değerler.
        # (Tekrarlı değerler her durumda kategori; tekrarsız değerler yalnızca
        #  küçük tablolarda ve kısa metinlerde kategori sayılır.)
        if n_unique <= 20 and mean_length <= 40:
            if n_unique < len(texts):
                return "category"
            if len(texts) <= 20 and n_unique <= 12 and mean_length <= 24:
                return "category"
        return "str"

    @staticmethod
    def _looks_like_id(name: str) -> bool:
        """Sütun adının kimlik (id) sütununa benzediğini sezer."""
        lowered = str(name).strip().lower()
        if lowered in ("id", "no", "kod", "code", "uuid", "key", "ref"):
            return True
        return any(lowered.endswith(f"_{hint}") or lowered.startswith(f"{hint}_")
                   or lowered == hint for hint in _ID_NAME_HINTS)

    @staticmethod
    def _is_numeric_column(values: Sequence[Any]) -> bool:
        """Değerlerin en az %90'ı sayıya çevrilebiliyorsa True."""
        if not values:
            return False
        sample = list(values[:200])
        converted = sum(1 for value in sample if TableEncoder._to_float(value) is not None)
        return converted >= max(1, int(0.9 * len(sample)))

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """
        Hücreyi float'a çevirir (Türkçe ondalık virgülü desteklenir).

        Args:
            value: Hücre değeri

        Returns:
            float veya çevrilemiyorsa None
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            try:
                number = float(value)
                return number if math.isfinite(number) else None
            except (TypeError, ValueError):
                return None
        text = str(value).strip()
        if not text:
            return None
        candidate = text.replace(" ", "").replace("₺", "").replace("$", "") \
            .replace("%", "").replace("€", "")
        if "," in candidate and "." in candidate:
            if candidate.rfind(",") > candidate.rfind("."):
                candidate = candidate.replace(".", "").replace(",", ".")
            else:
                candidate = candidate.replace(",", "")
        elif "," in candidate:
            candidate = candidate.replace(",", ".")
        try:
            number = float(candidate)
            return number if math.isfinite(number) else None
        except ValueError:
            return None

    @staticmethod
    def _value_counts(values: Sequence[Any]) -> Dict[str, int]:
        """Değerlerin görülme sayılarını (string anahtarlarla) döndürür."""
        counts: Dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _fold(text: str) -> str:
        """Türkçe karakterleri katlar ve küçük harfe indirger (eşleştirme için)."""
        return str(text).translate(_FOLD_MAP).lower()

    @staticmethod
    def _question_intent(folded_question: str) -> List[str]:
        """Sorudaki toplulaştırma niyetini (naif anahtar kelime) sezer."""
        intents: List[str] = []
        keywords = {
            "count": ("kaç", "kac", "adet", "sayi", "sayısı", "count", "how many"),
            "sum": ("toplam", "sum", "total", "genel toplam"),
            "mean": ("ortalama", "average", "mean", "avg"),
            "max": ("maks", "maksimum", "en yüksek", "en buyuk", "en büyük", "max"),
            "min": ("min", "minimum", "en düşük", "en dusuk", "en küçük", "min"),
            "most_common": ("en çok", "en cok", "most", "sıklık", "frekans", "hangi"),
            "distribution": ("dağılım", "dagilim", "distribution", "grafik", "histogram"),
        }
        for intent, words in keywords.items():
            if any(word in folded_question for word in words):
                intents.append(intent)
        return intents

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    @staticmethod
    def _is_dataframe_like(payload: Any) -> bool:
        """pandas DataFrame benzeri nesne sezgisi."""
        return (hasattr(payload, "columns") and hasattr(payload, "values")
                and hasattr(payload, "shape"))

    @staticmethod
    def _probe_module(name: str) -> bool:
        """Modül kurulu mu (import etmeden, ucuz kontrol)?"""
        try:
            import importlib.util

            return importlib.util.find_spec(name) is not None
        except Exception:  # pragma: no cover - savunma
            return False

    @staticmethod
    def _read_embedding_dim(config: Any) -> int:
        """`config.neural.embedding_dim` değerini güvenle okur."""
        if config is None:
            return DEFAULT_EMBEDDING_DIM
        try:
            return int(config.neural.embedding_dim)
        except Exception:
            return DEFAULT_EMBEDDING_DIM

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Ayrıştırma/kodlama sayaçları, yetenek bayrakları, önbellek durumu
        """
        parsed = max(1, self._stats["parsed"])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "native_feature_dim": NATIVE_FEATURE_DIM,
            "max_cols": MAX_COLS,
            "openpyxl_available": self._openpyxl_available,
            "pandas_available": self._pandas_available,
            "counters": dict(self._stats),
            "source_distribution": dict(self._format_counts),
            "avg_rows_per_table": round(self._stats["rows_processed"] / parsed, 1),
            "avg_cols_per_table": round(self._stats["cols_processed"] / parsed, 1),
            "projection_cache_entries": len(self._proj_cache),
            "proj_seed": TABLE_PROJ_SEED,
            "proj_salt": TABLE_PROJ_SALT,
        }


__all__ = ["TableEncoder", "NATIVE_FEATURE_DIM", "TYPE_BUCKETS", "MAX_COLS",
           "TABLE_PROJ_SEED", "TABLE_PROJ_SALT", "TABLE_HEADER_SALT"]
